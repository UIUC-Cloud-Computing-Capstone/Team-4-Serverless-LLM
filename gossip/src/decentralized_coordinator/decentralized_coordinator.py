"""
Decentralized Coordinator Node for LLM Inference Scheduling.

This node acts as a worker, a gossip participant, and a scheduler simultaneously,
creating a fully decentralized network.

It uses composition to combine logic from:
- udpnode.Node: For gossip-based state dissemination.
- worker_node.WorkerNode: For managing models and local worker state.

It now contains its own scheduling logic, identical to the
CentralCoordinator for a fair comparison, but without the direct dependency.
"""

import socket
import threading
import pickle
import time
import random
import logging
import argparse
import signal
from typing import Dict, Tuple, Optional, Any, List

# Import from existing project files
from udpnode import Node
from worker_node import WorkerNode
from contracts import (
    WorkerLoadReport,
    ScheduleRequest,
    ScheduleResponse,
    PlacementAction
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(threadName)s] %(message)s')


class DecentralizedNode:
    """
    A single node that is a worker, gossip peer, and scheduler.
    """

    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        peers: Optional[List[Tuple[str, int]]] = None,
        verbose: bool = True,
        use_real_models: bool = False
    ):
        """
        Initialize the decentralized node.
        ... (args) ...
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.verbose = verbose
        self.logger = logging.getLogger(f"DecentralizedNode-{node_id}")

        # 1. Gossip Component (from udpnode.py)
        # This handles peer discovery and state replication.
        self.gossip = Node(
            node_id=node_id,
            host=host,
            port=port,
            verbose=verbose
        )
        # Add initial peers
        if peers:
            for i, (peer_host, peer_port) in enumerate(peers):
                peer_id = f"peer-{i}-{peer_host}:{peer_port}"
                self.gossip.add_peer(peer_id, peer_host, peer_port)

        # 2. Worker Component (from worker_node.py)
        # This handles local model loading and state tracking.
        self.worker = WorkerNode(
            node_id=node_id,
            host=host,
            port=port + 1,  # Use a different port (or just don't start its listener)
            coordinator_host="dummy",
            coordinator_port=0,
            report_interval=1.0,
            verbose=verbose,
            use_real_models=use_real_models
        )

        # 3. Scheduler Component (LOGIC MOVED FROM CentralCoordinator)
        # This node now runs its own scheduling logic on its local view of the cluster.
        self.worker_timeout = 10.0
        self.queue_threshold = 10
        self.memory_threshold = 0.9

        # Global state view: {worker_id: (WorkerLoadReport, last_update_time)}
        self.worker_states: Dict[str, Tuple[WorkerLoadReport, float]] = {}
        self.state_lock = threading.RLock()

        # Statistics
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Threading and State
        self.is_running = True
        self._listener_thread = None
        self._gossiper_thread = None
        self._reporter_thread = None
        self._updater_thread = None

        # State for handling schedule request/response
        self._pending_requests: Dict[str, threading.Event] = {}
        self._request_responses: Dict[str, Any] = {}
        self._request_lock = threading.Lock()

        if self.verbose:
            self.logger.info(f"DecentralizedNode {node_id} initialized at {host}:{port}")

    def initialize_models(self, models_to_load: List[str]):
        """
        Initialize and load models into the worker component.
        """
        self.worker.initialize(models_to_load)

    # --- Core Threading Methods ---

    def _listener(self):
        """
        Main listener thread.
        Handles both gossip messages and schedule requests/responses.
        """
        if self.verbose:
            self.logger.info(f"{self.node_id} main listener started.")
        
        sock = self.gossip.sock  # Use the gossip node's socket

        while self.is_running:
            try:
                raw_data, sender_addr = sock.recvfrom(8192)
                message = pickle.loads(raw_data)
                
                msg_type = message.get('type')

                if msg_type == 'schedule_request':
                    # This is a request for scheduling from another node
                    if self.verbose:
                        self.logger.debug(f"Received schedule_request from {sender_addr}")
                    self._handle_schedule_request(message, sender_addr)
                
                elif msg_type == 'schedule_response':
                    # This is a response to our own request
                    if self.verbose:
                        self.logger.debug(f"Received schedule_response from {sender_addr}")
                    self._handle_schedule_response(message)
                
                elif msg_type == 'schedule_error':
                    # This is an error response to our own request
                    if self.verbose:
                        self.logger.warning(f"Received schedule_error from {sender_addr}")
                    self._handle_schedule_response(message) # Handle like a response

                else:
                    # Assume it's a gossip message
                    self.gossip._handle_gossip(message)

            except (socket.error, pickle.UnpicklingError, EOFError) as e:
                if self.is_running:
                    self.logger.error(
                        f"{self.node_id} listener error: {e}",
                        exc_info=True
                    )
                break

        if self.verbose:
            self.logger.info(f"{self.node_id} listener stopped.")

    def _state_reporter(self):
        """
        Periodically reports this node's worker state to the gossip network.
        Replaces WorkerNode's _reporter thread.
        """
        if self.verbose:
            self.logger.info(f"{self.node_id} state reporter started.")
        
        while self.is_running:
            try:
                # Get the latest report from our internal worker
                report = WorkerLoadReport(
                    node_id=self.worker.node_id,
                    loaded_models=self.worker.get_loaded_models(),
                    queue_depth=self.worker.queue_depth,
                    memory_utilization=self.worker.memory_utilization,
                    is_ready=self.worker.is_ready,
                    timestamp=time.time()
                )
                
                # Set this report as our data in the gossip protocol
                self.gossip.set_data(self.node_id, report.to_dict())
                
                if self.verbose:
                    self.logger.debug(f"Gossiped local state: queue={report.queue_depth}")

            except Exception as e:
                self.logger.error(f"Error in state reporter: {e}", exc_info=True)
            
            time.sleep(self.worker.report_interval)

        if self.verbose:
            self.logger.info(f"{self.node_id} state reporter stopped.")

    def _scheduler_updater(self):
        """
        Periodically updates this node's local view of the cluster
        based on the latest data from the gossip network.
        """
        if self.verbose:
            self.logger.info(f"{self.node_id} scheduler updater started.")
        
        while self.is_running:
            try:
                # Get the global state from the gossip component
                gossip_data = self.gossip.get_data()
                
                # Format this data
                # gossip_data = {'node_id': (report_dict, ts), ...}
                
                new_worker_states: Dict[str, Tuple[WorkerLoadReport, float]] = {}
                current_time = time.time()
                
                for node_id, (report_dict, ts) in gossip_data.items():
                    try:
                        report_obj = WorkerLoadReport.from_dict(report_dict)
                        # Use the gossip timestamp as the last_seen time
                        last_seen = self.gossip.peers.get(node_id, {}).get("last_seen", current_time)
                        new_worker_states[node_id] = (report_obj, last_seen)
                    except Exception as e:
                        self.logger.warning(f"Could not parse report from {node_id}: {e}")
                
                # Atomically update this node's scheduler state
                with self.state_lock:
                    self.worker_states = new_worker_states
                    # Also clean up dead workers
                    self._cleanup_dead_workers()

                if self.verbose:
                    active_count = len(self.worker_states)
                    self.logger.debug(f"Scheduler view updated with {active_count} active nodes.")
                    
            except Exception as e:
                self.logger.error(f"Error in scheduler updater: {e}", exc_info=True)
            
            time.sleep(1.0) # Update scheduler view every second

        if self.verbose:
            self.logger.info(f"{self.node_id} scheduler updater stopped.")

    # --- Scheduling Request Handling Methods ---

    def _handle_schedule_request(self, message: Dict, sender_addr: Tuple[str, int]):
        """
        Handle an incoming schedule request from a peer.
        """
        try:
            request = ScheduleRequest.from_dict(message['payload'])
            
            # Use our local schedule() method to make a placement decision.
            response = self.schedule(request)
            
            response_message = {
                'type': 'schedule_response',
                'request_id': request.request_id, # Include original request_id
                'payload': response.to_dict()
            }
            
            serialized_response = pickle.dumps(response_message)
            self.gossip.sock.sendto(serialized_response, sender_addr)

            if self.verbose:
                self.logger.info(
                    f"Scheduled {request.request_id} to {response.worker_id} "
                    f"(action: {response.action.value}) for peer {sender_addr}"
                )

        except RuntimeError as e:
            # No workers available
            error_message = {
                'type': 'schedule_error',
                'request_id': message['payload'].get('request_id'),
                'payload': {'error': str(e)}
            }
            serialized_error = pickle.dumps(error_message)
            self.gossip.sock.sendto(serialized_error, sender_addr)
            self.logger.error(f"Schedule error for peer {sender_addr}: {e}")

        except Exception as e:
            self.logger.error(f"Error handling schedule request: {e}", exc_info=True)

    def _handle_schedule_response(self, message: Dict):
        """
        Handle a schedule response (or error) to one of our requests.
        """
        request_id = message.get('request_id')
        if not request_id:
            return

        with self._request_lock:
            if request_id in self._pending_requests:
                # Store the response
                if message.get('type') == 'schedule_response':
                    response_obj = ScheduleResponse.from_dict(message['payload'])
                    self._request_responses[request_id] = response_obj
                else: # 'schedule_error'
                    self._request_responses[request_id] = message['payload']

                # Notify the waiting thread
                event = self._pending_requests.pop(request_id, None)
                if event:
                    event.set()
            else:
                if self.verbose:
                    self.logger.warning(f"Received unexpected response for {request_id}")

    def request_schedule(self, request_id: str, model_required: str, timeout: float = 2.0) -> Optional[ScheduleResponse]:
        """
        Send a schedule request to a random peer and wait for a response.
        (Replaces WorkerNode's request_schedule)
        """
        
        # 1. Get a random peer to send the request to
        peers = self.gossip.get_peers()
        if not peers:
            self.logger.error("No peers to send schedule request to.")
            return None
        
        random_peer_id = random.choice(list(peers.keys()))
        peer_info = peers[random_peer_id]
        peer_addr = (peer_info['host'], peer_info['port'])
        
        request = ScheduleRequest(
            request_id=request_id,
            model_required=model_required
        )
        message = {
            'type': 'schedule_request',
            'payload': request.to_dict()
        }
        
        # 2. Set up event to wait for response
        event = threading.Event()
        with self._request_lock:
            self._pending_requests[request_id] = event
            # Clear any stale response
            if request_id in self._request_responses:
                del self._request_responses[request_id]
        
        try:
            # 3. Send the request
            serialized = pickle.dumps(message)
            self.gossip.sock.sendto(serialized, peer_addr)
            
            if self.verbose:
                self.logger.info(f"Sent schedule_request {request_id} to peer {random_peer_id}")

            # 4. Wait for the response
            if not event.wait(timeout):
                # Timeout
                self.logger.warning(f"Timeout waiting for schedule response for {request_id}")
                return None
            
            # 5. Get response
            with self._request_lock:
                response = self._request_responses.pop(request_id, None)

            if isinstance(response, ScheduleResponse):
                return response
            elif isinstance(response, dict): # Error
                self.logger.error(f"Received schedule error for {request_id}: {response.get('error')}")
                return None
            else:
                self.logger.error(f"Invalid response object for {request_id}")
                return None

        except Exception as e:
            self.logger.error(f"Error requesting schedule: {e}", exc_info=True)
            return None
        
        finally:
            # Clean up pending request
            with self._request_lock:
                self._pending_requests.pop(request_id, None)

    # --- Start/Stop Methods ---

    def start(self):
        """Start all component threads."""
        self.is_running = True
        
        # Start gossip listener (handles all incoming UDP)
        self._listener_thread = threading.Thread(
            target=self._listener,
            name=f"{self.node_id}-MainListener"
        )
        self._listener_thread.start()
        
        # Start gossip sender
        self._gossiper_thread = threading.Thread(
            target=self.gossip._gossiper,
            args=(1,), # gossip_interval
            name=f"{self.node_id}-Gossiper"
        )
        self._gossiper_thread.start()
        
        # Start worker state reporter
        self._reporter_thread = threading.Thread(
            target=self._state_reporter,
            name=f"{self.node_id}-Reporter"
        )
        self._reporter_thread.start()

        # Start scheduler updater
        self._updater_thread = threading.Thread(
            target=self._scheduler_updater,
            name=f"{self.node_id}-SchedulerUpdater"
        )
        self._updater_thread.start()
        
        self.logger.info(f"{self.node_id} started all threads.")

    def stop(self):
        """Stop all component threads gracefully."""
        if not self.is_running:
            return

        self.is_running = False
        
        # Stop gossip components
        self.gossip.stop() # This closes the socket, stopping the listener

        # Wait for threads to join
        if self._listener_thread:
            self._listener_thread.join()
        if self._gossiper_thread:
            self._gossiper_thread.join()
        if self.reporte_thread:
            self.reporte_thread.join()
        if self._updater_thread:
            self._updater_thread.join()
            
        # Stop internal worker (just in case)
        self.worker.stop()

        if self.verbose:
            self.logger.info(f"{self.node_id} stopped.")

    # --- SCHEDULING LOGIC (MOVED FROM CentralCoordinator) ---

    def _cleanup_dead_workers(self) -> None:
        """Remove workers that have not reported in a while."""
        # Note: This is called inside _scheduler_updater, which already holds the lock
        with self.state_lock:
            current_time = time.time()
            dead_workers = [
                worker_id
                for worker_id, (_, last_seen) in self.worker_states.items()
                if current_time - last_seen > self.worker_timeout
            ]
            for worker_id in dead_workers:
                del self.worker_states[worker_id]
                self.logger.warning(f"Removed dead worker from view: {worker_id}")

    def _get_active_workers(self) -> Dict[str, WorkerLoadReport]:
        """
        Get all active workers (recently updated).
        Also cleans up dead workers first.
        """
        with self.state_lock:
            self._cleanup_dead_workers()
            return {
                worker_id: report
                for worker_id, (report, _) in self.worker_states.items()
            }

    def _calculate_worker_score(
        self,
        report: WorkerLoadReport,
        has_model: bool
    ) -> float:
        """
        Calculate a score for a worker (higher is better).
        (Copied from CentralCoordinator)
        """
        # Base score heavily favors workers with the model loaded
        score = 1000.0 if has_model else 0.0

        # Penalize based on queue depth (normalized)
        queue_penalty = report.queue_depth * 10
        score -= queue_penalty

        # Penalize based on memory utilization (normalized)
        memory_penalty = report.memory_utilization * 100
        score -= memory_penalty

        return score

    def schedule(self, request: ScheduleRequest) -> ScheduleResponse:
        """
        Find the optimal worker to handle an inference request.
        (Copied from CentralCoordinator)
        """
        self.total_requests += 1
        active_workers = self._get_active_workers()

        if not active_workers:
            self.logger.error(f"No active workers in view for request {request.request_id}")
            raise RuntimeError("No active workers available in local view")

        # Find workers with the required model
        workers_with_model: List[Tuple[str, WorkerLoadReport, float]] = []
        workers_without_model: List[Tuple[str, WorkerLoadReport, float]] = []

        for worker_id, report in active_workers.items():
            has_model = request.model_required in report.loaded_models
            score = self._calculate_worker_score(report, has_model)

            if has_model:
                workers_with_model.append((worker_id, report, score))
            else:
                workers_without_model.append((worker_id, report, score))

        # Decision logic
        if workers_with_model:
            # Cache hit: select best worker with the model
            self.cache_hits += 1
            workers_with_model.sort(key=lambda x: x[2], reverse=True)
            best_worker_id, best_report, _ = workers_with_model[0]

            estimated_wait = best_report.queue_depth * 0.5  # Rough estimate
            reason = (
                f"Worker has model {request.model_required} loaded. "
                f"Queue: {best_report.queue_depth}, Memory: {best_report.memory_utilization:.2f}"
            )

            self.logger.info(
                f"Scheduled request {request.request_id} to {best_worker_id} (SERVE)"
            )

            return ScheduleResponse(
                worker_id=best_worker_id,
                action=PlacementAction.SERVE,
                estimated_wait_time=estimated_wait,
                reason=reason
            )
        else:
            # Cache miss: cold start on least loaded worker
            self.cache_misses += 1
            workers_without_model.sort(key=lambda x: x[2], reverse=True)
            best_worker_id, best_report, _ = workers_without_model[0]

            estimated_wait = 10.0 + (best_report.queue_depth * 0.5)  # Cold start penalty
            reason = (
                f"No worker has model {request.model_required}. "
                f"Cold starting on {best_worker_id}. "
                f"Queue: {best_report.queue_depth}, Memory: {best_report.memory_utilization:.2f}"
            )

            self.logger.info(
                f"Scheduled request {request.request_id} to {best_worker_id} (COLD_START)"
            )

            return ScheduleResponse(
                worker_id=best_worker_id,
                action=PlacementAction.COLD_START,
                estimated_wait_time=estimated_wait,
                reason=reason
            )

    # --- STATE & DEBUGGING (MOVED FROM CentralCoordinator) ---

    def get_cluster_state(self) -> Dict:
        """
        Get a summary of the current cluster state *from this node's perspective*.
        """
        active_workers = self._get_active_workers()

        total_queue = sum(w.queue_depth for w in active_workers.values())
        avg_memory = (
            sum(w.memory_utilization for w in active_workers.values()) / len(active_workers)
            if active_workers
            else 0.0
        )

        # Count ready workers
        ready_workers = sum(1 for w in active_workers.values() if w.is_ready)

        return {
            "node_id": self.node_id, # Changed from coordinator_id
            "num_active_workers": len(active_workers),
            "num_ready_workers": ready_workers,
            "total_requests_processed": self.total_requests,
            "cache_hit_rate": self.cache_hits / self.total_requests if self.total_requests > 0 else 0.0,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_queue_depth": total_queue,
            "average_memory_utilization": avg_memory,
            "workers": {
                worker_id: report.to_dict()
                for worker_id, report in active_workers.items()
            }
        }

    def print_cluster_state(self):
        """Print a human-readable summary of this node's view of the cluster state."""
        state = self.get_cluster_state()
        print(f"\n--- Cluster State (View from Node: {state['node_id']}) ---")
        print(f"Active Workers in View: {state['num_active_workers']}")
        print(f"Ready Workers in View: {state['num_ready_workers']}")
        print(f"Total Requests Processed (by this node): {state['total_requests_processed']}")
        print(f"Cache Hit Rate (this node): {state['cache_hit_rate']:.2%}")
        print(f"Total Queue Depth (in view): {state['total_queue_depth']}")
        print(f"Average Memory Utilization (in view): {state['average_memory_utilization']:.2%}")
        print("\nWorker States (in view):")
        for worker_id, worker_data in state['workers'].items():
            ready_status = "✓" if worker_data.get('is_ready', False) else "✗"
            print(f"  {worker_id} [{ready_status}]: models={worker_data['loaded_models']}, "
                  f"queue={worker_data['queue_depth']}, "
                  f"memory={worker_data['memory_utilization']:.2f}")


def main():
    """Main entry point for running a decentralized node."""
    parser = argparse.ArgumentParser(description="Decentralized Worker/Scheduler Node")
    parser.add_argument("--node-id", required=True, help="Unique worker ID")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, required=True, help="Port to bind to")
    parser.add_argument("--peers", help="Comma-separated list of peer addresses, e.g., 127.0.0.1:9001,127.0.0.1:9002")
    parser.add_argument("--load-models", help="Comma-separated list of models to preload, e.g., opt-1.3b,opt-2.7b")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Parse peers
    peer_list = []
    if args.peers:
        for addr in args.peers.split(','):
            try:
                host, port = addr.split(':')
                peer_list.append((host, int(port)))
            except ValueError:
                print(f"Invalid peer address format: {addr}. Ignoring.")

    # Parse models to load
    models_to_load = []
    if args.load_models:
        models_to_load = args.load_models.split(',')

    node = DecentralizedNode(
        node_id=args.node_id,
        host=args.host,
        port=args.port,
        peers=peer_list,
        verbose=args.verbose,
        use_real_models=False # Set to True for real model loading
    )

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print(f"\n\nShutting down node {args.node_id}...")
        node.print_cluster_state()
        node.stop()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Initialize and start
    if models_to_load:
        print(f"Node {args.node_id} pre-loading models: {models_to_load}...")
        node.initialize_models(models_to_load)
    
    node.start()
    print(f"DecentralizedNode {args.node_id} running on {args.host}:{args.port}")
    print(f"Peers: {peer_list}")
    print("Press Ctrl+C to stop")

    # Keep main thread alive
    try:
        while True:
            time.sleep(10)
            if args.verbose:
                node.print_cluster_state()
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()
