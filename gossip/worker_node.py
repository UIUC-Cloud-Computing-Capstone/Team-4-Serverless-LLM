"""
Worker node that communicates with the central coordinator.

Workers maintain their local state and periodically report to the central coordinator.
They can also request scheduling decisions from the central coordinator.
"""

import socket
import threading
import pickle
import time
import random
import logging
from typing import List, Optional
from contracts import WorkerLoadReport, ScheduleRequest, ScheduleResponse
from model_loader import ModelLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(threadName)s] %(message)s')


class WorkerNode:
    """
    Worker node in the central coordinator architecture.

    Maintains local state (loaded models, queue depth, GPU utilization)
    and periodically reports to the central coordinator.
    """

    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        coordinator_host: str,
        coordinator_port: int,
        report_interval: float = 1.0,
        verbose: bool = True,
        use_real_models: bool = False,
        gcs_bucket: str = "remote_model",
        cache_dir: str = "/tmp/model_cache",
        device: str = "cpu"
    ):
        """
        Initialize a worker node.

        Args:
            node_id: Unique identifier for this worker
            host: Host address for this worker
            port: Port for this worker to listen on
            coordinator_host: Coordinator's host address
            coordinator_port: Coordinator's port
            report_interval: Seconds between load reports to coordinator
            verbose: Enable detailed logging
            use_real_models: If True, use actual PyTorch model loading from GCS
            gcs_bucket: GCS bucket name for model storage
            cache_dir: Local cache directory for models
            device: PyTorch device ('cpu', 'cuda', etc.)
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.report_interval = report_interval
        self.verbose = verbose
        self.use_real_models = use_real_models

        # Worker state
        self.queue_depth: int = 0
        self.gpu_utilization: float = 0.0
        self.state_lock = threading.RLock()

        # Model loader
        if use_real_models:
            self.model_loader = ModelLoader(
                gcs_bucket=gcs_bucket,
                cache_dir=cache_dir,
                device=device
            )
        else:
            self.model_loader = None
            self._simulated_models: List[str] = []  # For simulation mode

        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))

        # Threading
        self.is_running = True
        self._reporter_thread = None
        self._listener_thread = None

        self.logger = logging.getLogger(f"Worker-{node_id}")
        if self.verbose:
            self.logger.info(
                f"Worker {node_id} initialized at {host}:{port}, "
                f"coordinator at {coordinator_host}:{coordinator_port}"
            )
            if use_real_models:
                self.logger.info(f"Using real model loading from GCS bucket: {gcs_bucket}")

    def _reporter(self):
        """Periodically send load reports to the central coordinator."""
        if self.verbose:
            self.logger.info(f"{self.node_id} reporter started")

        while self.is_running:
            time.sleep(self.report_interval)

            try:
                with self.state_lock:
                    report = WorkerLoadReport(
                        node_id=self.node_id,
                        loaded_models=self.get_loaded_models(),
                        queue_depth=self.queue_depth,
                        gpu_utilization=self.gpu_utilization,
                        timestamp=time.time()
                    )

                message = {
                    'type': 'worker_report',
                    'payload': report.to_dict()
                }

                serialized = pickle.dumps(message)
                self.sock.sendto(
                    serialized,
                    (self.coordinator_host, self.coordinator_port)
                )

                if self.verbose:
                    self.logger.debug(
                        f"Sent report: models={len(self.loaded_models)}, "
                        f"queue={self.queue_depth}, gpu={self.gpu_utilization:.2f}"
                    )

            except Exception as e:
                self.logger.error(f"Error sending report: {e}")

        if self.verbose:
            self.logger.info(f"{self.node_id} reporter stopped")

    def _listener(self):
        """Listen for incoming messages (e.g., from central coordinator or other workers)."""
        if self.verbose:
            self.logger.info(f"{self.node_id} listener started")

        while self.is_running:
            try:
                raw_data, _ = self.sock.recvfrom(8192)
                message = pickle.loads(raw_data)
                # Can handle schedule responses or other messages here
                self.logger.debug(f"Received message: {message.get('type')}")

            except socket.timeout:
                # Timeout is normal when socket has timeout set for schedule requests
                continue
            except (socket.error, pickle.UnpicklingError, EOFError):
                if self.is_running:
                    self.logger.error(f"{self.node_id} listener error", exc_info=True)
                break

        if self.verbose:
            self.logger.info(f"{self.node_id} listener stopped")

    def request_schedule(self, request_id: str, model_required: str, timeout: float = 2.0) -> Optional[ScheduleResponse]:
        """
        Send a schedule request to the central coordinator and wait for response.

        Args:
            request_id: Unique identifier for the request
            model_required: Model needed for inference
            timeout: Timeout for waiting for response (seconds)

        Returns:
            ScheduleResponse if successful, None if timeout
        """
        request = ScheduleRequest(
            request_id=request_id,
            model_required=model_required
        )

        message = {
            'type': 'schedule_request',
            'payload': request.to_dict()
        }

        try:
            # Set socket timeout for receiving response
            self.sock.settimeout(timeout)

            # Send request
            serialized = pickle.dumps(message)
            self.sock.sendto(
                serialized,
                (self.coordinator_host, self.coordinator_port)
            )

            # Wait for response
            raw_data, _ = self.sock.recvfrom(8192)
            response_msg = pickle.loads(raw_data)

            # Reset socket to non-blocking
            self.sock.settimeout(None)

            if response_msg.get('type') == 'schedule_response':
                response = ScheduleResponse.from_dict(response_msg['payload'])
                self.logger.info(
                    f"Schedule response for {request_id}: "
                    f"worker={response.worker_id}, action={response.action.value}"
                )
                return response
            elif response_msg.get('type') == 'schedule_error':
                self.logger.error(
                    f"Schedule error: {response_msg['payload'].get('error')}"
                )
                return None

        except socket.timeout:
            self.logger.warning(f"Timeout waiting for schedule response for {request_id}")
            self.sock.settimeout(None)
            return None

        except Exception as e:
            self.logger.error(f"Error requesting schedule: {e}", exc_info=True)
            self.sock.settimeout(None)
            return None

    # State manipulation methods
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        if self.use_real_models:
            return self.model_loader.get_loaded_models()
        else:
            return self._simulated_models.copy()

    def load_model(self, model_id: str) -> bool:
        """
        Load a model (real or simulated).

        Args:
            model_id: Model identifier (e.g., 'opt-1.3b', 'opt-2.7b')

        Returns:
            True if successful, False otherwise
        """
        with self.state_lock:
            if self.use_real_models:
                success = self.model_loader.load_model(model_id)
                if success:
                    self.logger.info(f"Loaded real model: {model_id}")
                else:
                    self.logger.error(f"Failed to load real model: {model_id}")
                return success
            else:
                # Simulated mode
                if model_id not in self._simulated_models:
                    self._simulated_models.append(model_id)
                    self.logger.info(f"Loaded simulated model: {model_id}")
                return True

    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model (real or simulated).

        Args:
            model_id: Model identifier

        Returns:
            True if successful, False otherwise
        """
        with self.state_lock:
            if self.use_real_models:
                success = self.model_loader.unload_model(model_id)
                if success:
                    self.logger.info(f"Unloaded real model: {model_id}")
                else:
                    self.logger.error(f"Failed to unload real model: {model_id}")
                return success
            else:
                # Simulated mode
                if model_id in self._simulated_models:
                    self._simulated_models.remove(model_id)
                    self.logger.info(f"Unloaded simulated model: {model_id}")
                return True

    def set_queue_depth(self, depth: int):
        """Set the current queue depth."""
        with self.state_lock:
            self.queue_depth = depth

    def set_gpu_utilization(self, utilization: float):
        """Set the current GPU utilization (0.0 to 1.0)."""
        with self.state_lock:
            self.gpu_utilization = max(0.0, min(1.0, utilization))

    def simulate_workload_change(self):
        """Simulate random workload changes for testing."""
        with self.state_lock:
            # Randomly adjust queue depth
            self.queue_depth = max(0, self.queue_depth + random.randint(-2, 3))
            # Randomly adjust GPU utilization
            self.gpu_utilization = max(0.0, min(1.0, self.gpu_utilization + random.uniform(-0.1, 0.1)))

    def start(self):
        """Start the worker threads."""
        self._reporter_thread = threading.Thread(
            target=self._reporter,
            name=f"{self.node_id}-Reporter"
        )
        self._listener_thread = threading.Thread(
            target=self._listener,
            name=f"{self.node_id}-Listener"
        )

        self._reporter_thread.start()
        self._listener_thread.start()
        self.logger.info(f"{self.node_id} started")

    def stop(self):
        """Stop the worker gracefully."""
        if not self.is_running:
            return

        self.is_running = False
        self.sock.close()

        if self._reporter_thread:
            self._reporter_thread.join()
        if self._listener_thread:
            self._listener_thread.join()

        if self.verbose:
            self.logger.info(f"{self.node_id} stopped")


def main():
    """Main entry point for running a worker node."""
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="Worker Node")
    parser.add_argument("--node-id", required=True, help="Unique worker ID")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, required=True, help="Port to bind to")
    parser.add_argument("--coordinator-host", default="127.0.0.1",
                        help="Coordinator host")
    parser.add_argument("--coordinator-port", type=int, default=9000,
                        help="Coordinator port")
    parser.add_argument("--report-interval", type=float, default=1.0,
                        help="Report interval in seconds")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    worker = WorkerNode(
        node_id=args.node_id,
        host=args.host,
        port=args.port,
        coordinator_host=args.coordinator_host,
        coordinator_port=args.coordinator_port,
        report_interval=args.report_interval,
        verbose=args.verbose
    )

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print(f"\n\nShutting down worker {args.node_id}...")
        worker.stop()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start worker
    worker.start()
    print(f"Worker {args.node_id} running on {args.host}:{args.port}")
    print("Press Ctrl+C to stop")

    # Simulate workload for testing
    try:
        while True:
            time.sleep(2)
            worker.simulate_workload_change()
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
