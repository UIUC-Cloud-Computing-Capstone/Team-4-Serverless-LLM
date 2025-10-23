"""
UDP-based centralized coordinator for optimal worker placement.

This implementation uses UDP and pickle serialization to match the
existing gossip protocol architecture while providing centralized scheduling.
"""

import socket
import threading
import pickle
import time
import logging
from typing import Dict, Tuple, Optional
from central_coordinator import CentralCoordinator
from contracts import WorkerLoadReport, ScheduleRequest, ScheduleResponse

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(threadName)s] %(message)s')


class UDPCentralCoordinator:
    """
    UDP-based centralized coordinator matching the gossip architecture.

    Listens for worker load reports and schedule requests via UDP,
    maintains global cluster state, and responds with optimal placements.
    """

    def __init__(
        self,
        coordinator_id: str = "central-coordinator",
        host: str = "127.0.0.1",
        port: int = 9000,
        worker_timeout: float = 10.0,
        verbose: bool = True
    ):
        """
        Initialize the UDP coordinator.

        Args:
            coordinator_id: Unique identifier for this coordinator
            host: Host address to bind to
            port: UDP port to listen on
            worker_timeout: Time after which a worker is considered dead
            verbose: Enable detailed logging
        """
        self.coordinator_id = coordinator_id
        self.host = host
        self.port = port
        self.verbose = verbose

        # Core coordinator logic
        self.central_coordinator = CentralCoordinator(
            coordinator_id=coordinator_id,
            worker_timeout=worker_timeout
        )

        # UDP socket setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))

        # Threading
        self.is_running = True
        self._listener_thread = None

        self.logger = logging.getLogger(f"UDPCoordinator-{coordinator_id}")
        if self.verbose:
            self.logger.info(
                f"UDP Coordinator {coordinator_id} initialized at {host}:{port}"
            )

    def _listener(self):
        """Listen for incoming UDP messages from workers and clients."""
        if self.verbose:
            self.logger.info(f"{self.coordinator_id} listener started.")

        while self.is_running:
            try:
                raw_data, sender_addr = self.sock.recvfrom(8192)
                message = pickle.loads(raw_data)
                self._handle_message(message, sender_addr)

            except (socket.error, pickle.UnpicklingError, EOFError) as e:
                if self.is_running:
                    self.logger.error(
                        f"{self.coordinator_id} listener error: {e}",
                        exc_info=True
                    )
                break

        if self.verbose:
            self.logger.info(f"{self.coordinator_id} listener stopped.")

    def _handle_message(self, message: Dict, sender_addr: Tuple[str, int]):
        """
        Process incoming messages.

        Message types:
        - 'worker_report': Worker load state update
        - 'schedule_request': Request for worker placement
        """
        msg_type = message.get('type')

        if msg_type == 'worker_report':
            self._handle_worker_report(message['payload'])

        elif msg_type == 'schedule_request':
            self._handle_schedule_request(message['payload'], sender_addr)

        else:
            self.logger.warning(f"Unknown message type: {msg_type}")

    def _handle_worker_report(self, payload: Dict):
        """Handle worker load report."""
        try:
            report = WorkerLoadReport.from_dict(payload)
            self.central_coordinator.update_worker_state(report)

            if self.verbose:
                self.logger.debug(
                    f"Updated state for {report.node_id}: "
                    f"models={report.loaded_models}, queue={report.queue_depth}"
                )

        except Exception as e:
            self.logger.error(f"Error handling worker report: {e}")

    def _handle_schedule_request(self, payload: Dict, sender_addr: Tuple[str, int]):
        """Handle schedule request and send response."""
        try:
            request = ScheduleRequest.from_dict(payload)
            response = self.central_coordinator.schedule(request)

            # Send response back to requester
            response_message = {
                'type': 'schedule_response',
                'payload': response.to_dict()
            }

            serialized_response = pickle.dumps(response_message)
            self.sock.sendto(serialized_response, sender_addr)

            if self.verbose:
                self.logger.info(
                    f"Scheduled {request.request_id} to {response.worker_id} "
                    f"(action: {response.action.value})"
                )

        except RuntimeError as e:
            # No workers available
            error_message = {
                'type': 'schedule_error',
                'payload': {'error': str(e)}
            }
            serialized_error = pickle.dumps(error_message)
            self.sock.sendto(serialized_error, sender_addr)
            self.logger.error(f"Schedule error: {e}")

        except Exception as e:
            self.logger.error(f"Error handling schedule request: {e}", exc_info=True)

    def start(self):
        """Start the coordinator listener thread."""
        self._listener_thread = threading.Thread(
            target=self._listener,
            name=f"{self.coordinator_id}-Listener"
        )
        self._listener_thread.start()
        self.logger.info(f"{self.coordinator_id} started")

    def stop(self):
        """Stop the coordinator gracefully."""
        if not self.is_running:
            return

        self.is_running = False
        self.sock.close()

        if self._listener_thread:
            self._listener_thread.join()

        if self.verbose:
            self.logger.info(f"{self.coordinator_id} stopped")

    def print_cluster_state(self):
        """Print current cluster state."""
        self.central_coordinator.print_cluster_state()

    def get_cluster_state(self) -> Dict:
        """Get current cluster state as dictionary."""
        return self.central_coordinator.get_cluster_state()


def main():
    """Main entry point for running the UDP coordinator."""
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="UDP-based Central Coordinator")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9000, help="Port to bind to")
    parser.add_argument("--coordinator-id", default="central-coordinator", help="Central Coordinator ID")
    parser.add_argument("--worker-timeout", type=float, default=10.0,
                        help="Worker timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    central_coordinator = UDPCentralCoordinator(
        coordinator_id=args.coordinator_id,
        host=args.host,
        port=args.port,
        worker_timeout=args.worker_timeout,
        verbose=args.verbose
    )

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nShutting down central coordinator...")
        central_coordinator.print_cluster_state()
        central_coordinator.stop()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start coordinator
    central_coordinator.start()
    print(f"Central Coordinator running on {args.host}:{args.port}")
    print("Press Ctrl+C to stop")

    # Keep main thread alive
    try:
        while True:
            time.sleep(5)
            if args.verbose:
                central_coordinator.print_cluster_state()
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
