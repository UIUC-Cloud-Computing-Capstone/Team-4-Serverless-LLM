"""
Runner to start the UDP Central Coordinator and measure scheduling latencies.

This script:
- Starts an in-process `UDPCentralCoordinator` (UDP listener)
- Spawns a small set of simulated workers that periodically send `worker_report` messages
- Sends schedule requests and measures round-trip time (RTT) for placement responses
- Prints basic latency statistics (min/avg/p50/p95/p99/max)

Usage (example):
    python gossip/src/coordinator/main.py --workers 3 --requests 200 --rate 20

This script is intended for lightweight benchmarking and debugging; it's not a
full load generator.
"""

import argparse
import logging
import pickle
import socket
import statistics
import threading
import time
import uuid
from typing import List, Tuple

from udp_central_coordinator import UDPCentralCoordinator
from contracts import WorkerLoadReport, ScheduleRequest, ScheduleResponse

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("central-coord-runner")


def start_coordinator(host: str, port: int, worker_timeout: float, verbose: bool) -> UDPCentralCoordinator:
    coord = UDPCentralCoordinator(coordinator_id="central-coordinator", host=host, port=port, worker_timeout=worker_timeout, verbose=verbose)
    coord.start()
    return coord


def _send_udp_message(dest: Tuple[str, int], payload: dict, bind_addr: Tuple[str, int] = ("0.0.0.0", 0), timeout: float = 2.0) -> bytes:
    """Send a pickled UDP message and return raw response bytes (may be empty on timeout).

    The caller manages serialization/deserialization of the response.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(bind_addr)
    sock.settimeout(timeout)
    try:
        raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        sock.sendto(raw, dest)
        data, _ = sock.recvfrom(65536)
        return data
    except socket.timeout:
        return b""
    finally:
        sock.close()


class WorkerSimulator(threading.Thread):
    """Simulates a worker by periodically sending `worker_report` messages to the coordinator."""

    def __init__(self, worker_id: str, host: str, port: int, coordinator_addr: Tuple[str, int], interval: float = 1.0, models: List[str] = None, verbose: bool = False):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.host = host
        self.port = port
        self.coordinator_addr = coordinator_addr
        self.interval = interval
        self.models = models or ["opt-1.3b"]
        self.running = True
        self.verbose = verbose
        # Use a local socket for sending reports
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def run(self):
        while self.running:
            report = WorkerLoadReport(
                node_id=self.worker_id,
                loaded_models=self.models,
                queue_depth=0,
                memory_utilization=0.2,
                is_ready=True,
                timestamp=time.time(),
            )
            message = {"type": "worker_report", "payload": report.to_dict()}
            try:
                self.sock.sendto(pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL), self.coordinator_addr)
                if self.verbose:
                    logger.debug(f"{self.worker_id} sent worker_report")
            except Exception:
                logger.exception("failed to send worker_report")
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass


def warmup_workers(worker_count: int, coordinator_addr: Tuple[str, int], report_interval: float, start_port: int = 10000) -> List[WorkerSimulator]:
    sims: List[WorkerSimulator] = []
    for i in range(worker_count):
        wid = f"worker-{i+1}"
        sim = WorkerSimulator(worker_id=wid, host="127.0.0.1", port=start_port + i, coordinator_addr=coordinator_addr, interval=report_interval, models=["opt-1.3b" if i % 2 == 0 else "opt-2.7b"]) 
        sim.start()
        sims.append(sim)
    return sims


def measure_latencies(coordinator_addr: Tuple[str, int], num_requests: int, rate: float, timeout: float = 2.0, models: List[str] = None) -> List[float]:
    """Send schedule requests at the given rate (requests/sec) and measure RTTs in seconds.

    Returns list of observed RTTs (timeouts are recorded as `timeout`).
    """
    models = models or ["opt-1.3b", "opt-2.7b"]
    inter_arrival = 1.0 / rate if rate > 0 else 0
    latencies: List[float] = []

    # Use a persistent socket for each request to receive the reply (OS will pick ephemeral port)
    for i in range(num_requests):
        req_id = str(uuid.uuid4())
        model = models[i % len(models)]
        request = ScheduleRequest(request_id=req_id, model_required=model, priority=0)
        message = {"type": "schedule_request", "payload": request.to_dict()}

        # send and measure
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 0))
        sock.settimeout(timeout)
        raw = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        start = time.time()
        try:
            sock.sendto(raw, coordinator_addr)
            data, _ = sock.recvfrom(65536)
            end = time.time()
            # try to unpickle
            try:
                resp_obj = pickle.loads(data)
                # resp_obj expected to be dict representation of ScheduleResponse
                # If it is dict, we can parse it
            except Exception:
                # Not pickled dict, skip parsing - we only need RTT
                pass
            latencies.append(end - start)
        except socket.timeout:
            latencies.append(timeout)
        except Exception:
            latencies.append(timeout)
        finally:
            sock.close()

        if inter_arrival > 0:
            time.sleep(inter_arrival)
    return latencies


def print_stats(latencies: List[float]):
    if not latencies:
        print("No latencies recorded")
        return
    vals = sorted(latencies)
    count = len(vals)
    print("\n=== Latency statistics (seconds) ===")
    print(f"count: {count}")
    print(f"min:  {vals[0]:.6f}")
    print(f"p50:  {statistics.median(vals):.6f}")
    print(f"avg:  {statistics.mean(vals):.6f}")
    print(f"p95:  {vals[int(0.95*count)-1]:.6f}")
    print(f"p99:  {vals[int(0.99*count)-1]:.6f}")
    print(f"max:  {vals[-1]:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Run central coordinator and measure scheduling latencies")
    parser.add_argument("--host", default="127.0.0.1", help="Coordinator host to bind")
    parser.add_argument("--port", type=int, default=9000, help="Coordinator UDP port")
    parser.add_argument("--workers", type=int, default=3, help="Number of simulated workers")
    parser.add_argument("--requests", type=int, default=200, help="Number of schedule requests to send")
    parser.add_argument("--rate", type=float, default=20.0, help="Target request rate (requests/sec)")
    parser.add_argument("--report-interval", type=float, default=1.0, help="Worker report interval (s)")
    parser.add_argument("--worker-timeout", type=float, default=10.0, help="Worker timeout used by coordinator")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    coord_addr = (args.host, args.port)

    coord = start_coordinator(host=args.host, port=args.port, worker_timeout=args.worker_timeout, verbose=args.verbose)
    time.sleep(0.5)  # give coordinator a moment to bind

    try:
        sims = warmup_workers(worker_count=args.workers, coordinator_addr=coord_addr, report_interval=args.report_interval)
        # allow some reports to arrive
        logger.info("Warming up workers for 2s...")
        time.sleep(2.0)

        logger.info(f"Sending {args.requests} schedule requests at ~{args.rate} rps to {args.host}:{args.port}")
        latencies = measure_latencies(coordinator_addr=coord_addr, num_requests=args.requests, rate=args.rate)

        print_stats(latencies)

    finally:
        # stop workers
        for s in sims:
            s.stop()
        # stop coordinator
        coord.stop()


if __name__ == "__main__":
    main()
