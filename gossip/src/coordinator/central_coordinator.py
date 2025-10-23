"""
Central coordinator for optimal worker placement.

This module implements a central coordinator that maintains a global view
of all worker states and makes optimal placement decisions for inference requests.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Tuple
from contracts import (
    WorkerLoadReport,
    ScheduleRequest,
    ScheduleResponse,
    PlacementAction
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')


class CentralCoordinator:
    """
    Central coordinator that maintains global cluster state.

    The central coordinator receives periodic load reports from all workers and uses
    this global view to make optimal placement decisions for incoming requests.
    """

    def __init__(
        self,
        coordinator_id: str = "central-coordinator",
        worker_timeout: float = 10.0,
        queue_threshold: int = 10,
        memory_threshold: float = 0.9
    ):
        """
        Initialize the centralized coordinator.

        Args:
            coordinator_id: Unique identifier for this coordinator
            worker_timeout: Time (seconds) after which a worker is considered dead
            queue_threshold: Queue depth threshold for considering a worker overloaded
            memory_threshold: Memory utilization threshold for considering a worker overloaded
        """
        self.coordinator_id = coordinator_id
        self.worker_timeout = worker_timeout
        self.queue_threshold = queue_threshold
        self.memory_threshold = memory_threshold

        # Global state: {worker_id: (WorkerLoadReport, last_update_time)}
        self.worker_states: Dict[str, Tuple[WorkerLoadReport, float]] = {}
        self.state_lock = threading.RLock()

        # Statistics
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0

        self.logger = logging.getLogger(f"CentralCoordinator-{coordinator_id}")
        self.logger.info(f"Central coordinator {coordinator_id} initialized")

    def update_worker_state(self, report: WorkerLoadReport) -> None:
        """
        Update the global state with a worker's load report.

        Args:
            report: The load report from a worker
        """
        with self.state_lock:
            current_time = time.time()
            self.worker_states[report.node_id] = (report, current_time)
            self.logger.debug(
                f"Updated state for {report.node_id}: "
                f"models={report.loaded_models}, queue={report.queue_depth}, "
                f"memory={report.memory_utilization:.2f}"
            )

    def _cleanup_dead_workers(self) -> None:
        """Remove workers that have not reported in a while."""
        with self.state_lock:
            current_time = time.time()
            dead_workers = [
                worker_id
                for worker_id, (_, last_seen) in self.worker_states.items()
                if current_time - last_seen > self.worker_timeout
            ]
            for worker_id in dead_workers:
                del self.worker_states[worker_id]
                self.logger.warning(f"Removed dead worker: {worker_id}")

    def _get_active_workers(self) -> Dict[str, WorkerLoadReport]:
        """
        Get all active workers (recently updated).

        Returns:
            Dictionary mapping worker_id to their latest load report
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

        Scoring criteria:
        1. Having the model loaded is heavily weighted
        2. Lower queue depth is better
        3. Lower memory utilization is better

        Args:
            report: Worker's load report
            has_model: Whether the worker has the required model loaded

        Returns:
            Score value (higher is better)
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

        Placement strategy:
        1. First, try to find a worker with the model already loaded
        2. Among those, prefer workers with low queue depth and GPU utilization
        3. If no worker has the model, perform cold start on the least loaded worker

        Args:
            request: The scheduling request

        Returns:
            Scheduling response with worker assignment and action
        """
        self.total_requests += 1
        active_workers = self._get_active_workers()

        if not active_workers:
            self.logger.error(f"No active workers available for request {request.request_id}")
            raise RuntimeError("No active workers available")

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

    def get_cluster_state(self) -> Dict:
        """
        Get a summary of the current cluster state.

        Returns:
            Dictionary with cluster statistics and worker states
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
            "coordinator_id": self.coordinator_id,
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

    def print_cluster_state(self) -> None:
        """Print a human-readable summary of the cluster state."""
        state = self.get_cluster_state()
        print(f"\n--- Cluster State (Central Coordinator: {state['coordinator_id']}) ---")
        print(f"Active Workers: {state['num_active_workers']}")
        print(f"Ready Workers: {state['num_ready_workers']}")
        print(f"Total Requests: {state['total_requests_processed']}")
        print(f"Cache Hit Rate: {state['cache_hit_rate']:.2%}")
        print(f"Total Queue Depth: {state['total_queue_depth']}")
        print(f"Average Memory Utilization: {state['average_memory_utilization']:.2%}")
        print("\nWorker States:")
        for worker_id, worker_data in state['workers'].items():
            ready_status = "✓" if worker_data.get('is_ready', False) else "✗"
            print(f"  {worker_id} [{ready_status}]: models={worker_data['loaded_models']}, "
                  f"queue={worker_data['queue_depth']}, "
                  f"memory={worker_data['memory_utilization']:.2f}")
