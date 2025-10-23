"""
Benchmark runner for evaluating serverless LLM scheduling systems.

Orchestrates:
1. Dataset loading (GSM8K + ShareGPT)
2. Workload trace generation (Gamma-distributed arrivals)
3. Model placement across workers
4. Request replay and metric collection

This allows evaluation of scheduling policies under realistic workloads.
"""

import time
import random
import pickle
import socket
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import csv
import json

from .datasets import load_mixed_dataset, InferenceRequest
from .trace_generator import generate_trace, WorkloadTrace, TraceRequest
from .model_placement import place_models, PlacementPlan

# Import parent directory modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from contracts import ScheduleRequest, ScheduleResponse


@dataclass
class BenchmarkRequest:
    """A benchmark request combining trace and dataset information."""

    request_id: str
    timestamp: float  # Seconds from benchmark start
    model_id: str
    prompt: str
    dataset_source: str
    token_count: int
    priority: int = 0


@dataclass
class BenchmarkResult:
    """Result of a single request in the benchmark."""

    request_id: str
    submission_time: float  # Actual time request was submitted
    model_id: str
    dataset_source: str
    token_count: int

    # Scheduling decision
    assigned_worker: Optional[str] = None
    placement_action: Optional[str] = None  # SERVE, COLD_START, MIGRATE
    estimated_wait_time: Optional[float] = None

    # Timing
    scheduling_latency: Optional[float] = None  # Time to get scheduling response

    # Errors
    error: Optional[str] = None
    timeout: bool = False

    def to_dict(self) -> dict:
        return {
            'request_id': self.request_id,
            'submission_time': self.submission_time,
            'model_id': self.model_id,
            'dataset_source': self.dataset_source,
            'token_count': self.token_count,
            'assigned_worker': self.assigned_worker,
            'placement_action': self.placement_action,
            'estimated_wait_time': self.estimated_wait_time,
            'scheduling_latency': self.scheduling_latency,
            'error': self.error,
            'timeout': self.timeout
        }


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics from a benchmark run."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    timeout_requests: int

    # Latency metrics (seconds)
    mean_scheduling_latency: float
    median_scheduling_latency: float
    p95_scheduling_latency: float
    p99_scheduling_latency: float

    # Cache metrics
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float

    # Wait time estimates
    mean_estimated_wait: float
    median_estimated_wait: float

    # Per-model breakdown
    per_model_stats: Dict[str, Dict] = field(default_factory=dict)

    def print_summary(self):
        """Print benchmark metrics summary."""
        print("=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        print(f"Total requests: {self.total_requests}")
        print(f"  Successful: {self.successful_requests} ({self.successful_requests/self.total_requests*100:.1f}%)")
        print(f"  Failed: {self.failed_requests}")
        print(f"  Timeout: {self.timeout_requests}")

        print(f"\nScheduling Latency:")
        print(f"  Mean: {self.mean_scheduling_latency*1000:.2f} ms")
        print(f"  Median: {self.median_scheduling_latency*1000:.2f} ms")
        print(f"  P95: {self.p95_scheduling_latency*1000:.2f} ms")
        print(f"  P99: {self.p99_scheduling_latency*1000:.2f} ms")

        print(f"\nCache Performance:")
        print(f"  Cache hits: {self.cache_hits} ({self.cache_hit_rate*100:.1f}%)")
        print(f"  Cache misses: {self.cache_misses} ({(1-self.cache_hit_rate)*100:.1f}%)")

        print(f"\nEstimated Wait Time:")
        print(f"  Mean: {self.mean_estimated_wait:.2f} s")
        print(f"  Median: {self.median_estimated_wait:.2f} s")

        if self.per_model_stats:
            print(f"\nPer-Model Statistics:")
            for model_id, stats in sorted(self.per_model_stats.items(), key=lambda x: x[1]['count'], reverse=True):
                print(f"  {model_id}:")
                print(f"    Requests: {stats['count']}")
                print(f"    Cache hit rate: {stats['cache_hit_rate']*100:.1f}%")
                print(f"    Avg latency: {stats['avg_latency']*1000:.2f} ms")

        print("=" * 70)


class BenchmarkRunner:
    """
    Benchmark runner for serverless LLM scheduling systems.

    Integrates dataset loading, trace generation, and request replay
    to evaluate scheduling under realistic workloads.
    """

    def __init__(
        self,
        coordinator_host: str = "127.0.0.1",
        coordinator_port: int = 9000,
        request_timeout: float = 5.0
    ):
        """
        Initialize benchmark runner.

        Args:
            coordinator_host: Central coordinator host
            coordinator_port: Central coordinator port
            request_timeout: Timeout for scheduling requests (seconds)
        """
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.request_timeout = request_timeout

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(request_timeout)

    def prepare_benchmark(
        self,
        duration: float = 300,
        mean_rps: float = 10,
        cv: float = 8.0,
        models: Optional[List[str]] = None,
        num_models: int = 10,
        popularity_alpha: float = 1.2,
        gsm8k_samples: int = 4000,
        sharegpt_samples: int = 4000,
        max_tokens: int = 2048,
        seed: Optional[int] = None
    ) -> Tuple[List[BenchmarkRequest], PlacementPlan]:
        """
        Prepare benchmark workload.

        Args:
            duration: Trace duration (seconds)
            mean_rps: Average requests per second
            cv: Coefficient of variation for burstiness
            models: List of model IDs
            num_models: Number of models if not specified
            popularity_alpha: Zipf parameter
            gsm8k_samples: Number of GSM8K samples
            sharegpt_samples: Number of ShareGPT samples
            max_tokens: Max context length
            seed: Random seed

        Returns:
            (benchmark_requests, placement_plan)
        """
        if seed is not None:
            random.seed(seed)

        print("=" * 70)
        print("PREPARING BENCHMARK")
        print("=" * 70)

        # 1. Load datasets
        print("\n1. Loading datasets...")
        dataset = load_mixed_dataset(
            gsm8k_samples=gsm8k_samples,
            sharegpt_samples=sharegpt_samples,
            max_tokens=max_tokens,
            shuffle=True
        )

        # 2. Generate trace
        print("\n2. Generating workload trace...")
        trace = generate_trace(
            duration=duration,
            mean_rps=mean_rps,
            cv=cv,
            models=models,
            num_models=num_models,
            popularity_alpha=popularity_alpha,
            seed=seed
        )

        # 3. Combine trace with dataset
        print("\n3. Combining trace with dataset prompts...")
        benchmark_requests = self._combine_trace_and_dataset(trace, dataset)

        print(f"\nCreated {len(benchmark_requests)} benchmark requests")
        print("=" * 70)

        # Note: Placement plan would be used to initialize workers
        # For now, we return it for informational purposes
        placement_plan = None

        return benchmark_requests, placement_plan

    def _combine_trace_and_dataset(
        self,
        trace: WorkloadTrace,
        dataset: List[InferenceRequest]
    ) -> List[BenchmarkRequest]:
        """
        Combine trace timing with dataset prompts.

        Args:
            trace: Workload trace with timing and models
            dataset: Dataset with prompts

        Returns:
            List of benchmark requests
        """
        benchmark_requests = []

        # Shuffle dataset for random sampling
        dataset_shuffled = dataset.copy()
        random.shuffle(dataset_shuffled)

        for i, trace_req in enumerate(trace.requests):
            # Sample a prompt from dataset
            dataset_req = dataset_shuffled[i % len(dataset_shuffled)]

            benchmark_req = BenchmarkRequest(
                request_id=trace_req.request_id,
                timestamp=trace_req.timestamp,
                model_id=trace_req.model_id,
                prompt=dataset_req.prompt,
                dataset_source=dataset_req.dataset_source,
                token_count=dataset_req.token_count,
                priority=trace_req.priority
            )

            benchmark_requests.append(benchmark_req)

        return benchmark_requests

    def run_benchmark(
        self,
        requests: List[BenchmarkRequest],
        output_dir: Optional[str] = None,
        verbose: bool = False
    ) -> Tuple[List[BenchmarkResult], BenchmarkMetrics]:
        """
        Run benchmark by replaying requests.

        Args:
            requests: List of benchmark requests to replay
            output_dir: Directory to save results
            verbose: Print per-request info

        Returns:
            (results, metrics)
        """
        print("\n" + "=" * 70)
        print("RUNNING BENCHMARK")
        print("=" * 70)
        print(f"Total requests: {len(requests)}")
        print(f"Duration: {requests[-1].timestamp:.1f}s" if requests else "0s")
        print("Starting replay...\n")

        results = []
        start_time = time.time()

        for i, req in enumerate(requests):
            # Wait until scheduled time
            target_time = start_time + req.timestamp
            current_time = time.time()

            if current_time < target_time:
                time.sleep(target_time - current_time)

            # Send scheduling request
            result = self._send_schedule_request(req)
            results.append(result)

            if verbose or (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(requests)}] {req.request_id}: "
                      f"{result.assigned_worker or 'FAILED'} "
                      f"({result.placement_action or 'N/A'}) "
                      f"{result.scheduling_latency*1000:.1f}ms" if result.scheduling_latency else "TIMEOUT")

        print(f"\nBenchmark complete! Total time: {time.time() - start_time:.1f}s")

        # Calculate metrics
        print("\nCalculating metrics...")
        metrics = self._calculate_metrics(results)

        # Save results
        if output_dir:
            self._save_results(results, metrics, output_dir)

        return results, metrics

    def _send_schedule_request(self, req: BenchmarkRequest) -> BenchmarkResult:
        """Send a single scheduling request to coordinator."""
        result = BenchmarkResult(
            request_id=req.request_id,
            submission_time=time.time(),
            model_id=req.model_id,
            dataset_source=req.dataset_source,
            token_count=req.token_count
        )

        try:
            # Create schedule request
            schedule_req = ScheduleRequest(
                request_id=req.request_id,
                model_required=req.model_id,
                priority=req.priority
            )

            # Send to coordinator
            message = {
                'type': 'schedule_request',
                'data': schedule_req.to_dict()
            }

            send_start = time.time()
            self.sock.sendto(
                pickle.dumps(message),
                (self.coordinator_host, self.coordinator_port)
            )

            # Wait for response
            response_data, _ = self.sock.recvfrom(65535)
            response_msg = pickle.loads(response_data)

            result.scheduling_latency = time.time() - send_start

            if response_msg['type'] == 'schedule_response':
                schedule_resp = ScheduleResponse.from_dict(response_msg['data'])
                result.assigned_worker = schedule_resp.worker_id
                result.placement_action = schedule_resp.action.value
                result.estimated_wait_time = schedule_resp.estimated_wait_time

            elif response_msg['type'] == 'schedule_error':
                result.error = response_msg['data'].get('error', 'Unknown error')

        except socket.timeout:
            result.timeout = True
            result.error = "Request timeout"

        except Exception as e:
            result.error = str(e)

        return result

    def _calculate_metrics(self, results: List[BenchmarkResult]) -> BenchmarkMetrics:
        """Calculate aggregated metrics from results."""
        total = len(results)
        successful = [r for r in results if r.assigned_worker is not None]
        failed = [r for r in results if r.error and not r.timeout]
        timeout = [r for r in results if r.timeout]

        # Latency metrics
        latencies = [r.scheduling_latency for r in successful if r.scheduling_latency]
        latencies.sort()

        mean_lat = sum(latencies) / len(latencies) if latencies else 0
        median_lat = latencies[len(latencies)//2] if latencies else 0
        p95_lat = latencies[int(len(latencies)*0.95)] if latencies else 0
        p99_lat = latencies[int(len(latencies)*0.99)] if latencies else 0

        # Cache metrics
        cache_hits = sum(1 for r in successful if r.placement_action == "SERVE")
        cache_misses = sum(1 for r in successful if r.placement_action == "COLD_START")
        cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0

        # Wait time
        wait_times = [r.estimated_wait_time for r in successful if r.estimated_wait_time]
        wait_times.sort()
        mean_wait = sum(wait_times) / len(wait_times) if wait_times else 0
        median_wait = wait_times[len(wait_times)//2] if wait_times else 0

        # Per-model stats
        per_model_stats = {}
        for model_id in set(r.model_id for r in results):
            model_results = [r for r in successful if r.model_id == model_id]
            if model_results:
                model_hits = sum(1 for r in model_results if r.placement_action == "SERVE")
                per_model_stats[model_id] = {
                    'count': len(model_results),
                    'cache_hit_rate': model_hits / len(model_results),
                    'avg_latency': sum(r.scheduling_latency for r in model_results if r.scheduling_latency) / len(model_results)
                }

        metrics = BenchmarkMetrics(
            total_requests=total,
            successful_requests=len(successful),
            failed_requests=len(failed),
            timeout_requests=len(timeout),
            mean_scheduling_latency=mean_lat,
            median_scheduling_latency=median_lat,
            p95_scheduling_latency=p95_lat,
            p99_scheduling_latency=p99_lat,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_hit_rate=cache_hit_rate,
            mean_estimated_wait=mean_wait,
            median_estimated_wait=median_wait,
            per_model_stats=per_model_stats
        )

        return metrics

    def _save_results(
        self,
        results: List[BenchmarkResult],
        metrics: BenchmarkMetrics,
        output_dir: str
    ):
        """Save results and metrics to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results as CSV
        csv_path = output_path / "results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
            writer.writeheader()
            for result in results:
                writer.writerow(result.to_dict())

        print(f"Saved detailed results to {csv_path}")

        # Save metrics as JSON
        metrics_path = output_path / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'total_requests': metrics.total_requests,
                'successful_requests': metrics.successful_requests,
                'failed_requests': metrics.failed_requests,
                'timeout_requests': metrics.timeout_requests,
                'mean_scheduling_latency_ms': metrics.mean_scheduling_latency * 1000,
                'median_scheduling_latency_ms': metrics.median_scheduling_latency * 1000,
                'p95_scheduling_latency_ms': metrics.p95_scheduling_latency * 1000,
                'p99_scheduling_latency_ms': metrics.p99_scheduling_latency * 1000,
                'cache_hits': metrics.cache_hits,
                'cache_misses': metrics.cache_misses,
                'cache_hit_rate': metrics.cache_hit_rate,
                'mean_estimated_wait_s': metrics.mean_estimated_wait,
                'median_estimated_wait_s': metrics.median_estimated_wait,
                'per_model_stats': metrics.per_model_stats
            }, f, indent=2)

        print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run serverless LLM benchmark")
    parser.add_argument("--duration", type=float, default=60, help="Trace duration (seconds)")
    parser.add_argument("--rps", type=float, default=5, help="Average requests per second")
    parser.add_argument("--cv", type=float, default=8.0, help="Coefficient of variation (burstiness)")
    parser.add_argument("--coordinator-host", default="127.0.0.1", help="Coordinator host")
    parser.add_argument("--coordinator-port", type=int, default=9000, help="Coordinator port")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Run benchmark
    runner = BenchmarkRunner(
        coordinator_host=args.coordinator_host,
        coordinator_port=args.coordinator_port
    )

    # Prepare workload
    requests, placement = runner.prepare_benchmark(
        duration=args.duration,
        mean_rps=args.rps,
        cv=args.cv,
        seed=args.seed
    )

    # Run
    results, metrics = runner.run_benchmark(
        requests=requests,
        output_dir=args.output_dir,
        verbose=args.verbose
    )

    # Print summary
    metrics.print_summary()
