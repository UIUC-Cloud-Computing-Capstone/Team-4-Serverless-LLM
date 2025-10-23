"""
Workload trace generator for LLM inference requests.

Creates bursty request traces using Gamma distribution following
the AlpaServe methodology (CV=8).
"""

import numpy as np
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')


@dataclass
class RequestTrace:
    """Represents a single request in the trace."""
    request_id: str
    timestamp: float  # Relative timestamp in seconds
    model_id: str
    dataset_source: str  # 'gsm8k' or 'sharegpt'
    data: Dict[str, Any]


class WorkloadGenerator:
    """
    Generates bursty workload traces using Gamma distribution.

    Follows the AlpaServe methodology with CV=8 (coefficient of variation)
    to create realistic bursty request patterns.
    """

    def __init__(
        self,
        target_rps: float = 10.0,
        cv: float = 8.0,
        duration: float = 300.0,
        models: Optional[List[str]] = None,
        seed: Optional[int] = 42
    ):
        """
        Initialize workload generator.

        Args:
            target_rps: Target requests per second (RPS)
            cv: Coefficient of variation for Gamma distribution (default: 8)
            duration: Duration of trace in seconds
            models: List of model IDs to assign to requests
            seed: Random seed for reproducibility
        """
        self.target_rps = target_rps
        self.cv = cv
        self.duration = duration
        self.models = models or ['opt-1.3b', 'opt-2.7b']
        self.logger = logging.getLogger("WorkloadGenerator")

        if seed is not None:
            np.random.seed(seed)

        # Gamma distribution parameters
        # For Gamma distribution: CV = 1/sqrt(k), where k is shape parameter
        # Given CV=8: k = 1/CV^2 = 1/64
        self.shape = 1.0 / (cv ** 2)  # k parameter

        # Mean inter-arrival time (in seconds)
        self.mean_inter_arrival = 1.0 / target_rps

        # Scale parameter: theta = mean / k
        self.scale = self.mean_inter_arrival / self.shape  # θ parameter

        self.logger.info(
            f"Workload generator initialized: target_rps={target_rps}, CV={cv}, "
            f"shape={self.shape:.4f}, scale={self.scale:.4f}"
        )

        self.trace: List[RequestTrace] = []

    def generate_inter_arrival_times(self, num_requests: Optional[int] = None) -> np.ndarray:
        """
        Generate inter-arrival times using Gamma distribution.

        Args:
            num_requests: Number of requests to generate. If None, estimates based on duration.

        Returns:
            Array of inter-arrival times (in seconds)
        """
        if num_requests is None:
            # Estimate number of requests based on duration and target RPS
            num_requests = int(self.duration * self.target_rps * 1.5)  # 1.5x buffer

        # Generate inter-arrival times from Gamma distribution
        inter_arrival_times = np.random.gamma(
            shape=self.shape,
            scale=self.scale,
            size=num_requests
        )

        self.logger.info(
            f"Generated {num_requests} inter-arrival times with "
            f"mean={np.mean(inter_arrival_times):.4f}s, "
            f"std={np.std(inter_arrival_times):.4f}s, "
            f"CV={np.std(inter_arrival_times)/np.mean(inter_arrival_times):.2f}"
        )

        return inter_arrival_times

    def generate_trace(
        self,
        dataset_samples: Optional[List[Dict[str, Any]]] = None
    ) -> List[RequestTrace]:
        """
        Generate a complete request trace.

        Args:
            dataset_samples: Optional list of dataset samples to use for requests

        Returns:
            List of RequestTrace objects
        """
        # Generate inter-arrival times
        inter_arrival_times = self.generate_inter_arrival_times()

        # Calculate cumulative arrival times
        arrival_times = np.cumsum(inter_arrival_times)

        # Filter to only requests within duration
        mask = arrival_times <= self.duration
        arrival_times = arrival_times[mask]

        num_requests = len(arrival_times)
        self.logger.info(
            f"Generating trace with {num_requests} requests over {self.duration}s "
            f"(actual RPS: {num_requests/self.duration:.2f})"
        )

        # Generate trace
        self.trace = []
        for i, timestamp in enumerate(arrival_times):
            # Assign model (round-robin or based on dataset)
            model_id = self.models[i % len(self.models)]

            # Get dataset sample if available
            if dataset_samples and i < len(dataset_samples):
                sample = dataset_samples[i]
                dataset_source = sample.get('dataset', 'unknown')
                data = sample.get('data', {})
            else:
                dataset_source = 'synthetic'
                data = {'request_index': i}

            request = RequestTrace(
                request_id=f"req-{i}",
                timestamp=timestamp,
                model_id=model_id,
                dataset_source=dataset_source,
                data=data
            )
            self.trace.append(request)

        self.logger.info(f"Trace generation complete: {len(self.trace)} requests")
        return self.trace

    def get_trace(self) -> List[RequestTrace]:
        """
        Get the generated trace.

        Returns:
            List of RequestTrace objects
        """
        if not self.trace:
            self.logger.warning("No trace generated yet, generating now")
            self.generate_trace()
        return self.trace

    def scale_to_rps(self, new_rps: float) -> List[RequestTrace]:
        """
        Scale the trace to a new target RPS.

        This rescales the timestamps while maintaining the same burstiness pattern.

        Args:
            new_rps: New target RPS

        Returns:
            List of RequestTrace objects with scaled timestamps
        """
        if not self.trace:
            self.logger.warning("No trace to scale, generating first")
            self.generate_trace()

        # Calculate scaling factor
        current_rps = len(self.trace) / self.duration
        scale_factor = current_rps / new_rps

        self.logger.info(f"Scaling trace from {current_rps:.2f} RPS to {new_rps:.2f} RPS")

        # Scale timestamps
        scaled_trace = []
        for req in self.trace:
            scaled_req = RequestTrace(
                request_id=req.request_id,
                timestamp=req.timestamp * scale_factor,
                model_id=req.model_id,
                dataset_source=req.dataset_source,
                data=req.data
            )
            # Only include if still within duration
            if scaled_req.timestamp <= self.duration:
                scaled_trace.append(scaled_req)

        self.trace = scaled_trace
        self.target_rps = new_rps
        self.logger.info(f"Scaled trace: {len(self.trace)} requests")
        return self.trace

    def get_trace_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the generated trace.

        Returns:
            Dictionary with trace statistics
        """
        if not self.trace:
            return {}

        timestamps = [req.timestamp for req in self.trace]
        inter_arrivals = np.diff([0] + timestamps)

        stats = {
            'num_requests': len(self.trace),
            'duration': self.duration,
            'actual_rps': len(self.trace) / self.duration,
            'target_rps': self.target_rps,
            'mean_inter_arrival': np.mean(inter_arrivals),
            'std_inter_arrival': np.std(inter_arrivals),
            'cv_inter_arrival': np.std(inter_arrivals) / np.mean(inter_arrivals),
            'min_inter_arrival': np.min(inter_arrivals),
            'max_inter_arrival': np.max(inter_arrivals),
            'models': list(set(req.model_id for req in self.trace)),
            'datasets': list(set(req.dataset_source for req in self.trace))
        }

        return stats

    def assign_models_to_requests(
        self,
        model_assignment: Dict[str, float]
    ) -> None:
        """
        Assign models to requests based on specified distribution.

        Args:
            model_assignment: Dictionary mapping model_id to probability
                             e.g., {'opt-1.3b': 0.6, 'opt-2.7b': 0.4}
        """
        if not self.trace:
            self.logger.warning("No trace to assign models, generating first")
            self.generate_trace()

        # Normalize probabilities
        total = sum(model_assignment.values())
        normalized = {k: v/total for k, v in model_assignment.items()}

        # Assign models based on probabilities
        models = list(normalized.keys())
        probs = list(normalized.values())

        for req in self.trace:
            req.model_id = np.random.choice(models, p=probs)

        self.logger.info(f"Assigned models to {len(self.trace)} requests")

    def save_trace(self, file_path: str) -> None:
        """
        Save trace to JSON file.

        Args:
            file_path: Path to save trace
        """
        import json

        if not self.trace:
            self.logger.warning("No trace to save")
            return

        trace_data = []
        for req in self.trace:
            trace_data.append({
                'request_id': req.request_id,
                'timestamp': req.timestamp,
                'model_id': req.model_id,
                'dataset_source': req.dataset_source,
                'data': req.data
            })

        with open(file_path, 'w') as f:
            json.dump(trace_data, f, indent=2)

        self.logger.info(f"Saved trace to {file_path}")

    def load_trace(self, file_path: str) -> List[RequestTrace]:
        """
        Load trace from JSON file.

        Args:
            file_path: Path to trace file

        Returns:
            List of RequestTrace objects
        """
        import json

        with open(file_path, 'r') as f:
            trace_data = json.load(f)

        self.trace = []
        for item in trace_data:
            req = RequestTrace(
                request_id=item['request_id'],
                timestamp=item['timestamp'],
                model_id=item['model_id'],
                dataset_source=item['dataset_source'],
                data=item['data']
            )
            self.trace.append(req)

        self.logger.info(f"Loaded trace from {file_path}: {len(self.trace)} requests")
        return self.trace


def main():
    """Example usage of WorkloadGenerator."""
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Workload Generator Test")
    parser.add_argument("--rps", type=float, default=10.0, help="Target RPS")
    parser.add_argument("--cv", type=float, default=8.0, help="Coefficient of variation")
    parser.add_argument("--duration", type=float, default=300.0, help="Duration in seconds")
    parser.add_argument("--output", type=str, help="Output trace file (JSON)")
    parser.add_argument("--plot", action="store_true", help="Plot trace visualization")

    args = parser.parse_args()

    # Create generator
    generator = WorkloadGenerator(
        target_rps=args.rps,
        cv=args.cv,
        duration=args.duration,
        models=['opt-1.3b', 'opt-2.7b']
    )

    # Generate trace
    trace = generator.generate_trace()

    # Print statistics
    stats = generator.get_trace_stats()
    print("\n=== Trace Statistics ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Save trace if requested
    if args.output:
        generator.save_trace(args.output)

    # Plot if requested
    if args.plot:
        timestamps = [req.timestamp for req in trace]

        plt.figure(figsize=(12, 6))

        # Plot 1: Request arrivals over time
        plt.subplot(2, 1, 1)
        plt.scatter(timestamps, range(len(timestamps)), alpha=0.5, s=1)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Request Number')
        plt.title(f'Request Arrivals (RPS={args.rps}, CV={args.cv})')
        plt.grid(True, alpha=0.3)

        # Plot 2: Inter-arrival time distribution
        plt.subplot(2, 1, 2)
        inter_arrivals = np.diff([0] + timestamps)
        plt.hist(inter_arrivals, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Inter-arrival Time (seconds)')
        plt.ylabel('Frequency')
        plt.title(f'Inter-arrival Time Distribution (mean={np.mean(inter_arrivals):.4f}s)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('workload_trace.png', dpi=150)
        print("\nPlot saved to workload_trace.png")


if __name__ == "__main__":
    main()
