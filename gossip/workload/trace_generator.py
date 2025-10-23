"""
Trace generator for serverless LLM workloads.

Implements realistic request arrival patterns based on:
- Azure Serverless Trace characteristics
- Gamma distribution for bursty arrivals (configurable CV)
- Zipf distribution for model popularity
- Function→Model mapping from serverless traces

The generator creates time-stamped request traces that can be replayed
to evaluate scheduling and caching strategies under realistic load.
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path


@dataclass
class TraceRequest:
    """A single request in the trace with timing and model information."""

    request_id: str
    timestamp: float  # Seconds from trace start
    model_id: str
    function_id: Optional[str] = None  # Serverless function that triggered this
    priority: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'timestamp': self.timestamp,
            'model_id': self.model_id,
            'function_id': self.function_id,
            'priority': self.priority
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TraceRequest':
        """Create from dictionary."""
        return cls(
            request_id=data['request_id'],
            timestamp=data['timestamp'],
            model_id=data['model_id'],
            function_id=data.get('function_id'),
            priority=data.get('priority', 0)
        )


@dataclass
class WorkloadTrace:
    """Complete workload trace with requests and metadata."""

    requests: List[TraceRequest]
    duration: float  # Total trace duration in seconds
    mean_rps: float  # Target average requests per second
    cv: float  # Coefficient of variation (burstiness)
    models: List[str]  # All models in the trace
    model_popularity: Dict[str, float]  # Model ID → popularity score
    metadata: Dict = field(default_factory=dict)

    @property
    def total_requests(self) -> int:
        """Total number of requests in trace."""
        return len(self.requests)

    @property
    def actual_rps(self) -> float:
        """Actual requests per second in the trace."""
        return self.total_requests / self.duration if self.duration > 0 else 0

    def get_model_counts(self) -> Dict[str, int]:
        """Get request count per model."""
        counts = {}
        for req in self.requests:
            counts[req.model_id] = counts.get(req.model_id, 0) + 1
        return counts

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'requests': [r.to_dict() for r in self.requests],
            'duration': self.duration,
            'mean_rps': self.mean_rps,
            'cv': self.cv,
            'models': self.models,
            'model_popularity': self.model_popularity,
            'metadata': self.metadata
        }

    def save(self, filepath: str):
        """Save trace to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"Saved trace to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'WorkloadTrace':
        """Load trace from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        requests = [TraceRequest.from_dict(r) for r in data['requests']]

        return cls(
            requests=requests,
            duration=data['duration'],
            mean_rps=data['mean_rps'],
            cv=data['cv'],
            models=data['models'],
            model_popularity=data['model_popularity'],
            metadata=data.get('metadata', {})
        )


class TraceGenerator:
    """
    Generate realistic serverless LLM workload traces.

    Uses Gamma distribution for inter-arrival times to create bursty traffic,
    and Zipf distribution for model popularity.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize trace generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.seed = seed

    def generate(
        self,
        duration: float,
        mean_rps: float,
        cv: float = 8.0,
        models: Optional[List[str]] = None,
        num_models: int = 10,
        popularity_alpha: float = 1.2,
        metadata: Optional[Dict] = None
    ) -> WorkloadTrace:
        """
        Generate a workload trace.

        Args:
            duration: Trace duration in seconds
            mean_rps: Average requests per second
            cv: Coefficient of variation for burstiness (CV=8 is highly bursty)
                - CV=0: Constant arrival rate
                - CV=1: Exponential distribution (Poisson process)
                - CV>1: Bursty arrivals
            models: List of model IDs to use (if None, generates generic model IDs)
            num_models: Number of models to generate if models not provided
            popularity_alpha: Zipf distribution parameter (higher = more skewed)
                - alpha=1.0: Classic Zipf (80/20 rule)
                - alpha>1.0: More concentrated on popular models
                - alpha<1.0: More uniform distribution
            metadata: Optional metadata to attach to trace

        Returns:
            WorkloadTrace with timestamped requests
        """
        # Generate model list if not provided
        if models is None:
            models = [f"model-{i}" for i in range(num_models)]

        # Generate model popularity using Zipf distribution
        model_popularity = self._generate_popularity(models, alpha=popularity_alpha)

        # Generate request arrival times using Gamma distribution
        arrival_times = self._generate_arrivals_gamma(
            duration=duration,
            mean_rps=mean_rps,
            cv=cv
        )

        # Assign models to requests based on popularity
        requests = []
        for i, timestamp in enumerate(arrival_times):
            model_id = self._sample_model_by_popularity(models, model_popularity)

            request = TraceRequest(
                request_id=f"req-{i}",
                timestamp=timestamp,
                model_id=model_id,
                priority=0
            )
            requests.append(request)

        # Create trace
        trace = WorkloadTrace(
            requests=requests,
            duration=duration,
            mean_rps=mean_rps,
            cv=cv,
            models=models,
            model_popularity=model_popularity,
            metadata=metadata or {}
        )

        return trace

    def _generate_arrivals_gamma(
        self,
        duration: float,
        mean_rps: float,
        cv: float
    ) -> List[float]:
        """
        Generate request arrival times using Gamma distribution.

        The Gamma distribution allows control over burstiness via CV:
        - shape parameter k = 1/CV²
        - scale parameter θ = mean/k

        Args:
            duration: Trace duration in seconds
            mean_rps: Average requests per second
            cv: Coefficient of variation

        Returns:
            List of arrival timestamps (sorted)
        """
        # Calculate Gamma distribution parameters
        # For Gamma distribution: mean = k*θ, variance = k*θ²
        # CV² = variance/mean² = (k*θ²)/(k*θ)² = 1/k
        # Therefore: k = 1/CV²

        mean_inter_arrival = 1.0 / mean_rps  # Average time between requests
        shape = 1.0 / (cv ** 2)  # k parameter
        scale = mean_inter_arrival * (cv ** 2)  # θ parameter

        # Generate inter-arrival times
        arrival_times = []
        current_time = 0.0

        while current_time < duration:
            # Sample inter-arrival time from Gamma distribution
            inter_arrival = np.random.gamma(shape, scale)
            current_time += inter_arrival

            if current_time < duration:
                arrival_times.append(current_time)

        return arrival_times

    def _generate_popularity(
        self,
        models: List[str],
        alpha: float = 1.2
    ) -> Dict[str, float]:
        """
        Generate model popularity distribution using Zipf's law.

        Zipf distribution: P(k) ∝ 1/k^α
        This creates a power-law distribution where few models are very popular.

        Args:
            models: List of model IDs
            alpha: Zipf parameter (higher = more concentrated)

        Returns:
            Dictionary mapping model_id → popularity_score (probabilities sum to 1)
        """
        n = len(models)

        # Calculate Zipf probabilities
        # p(k) = (1/k^α) / Σ(1/i^α) for i=1 to n
        ranks = np.arange(1, n + 1)
        probabilities = 1.0 / (ranks ** alpha)
        probabilities = probabilities / probabilities.sum()  # Normalize

        # Map to models
        popularity = {model: prob for model, prob in zip(models, probabilities)}

        return popularity

    def _sample_model_by_popularity(
        self,
        models: List[str],
        popularity: Dict[str, float]
    ) -> str:
        """Sample a model according to popularity distribution."""
        # Extract probabilities in same order as models
        probs = [popularity[m] for m in models]

        # Sample
        model = np.random.choice(models, p=probs)
        return model


def generate_trace(
    duration: float = 300,
    mean_rps: float = 10,
    cv: float = 8.0,
    models: Optional[List[str]] = None,
    num_models: int = 10,
    popularity_alpha: float = 1.2,
    seed: Optional[int] = None,
    save_path: Optional[str] = None
) -> WorkloadTrace:
    """
    Convenience function to generate a workload trace.

    Args:
        duration: Trace duration in seconds (default: 300 = 5 minutes)
        mean_rps: Average requests per second (default: 10)
        cv: Coefficient of variation for burstiness (default: 8 = highly bursty)
        models: List of model IDs (if None, generates generic IDs)
        num_models: Number of models if not provided (default: 10)
        popularity_alpha: Zipf parameter for popularity (default: 1.2)
        seed: Random seed for reproducibility
        save_path: Optional path to save trace as JSON

    Returns:
        WorkloadTrace object
    """
    generator = TraceGenerator(seed=seed)

    trace = generator.generate(
        duration=duration,
        mean_rps=mean_rps,
        cv=cv,
        models=models,
        num_models=num_models,
        popularity_alpha=popularity_alpha,
        metadata={
            'seed': seed,
            'generator': 'TraceGenerator'
        }
    )

    # Print statistics
    print("=" * 70)
    print("GENERATED WORKLOAD TRACE")
    print("=" * 70)
    print(f"Duration: {duration}s")
    print(f"Total requests: {trace.total_requests}")
    print(f"Target RPS: {mean_rps:.2f}")
    print(f"Actual RPS: {trace.actual_rps:.2f}")
    print(f"CV (burstiness): {cv}")
    print(f"Models: {len(trace.models)}")
    print(f"\nModel popularity (top 5):")

    # Sort models by popularity
    sorted_models = sorted(
        trace.model_popularity.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for i, (model, pop) in enumerate(sorted_models[:5]):
        count = sum(1 for r in trace.requests if r.model_id == model)
        print(f"  {i+1}. {model}: {pop:.3f} ({count} requests, {count/trace.total_requests*100:.1f}%)")

    print("=" * 70)

    # Save if requested
    if save_path:
        trace.save(save_path)

    return trace


if __name__ == "__main__":
    # Example usage
    print("Testing trace generator...\n")

    # Generate a bursty trace with realistic parameters
    trace = generate_trace(
        duration=60,  # 1 minute
        mean_rps=5,   # 5 requests/sec average
        cv=8,         # Highly bursty (like serverless workloads)
        models=["llama-7b", "llama-13b", "gpt-neo", "falcon-7b", "mistral-7b"],
        popularity_alpha=1.2,  # Top models get most requests
        seed=42
    )

    # Show arrival pattern
    print("\nFirst 10 request arrivals:")
    for i, req in enumerate(trace.requests[:10]):
        print(f"  {req.timestamp:.3f}s: {req.request_id} → {req.model_id}")

    # Analyze inter-arrival times
    if len(trace.requests) > 1:
        inter_arrivals = [
            trace.requests[i+1].timestamp - trace.requests[i].timestamp
            for i in range(len(trace.requests) - 1)
        ]
        print(f"\nInter-arrival time stats:")
        print(f"  Mean: {np.mean(inter_arrivals):.3f}s")
        print(f"  Std: {np.std(inter_arrivals):.3f}s")
        print(f"  Min: {np.min(inter_arrivals):.3f}s")
        print(f"  Max: {np.max(inter_arrivals):.3f}s")
        print(f"  Actual CV: {np.std(inter_arrivals) / np.mean(inter_arrivals):.2f}")
