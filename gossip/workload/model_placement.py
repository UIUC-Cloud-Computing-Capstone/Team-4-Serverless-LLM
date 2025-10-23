"""
Model placement strategy for serverless LLM clusters.

Implements popularity-based model replication and round-robin placement
across worker nodes with storage constraints, following the methodology
used in AlpaServe and similar systems.
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path


@dataclass
class ModelInfo:
    """Information about a model."""

    model_id: str
    size_gb: float  # Model size in GB
    popularity: float  # Popularity score (0-1)
    replicas: int = 1  # Number of replicas to create

    def to_dict(self) -> dict:
        return {
            'model_id': self.model_id,
            'size_gb': self.size_gb,
            'popularity': self.popularity,
            'replicas': self.replicas
        }


@dataclass
class WorkerStorage:
    """Storage state for a worker node."""

    worker_id: str
    capacity_gb: float
    used_gb: float = 0.0
    models: List[str] = field(default_factory=list)

    @property
    def available_gb(self) -> float:
        """Available storage in GB."""
        return self.capacity_gb - self.used_gb

    @property
    def utilization(self) -> float:
        """Storage utilization (0-1)."""
        return self.used_gb / self.capacity_gb if self.capacity_gb > 0 else 0.0

    def can_fit(self, size_gb: float) -> bool:
        """Check if model of given size can fit."""
        return self.available_gb >= size_gb

    def add_model(self, model_id: str, size_gb: float) -> bool:
        """
        Add model to worker storage.

        Returns:
            True if added successfully, False if insufficient space
        """
        if not self.can_fit(size_gb):
            return False

        self.models.append(model_id)
        self.used_gb += size_gb
        return True

    def to_dict(self) -> dict:
        return {
            'worker_id': self.worker_id,
            'capacity_gb': self.capacity_gb,
            'used_gb': self.used_gb,
            'available_gb': self.available_gb,
            'utilization': self.utilization,
            'models': self.models
        }


@dataclass
class PlacementPlan:
    """Model placement plan across workers."""

    worker_placements: Dict[str, List[str]]  # worker_id → [model_ids]
    model_locations: Dict[str, List[str]]  # model_id → [worker_ids]
    worker_storage: Dict[str, WorkerStorage]
    total_models_placed: int
    total_replicas_placed: int
    models_not_placed: List[str] = field(default_factory=list)

    @property
    def cluster_utilization(self) -> float:
        """Average storage utilization across cluster."""
        if not self.worker_storage:
            return 0.0

        return sum(w.utilization for w in self.worker_storage.values()) / len(self.worker_storage)

    def get_model_replication_factor(self, model_id: str) -> int:
        """Get number of replicas for a model."""
        return len(self.model_locations.get(model_id, []))

    def to_dict(self) -> dict:
        return {
            'worker_placements': self.worker_placements,
            'model_locations': self.model_locations,
            'worker_storage': {k: v.to_dict() for k, v in self.worker_storage.items()},
            'total_models_placed': self.total_models_placed,
            'total_replicas_placed': self.total_replicas_placed,
            'models_not_placed': self.models_not_placed,
            'cluster_utilization': self.cluster_utilization
        }

    def save(self, filepath: str):
        """Save placement plan to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"Saved placement plan to {filepath}")

    def print_summary(self):
        """Print placement summary."""
        print("=" * 70)
        print("MODEL PLACEMENT PLAN")
        print("=" * 70)
        print(f"Total unique models: {self.total_models_placed}")
        print(f"Total replicas placed: {self.total_replicas_placed}")
        print(f"Cluster utilization: {self.cluster_utilization:.1%}")

        if self.models_not_placed:
            print(f"\nModels not placed (insufficient storage): {len(self.models_not_placed)}")
            for model in self.models_not_placed[:5]:
                print(f"  - {model}")
            if len(self.models_not_placed) > 5:
                print(f"  ... and {len(self.models_not_placed) - 5} more")

        print(f"\nWorker storage:")
        for worker_id, storage in sorted(self.worker_storage.items()):
            print(f"  {worker_id}: {storage.used_gb:.1f}/{storage.capacity_gb:.1f} GB "
                  f"({storage.utilization:.1%}) - {len(storage.models)} models")

        print("=" * 70)


class ModelPlacer:
    """
    Model placement strategy for serverless LLM clusters.

    Implements:
    1. Popularity-based replication (more replicas for popular models)
    2. Round-robin placement across workers
    3. Storage capacity constraints
    """

    def __init__(
        self,
        default_model_size_gb: float = 13.0,
        model_sizes: Optional[Dict[str, float]] = None
    ):
        """
        Initialize model placer.

        Args:
            default_model_size_gb: Default model size if not specified
            model_sizes: Dictionary of model_id → size_gb overrides
        """
        self.default_model_size_gb = default_model_size_gb
        self.model_sizes = model_sizes or {}

    def get_model_size(self, model_id: str) -> float:
        """Get model size in GB."""
        return self.model_sizes.get(model_id, self.default_model_size_gb)

    def calculate_replicas(
        self,
        models: List[str],
        popularity: Dict[str, float],
        total_replicas: int,
        min_replicas: int = 1,
        max_replicas: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Calculate number of replicas for each model based on popularity.

        Args:
            models: List of model IDs
            popularity: Dictionary of model_id → popularity_score (0-1, sum to 1)
            total_replicas: Total number of replica slots to distribute
            min_replicas: Minimum replicas per model
            max_replicas: Maximum replicas per model (None = unlimited)

        Returns:
            Dictionary of model_id → num_replicas
        """
        # Start with minimum replicas for all
        replicas = {model: min_replicas for model in models}
        remaining = total_replicas - (len(models) * min_replicas)

        if remaining <= 0:
            return replicas

        # Distribute remaining slots proportional to popularity
        sorted_models = sorted(models, key=lambda m: popularity.get(m, 0), reverse=True)

        for model in sorted_models:
            if remaining <= 0:
                break

            # Allocate proportional to popularity
            pop_score = popularity.get(model, 0)
            additional = int(pop_score * remaining * 2)  # Scale factor for more aggressive replication

            if max_replicas:
                additional = min(additional, max_replicas - replicas[model])

            if additional > 0:
                replicas[model] += additional
                remaining -= additional

        return replicas

    def place_models(
        self,
        models: List[str],
        workers: List[str],
        popularity: Dict[str, float],
        storage_per_worker_gb: float,
        total_replicas: Optional[int] = None,
        placement_strategy: str = "round_robin",
        seed: Optional[int] = None
    ) -> PlacementPlan:
        """
        Place models across workers using popularity-based replication.

        Args:
            models: List of model IDs to place
            workers: List of worker IDs
            popularity: Dictionary of model_id → popularity_score
            storage_per_worker_gb: Storage capacity per worker in GB
            total_replicas: Total replica slots (if None, fill to capacity)
            placement_strategy: Placement strategy ("round_robin" or "random")
            seed: Random seed for reproducibility

        Returns:
            PlacementPlan with model assignments
        """
        if seed is not None:
            random.seed(seed)

        # Initialize worker storage
        worker_storage = {
            worker_id: WorkerStorage(worker_id=worker_id, capacity_gb=storage_per_worker_gb)
            for worker_id in workers
        }

        # Calculate total available replicas if not specified
        if total_replicas is None:
            total_capacity = len(workers) * storage_per_worker_gb
            avg_model_size = sum(self.get_model_size(m) for m in models) / len(models)
            total_replicas = int(total_capacity / avg_model_size)

        # Calculate replicas per model
        model_replicas = self.calculate_replicas(
            models=models,
            popularity=popularity,
            total_replicas=total_replicas,
            min_replicas=1
        )

        # Create list of (model_id, size_gb) to place, with replicas
        placement_queue = []
        for model_id, num_replicas in model_replicas.items():
            size_gb = self.get_model_size(model_id)
            for _ in range(num_replicas):
                placement_queue.append((model_id, size_gb))

        # Sort by popularity (place popular models first to ensure they fit)
        placement_queue.sort(
            key=lambda x: popularity.get(x[0], 0),
            reverse=True
        )

        # Place models using selected strategy
        if placement_strategy == "round_robin":
            worker_placements, model_locations, not_placed = self._place_round_robin(
                placement_queue, worker_storage
            )
        elif placement_strategy == "random":
            worker_placements, model_locations, not_placed = self._place_random(
                placement_queue, worker_storage
            )
        else:
            raise ValueError(f"Unknown placement strategy: {placement_strategy}")

        # Create placement plan
        unique_models_placed = len(model_locations)
        total_replicas_placed = sum(len(locs) for locs in model_locations.values())

        plan = PlacementPlan(
            worker_placements=worker_placements,
            model_locations=model_locations,
            worker_storage=worker_storage,
            total_models_placed=unique_models_placed,
            total_replicas_placed=total_replicas_placed,
            models_not_placed=list(set(not_placed))
        )

        return plan

    def _place_round_robin(
        self,
        placement_queue: List[Tuple[str, float]],
        worker_storage: Dict[str, WorkerStorage]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str]]:
        """
        Place models using round-robin strategy.

        Returns:
            (worker_placements, model_locations, not_placed)
        """
        workers = list(worker_storage.keys())
        worker_idx = 0

        worker_placements = {w: [] for w in workers}
        model_locations = {}
        not_placed = []

        for model_id, size_gb in placement_queue:
            placed = False

            # Try each worker starting from current index
            for _ in range(len(workers)):
                worker_id = workers[worker_idx]
                storage = worker_storage[worker_id]

                if storage.add_model(model_id, size_gb):
                    worker_placements[worker_id].append(model_id)
                    if model_id not in model_locations:
                        model_locations[model_id] = []
                    model_locations[model_id].append(worker_id)
                    placed = True
                    worker_idx = (worker_idx + 1) % len(workers)
                    break

                worker_idx = (worker_idx + 1) % len(workers)

            if not placed:
                not_placed.append(model_id)

        return worker_placements, model_locations, not_placed

    def _place_random(
        self,
        placement_queue: List[Tuple[str, float]],
        worker_storage: Dict[str, WorkerStorage]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str]]:
        """
        Place models using random strategy.

        Returns:
            (worker_placements, model_locations, not_placed)
        """
        workers = list(worker_storage.keys())

        worker_placements = {w: [] for w in workers}
        model_locations = {}
        not_placed = []

        for model_id, size_gb in placement_queue:
            # Shuffle workers for random placement
            shuffled_workers = workers.copy()
            random.shuffle(shuffled_workers)

            placed = False
            for worker_id in shuffled_workers:
                storage = worker_storage[worker_id]

                if storage.add_model(model_id, size_gb):
                    worker_placements[worker_id].append(model_id)
                    if model_id not in model_locations:
                        model_locations[model_id] = []
                    model_locations[model_id].append(worker_id)
                    placed = True
                    break

            if not placed:
                not_placed.append(model_id)

        return worker_placements, model_locations, not_placed


def place_models(
    models: List[str],
    workers: List[str],
    popularity: Dict[str, float],
    storage_per_worker_gb: float = 100,
    model_sizes: Optional[Dict[str, float]] = None,
    total_replicas: Optional[int] = None,
    placement_strategy: str = "round_robin",
    seed: Optional[int] = None
) -> PlacementPlan:
    """
    Convenience function to place models across workers.

    Args:
        models: List of model IDs
        workers: List of worker IDs
        popularity: Model popularity scores
        storage_per_worker_gb: Storage capacity per worker
        model_sizes: Optional dictionary of model sizes (GB)
        total_replicas: Total replicas to distribute (None = fill to capacity)
        placement_strategy: "round_robin" or "random"
        seed: Random seed

    Returns:
        PlacementPlan
    """
    placer = ModelPlacer(model_sizes=model_sizes)

    plan = placer.place_models(
        models=models,
        workers=workers,
        popularity=popularity,
        storage_per_worker_gb=storage_per_worker_gb,
        total_replicas=total_replicas,
        placement_strategy=placement_strategy,
        seed=seed
    )

    plan.print_summary()
    return plan


if __name__ == "__main__":
    # Example usage
    print("Testing model placement...\n")

    # Sample models with Zipf popularity
    models = ["llama-7b", "llama-13b", "gpt-neo", "falcon-7b", "mistral-7b"]
    popularity = {
        "llama-7b": 0.35,
        "llama-13b": 0.25,
        "gpt-neo": 0.20,
        "falcon-7b": 0.12,
        "mistral-7b": 0.08
    }

    # Model sizes
    model_sizes = {
        "llama-7b": 13,
        "llama-13b": 26,
        "gpt-neo": 5,
        "falcon-7b": 14,
        "mistral-7b": 14
    }

    # Workers
    workers = ["worker-0", "worker-1", "worker-2"]

    # Place models
    plan = place_models(
        models=models,
        workers=workers,
        popularity=popularity,
        storage_per_worker_gb=100,
        model_sizes=model_sizes,
        placement_strategy="round_robin",
        seed=42
    )

    # Show details
    print("\nDetailed placement:")
    for worker_id, model_list in sorted(plan.worker_placements.items()):
        print(f"\n{worker_id}:")
        for model_id in model_list:
            size = model_sizes.get(model_id, 13)
            print(f"  - {model_id} ({size} GB)")
