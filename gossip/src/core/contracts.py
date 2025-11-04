"""
Data contracts for the centralized scheduler and worker coordination.

This module defines the data structures used for communication between workers
and the centralized coordinator in the serverless LLM cluster.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from enum import Enum
import json


class PlacementAction(Enum):
    """Actions that can be taken when placing a request."""
    SERVE = "serve"
    COLD_START = "cold_start"
    MIGRATE = "migrate"


@dataclass
class WorkerLoadReport:
    """
    Represents the current state of a worker node.

    This is sent periodically by workers to the centralized coordinator
    to maintain a real-time view of cluster state.

    Attributes:
        node_id: Unique identifier for the worker node
        loaded_models: List of model identifiers currently loaded in memory
        queue_depth: Number of pending requests in the worker's queue
        memory_utilization: Memory utilization as a fraction (0.0 to 1.0)
        is_ready: Whether worker has finished initialization and is ready for inference
        timestamp: Unix timestamp when this report was generated (optional)
    """
    node_id: str
    loaded_models: List[str]
    queue_depth: int
    memory_utilization: float
    is_ready: bool = False
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> 'WorkerLoadReport':
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'WorkerLoadReport':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def is_overloaded(self, queue_threshold: int = 10, memory_threshold: float = 0.9) -> bool:
        """Check if worker is overloaded based on thresholds."""
        return self.queue_depth > queue_threshold or self.memory_utilization > memory_threshold


@dataclass
class ScheduleRequest:
    """
    Request to schedule an inference task.

    Sent by an entry node to the centralized coordinator to find the best
    worker for serving an inference request.

    Attributes:
        request_id: Unique identifier for this inference request
        model_required: Model identifier needed for inference
        priority: Optional priority level (higher is more urgent)
    """
    request_id: str
    model_required: str
    priority: Optional[int] = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> 'ScheduleRequest':
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'ScheduleRequest':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ScheduleResponse:
    """
    Response from the centralized coordinator with placement decision.

    Attributes:
        worker_id: Identifier of the worker node that should handle the request
        action: The action to take (SERVE, COLD_START, or MIGRATE)
        estimated_wait_time: Optional estimated time until processing can begin (seconds)
        reason: Optional explanation for the placement decision
    """
    worker_id: str
    action: PlacementAction
    estimated_wait_time: Optional[float] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['action'] = self.action.value
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> 'ScheduleResponse':
        """Create from dictionary."""
        action = PlacementAction(data['action'])
        return cls(
            worker_id=data['worker_id'],
            action=action,
            estimated_wait_time=data.get('estimated_wait_time'),
            reason=data.get('reason')
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'ScheduleResponse':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# Example JSON formats for reference
EXAMPLE_WORKER_LOAD_REPORT = {
    "node_id": "w1",
    "loaded_models": ["m1", "m2"],
    "queue_depth": 3,
    "memory_utilization": 0.65,
    "is_ready": True,
    "timestamp": 1234567890.123
}

EXAMPLE_SCHEDULE_REQUEST = {
    "request_id": "r1",
    "model_required": "m3",
    "priority": 1
}

EXAMPLE_SCHEDULE_RESPONSE = {
    "worker_id": "wX",
    "action": "serve",
    "estimated_wait_time": 2.5,
    "reason": "Worker has model loaded with low queue depth"
}


@dataclass
class InferenceRequest:
    """
    Request to perform inference on a worker.

    Attributes:
        request_id: Unique identifier for this inference request
        model_id: Model identifier to use for inference
        prompt: Input text prompt for the model
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 = deterministic)
    """
    request_id: str
    model_id: str
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> 'InferenceRequest':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class InferenceResponse:
    """
    Response from inference execution.

    Attributes:
        request_id: Unique identifier for this inference request
        worker_id: Worker that performed the inference
        output_text: Generated text from the model
        num_tokens: Number of tokens generated
        latency_ms: Inference latency in milliseconds
        success: Whether inference completed successfully
        error: Optional error message if failed
    """
    request_id: str
    worker_id: str
    output_text: str
    num_tokens: int
    latency_ms: float
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> 'InferenceResponse':
        """Create from dictionary."""
        return cls(**data)
