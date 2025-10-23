"""
Configuration management for containerized deployment.

Supports environment variables for Docker/Kubernetes deployments.
"""

import os
from typing import Optional


class Config:
    """Configuration class that reads from environment variables."""

    # Network Configuration
    BIND_HOST: str = os.getenv("BIND_HOST", "0.0.0.0")  # Bind to all interfaces in containers
    WORKER_PORT: int = int(os.getenv("WORKER_PORT", "8000"))
    COORDINATOR_PORT: int = int(os.getenv("COORDINATOR_PORT", "9000"))

    # Service Discovery
    COORDINATOR_HOST: str = os.getenv("COORDINATOR_HOST", "central-coordinator")  # DNS name in Docker
    WORKER_ID: Optional[str] = os.getenv("WORKER_ID", None)  # Auto-generated if not set

    # Timing Configuration
    REPORT_INTERVAL: float = float(os.getenv("REPORT_INTERVAL", "1.0"))
    WORKER_TIMEOUT: float = float(os.getenv("WORKER_TIMEOUT", "10.0"))
    GOSSIP_INTERVAL: float = float(os.getenv("GOSSIP_INTERVAL", "1.0"))

    # Resource Thresholds
    QUEUE_THRESHOLD: int = int(os.getenv("QUEUE_THRESHOLD", "10"))
    MEMORY_THRESHOLD: float = float(os.getenv("MEMORY_THRESHOLD", "0.9"))
    GPU_THRESHOLD: float = float(os.getenv("GPU_THRESHOLD", "0.9"))  # Legacy, kept for backward compatibility

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
    )

    # Deployment Mode
    DEPLOYMENT_MODE: str = os.getenv("DEPLOYMENT_MODE", "local")  # local, docker, kubernetes

    # Model Loading Configuration
    USE_REAL_MODELS: bool = os.getenv("USE_REAL_MODELS", "false").lower() == "true"
    GCS_BUCKET: str = os.getenv("GCS_BUCKET", "remote_model")
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/tmp/model_cache")
    PYTORCH_DEVICE: str = os.getenv("PYTORCH_DEVICE", "cpu")  # cpu, cuda, cuda:0, etc.

    @classmethod
    def get_worker_id(cls) -> str:
        """Get or generate worker ID."""
        if cls.WORKER_ID:
            return cls.WORKER_ID

        # Try to get from hostname (useful in Kubernetes)
        hostname = os.getenv("HOSTNAME", "worker")

        # In K8s StatefulSets, hostname will be like: worker-0, worker-1, etc.
        return hostname

    @classmethod
    def get_coordinator_address(cls) -> tuple[str, int]:
        """Get coordinator address based on deployment mode."""
        return (cls.COORDINATOR_HOST, cls.COORDINATOR_PORT)

    @classmethod
    def is_containerized(cls) -> bool:
        """Check if running in a container."""
        return cls.DEPLOYMENT_MODE in ["docker", "kubernetes"]

    @classmethod
    def print_config(cls):
        """Print current configuration (useful for debugging)."""
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(f"Deployment Mode: {cls.DEPLOYMENT_MODE}")
        print(f"Bind Host: {cls.BIND_HOST}")
        print(f"Worker Port: {cls.WORKER_PORT}")
        print(f"Coordinator: {cls.COORDINATOR_HOST}:{cls.COORDINATOR_PORT}")
        print(f"Worker ID: {cls.get_worker_id()}")
        print(f"Report Interval: {cls.REPORT_INTERVAL}s")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print("=" * 60)


# Singleton instance
config = Config()
