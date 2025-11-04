#!/usr/bin/env python3
"""
Container-aware entrypoint for worker nodes.

Handles configuration from environment variables and provides
better logging for containerized environments.
"""

import sys
import signal
import logging
from config import config
from worker_node import WorkerNode

# Configure logging for containers
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    stream=sys.stdout  # Important: stdout for container logs
)

logger = logging.getLogger("WorkerEntrypoint")


def main():
    """Main entrypoint for containerized worker."""

    # Print configuration
    if config.LOG_LEVEL == "DEBUG":
        config.print_config()

    # Get configuration
    worker_id = config.get_worker_id()
    coordinator_host, coordinator_port = config.get_coordinator_address()

    logger.info(f"Starting worker: {worker_id}")
    logger.info(f"Coordinator: {coordinator_host}:{coordinator_port}")

    # Create worker node
    worker = WorkerNode(
        node_id=worker_id,
        host=config.BIND_HOST,
        port=config.WORKER_PORT,
        coordinator_host=coordinator_host,
        coordinator_port=coordinator_port,
        report_interval=config.REPORT_INTERVAL,
        verbose=(config.LOG_LEVEL == "DEBUG"),
        use_real_models=config.USE_REAL_MODELS,
        gcs_bucket=config.GCS_BUCKET,
        cache_dir=config.MODEL_CACHE_DIR,
        device=config.PYTORCH_DEVICE
    )

    # Setup signal handlers for graceful shutdown
    def shutdown_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    # Initialize worker with models BEFORE starting
    # This ensures the worker is ready for inference before reporting to coordinator
    if config.DEPLOYMENT_MODE == "docker" or config.DEPLOYMENT_MODE == "kubernetes":
        import random

        models_to_load = []

        # IMPORTANT: Using OPT-125M for testing
        # This is a very small model (125M params) that's good for testing
        # and doesn't require authentication
        available_models = ["facebook/opt-125m"]

        # All workers will load the same small model for now
        model_to_load = available_models[0]
        models_to_load = [model_to_load]

        if config.USE_REAL_MODELS:
            logger.info(f"Worker will initialize with REAL model: {model_to_load}")
        else:
            logger.info(f"Worker will initialize with SIMULATED model: {model_to_load} (for demo purposes)")

        # Initialize worker (loads models and marks ready)
        logger.info("Initializing worker...")
        init_time = worker.initialize(models_to_load=models_to_load)
        logger.info(f"Worker initialized in {init_time:.2f}s with models: {worker.get_loaded_models()}")

        # Set initial state
        worker.set_queue_depth(0)  # Start with empty queue

        # Get actual memory utilization from system
        if config.USE_REAL_MODELS:
            actual_memory = worker._get_memory_utilization()
            logger.info(f"Actual memory utilization after model loading: {actual_memory:.2%}")
        else:
            # Only use random values in simulation mode
            worker.set_memory_utilization(random.uniform(0.1, 0.3))
    else:
        logger.warning("Not in docker/kubernetes mode, skipping model initialization")

    # Start worker (now that it's initialized)
    worker.start()
    logger.info(f"Worker {worker_id} running on {config.BIND_HOST}:{config.WORKER_PORT}")
    logger.info(f"Worker ready: {worker.is_ready}, models loaded: {worker.get_loaded_models()}")

    # Keep running
    logger.info("Worker ready and reporting to coordinator")
    signal.pause()  # Wait for signals


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
