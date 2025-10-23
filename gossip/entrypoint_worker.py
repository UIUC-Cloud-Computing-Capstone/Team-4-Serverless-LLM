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

    # Start worker
    worker.start()
    logger.info(f"Worker {worker_id} running on {config.BIND_HOST}:{config.WORKER_PORT}")

    # For demo purposes, load some models
    if config.DEPLOYMENT_MODE == "docker" or config.DEPLOYMENT_MODE == "kubernetes":
        import random

        if config.USE_REAL_MODELS:
            # Load actual PyTorch models from GCS
            available_models = ["opt-1.3b", "opt-2.7b"]
            # Each worker loads one random model
            model_to_load = random.choice(available_models)
            logger.info(f"Loading real model: {model_to_load}")
            success = worker.load_model(model_to_load)
            if success:
                logger.info(f"Successfully loaded real model: {model_to_load}")
            else:
                logger.error(f"Failed to load real model: {model_to_load}")
        else:
            # Simulated mode
            models = ["llama-7b", "llama-13b", "gpt-neo", "falcon-7b", "mistral-7b"]
            num_models = random.randint(1, 3)
            for model in random.sample(models, num_models):
                worker.load_model(model)
                logger.info(f"Loaded model: {model}")

        # Set initial state
        worker.set_queue_depth(random.randint(0, 5))
        worker.set_gpu_utilization(random.uniform(0.2, 0.7))

    # Keep running
    logger.info("Worker ready and reporting to coordinator")
    signal.pause()  # Wait for signals


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
