#!/usr/bin/env python3
"""
Container-aware entrypoint for central coordinator.

Handles configuration from environment variables and provides
better logging for containerized environments.
"""

import sys
import signal
import time
import logging
from config import config
from udp_central_coordinator import UDPCentralCoordinator

# Configure logging for containers
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    stream=sys.stdout  # Important: stdout for container logs
)

logger = logging.getLogger("CoordinatorEntrypoint")


def main():
    """Main entrypoint for containerized central coordinator."""

    # Print configuration
    if config.LOG_LEVEL == "DEBUG":
        config.print_config()

    logger.info("Starting central coordinator")

    # Create central coordinator
    central_coordinator = UDPCentralCoordinator(
        coordinator_id="central-coordinator",
        host=config.BIND_HOST,
        port=config.COORDINATOR_PORT,
        worker_timeout=config.WORKER_TIMEOUT,
        verbose=(config.LOG_LEVEL == "DEBUG")
    )

    # Setup signal handlers for graceful shutdown
    def shutdown_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        logger.info("Final cluster state:")
        central_coordinator.print_cluster_state()
        central_coordinator.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    # Start coordinator
    central_coordinator.start()
    logger.info(f"Central coordinator running on {config.BIND_HOST}:{config.COORDINATOR_PORT}")
    logger.info("Waiting for workers to connect...")

    # Periodic status reporting
    try:
        while True:
            time.sleep(30)  # Every 30 seconds
            if config.LOG_LEVEL == "INFO":
                central_coordinator.print_cluster_state()
    except KeyboardInterrupt:
        shutdown_handler(signal.SIGINT, None)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
