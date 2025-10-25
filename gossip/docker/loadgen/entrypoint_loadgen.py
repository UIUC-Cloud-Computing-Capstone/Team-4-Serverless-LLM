#!/usr/bin/env python3
"""
Entrypoint for the load generator container.

Reads configuration from environment variables and starts the load generator.
"""

import os
import sys
import logging
from load_generator import LoadGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("LoadGenEntrypoint")


def main():
    """Start the load generator with environment-based configuration."""

    # Read configuration from environment
    coordinator_host = os.getenv("COORDINATOR_HOST", "central-coordinator")
    coordinator_port = int(os.getenv("COORDINATOR_PORT", "9000"))
    target_rps = float(os.getenv("TARGET_RPS", "10.0"))
    cv = float(os.getenv("CV", "8.0"))
    duration = float(os.getenv("DURATION", "300.0"))
    gsm8k_path = os.getenv("GSM8K_PATH")
    sharegpt_path = os.getenv("SHAREGPT_PATH")

    logger.info("=" * 70)
    logger.info("LOAD GENERATOR STARTING")
    logger.info("=" * 70)
    logger.info(f"Coordinator: {coordinator_host}:{coordinator_port}")
    logger.info(f"Target RPS: {target_rps}")
    logger.info(f"Coefficient of Variation: {cv}")
    logger.info(f"Duration: {duration}s")
    logger.info(f"GSM8K path: {gsm8k_path or 'None (will use synthetic data)'}")
    logger.info(f"ShareGPT path: {sharegpt_path or 'None (will use synthetic data)'}")
    logger.info("=" * 70)

    # Create and run load generator
    try:
        generator = LoadGenerator(
            coordinator_host=coordinator_host,
            coordinator_port=coordinator_port,
            target_rps=target_rps,
            cv=cv,
            duration=duration,
            gsm8k_path=gsm8k_path,
            sharegpt_path=sharegpt_path
        )

        generator.run()

    except KeyboardInterrupt:
        logger.info("\nLoad generation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Load generator failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
