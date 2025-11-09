"""
Load generator for sending inference requests to the central coordinator system.

Generates requests from datasets with bursty patterns and sends them to workers
who then request scheduling from the central coordinator.
"""

import time
import random
import logging
import sys
import argparse
from typing import List, Dict
from dataset_loader import DatasetLoader
from workload_generator import WorkloadGenerator, RequestTrace
from worker_node import WorkerNode

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("LoadGenerator")


class LoadGenerator:
    """
    Generates inference requests and sends them to the cluster.

    Uses datasets and workload traces to create realistic request patterns.
    """

    def __init__(
        self,
        coordinator_host: str = "central-coordinator",
        coordinator_port: int = 9000,
        target_rps: float = 10.0,
        cv: float = 8.0,
        duration: float = 300.0,
        gsm8k_path: str = None,
        sharegpt_path: str = None
    ):
        """
        Initialize load generator.

        Args:
            coordinator_host: Coordinator hostname
            coordinator_port: Coordinator port
            target_rps: Target requests per second
            cv: Coefficient of variation for burstiness
            duration: Duration to run (seconds)
            gsm8k_path: Path to GSM8K dataset
            sharegpt_path: Path to ShareGPT dataset
        """
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.target_rps = target_rps
        self.cv = cv
        self.duration = duration

        logger.info(f"Initializing load generator: RPS={target_rps}, CV={cv}, Duration={duration}s")

        # Load datasets
        logger.info("Loading datasets...")
        self.dataset_loader = DatasetLoader(max_tokens=2048, samples_per_dataset=4000)
        self.dataset_loader.load_gsm8k(gsm8k_path)
        self.dataset_loader.load_sharegpt(sharegpt_path)
        self.mixed_workload = self.dataset_loader.create_mixed_workload()

        stats = self.dataset_loader.get_dataset_stats()
        logger.info(f"Dataset loaded: {stats['gsm8k_count']} GSM8K + {stats['sharegpt_count']} ShareGPT")

        # Generate workload trace
        logger.info("Generating workload trace...")
        self.workload_generator = WorkloadGenerator(
            target_rps=target_rps,
            cv=cv,
            duration=duration,
            models=['facebook/opt-125m'],  # Using the small model we have loaded
            seed=42
        )
        self.trace = self.workload_generator.generate_trace(dataset_samples=self.mixed_workload)

        trace_stats = self.workload_generator.get_trace_stats()
        logger.info(
            f"Trace generated: {trace_stats['num_requests']} requests, "
            f"actual RPS={trace_stats['actual_rps']:.2f}, CV={trace_stats['cv_inter_arrival']:.2f}"
        )

        # Create a client worker for sending requests
        self.client = WorkerNode(
            node_id="load-generator",
            host="0.0.0.0",
            port=7999,
            coordinator_host=coordinator_host,
            coordinator_port=coordinator_port,
            verbose=False,
            use_real_models=False
        )

        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def wait_for_ready_workers(self, min_workers: int = 3, timeout: int = 300):
        """
        Wait for workers to report as ready before starting load generation.
        
        Args:
            min_workers: Minimum number of workers that must be ready
            timeout: Maximum time to wait in seconds
        """
        logger.info(f"Waiting for at least {min_workers} workers to be ready (timeout: {timeout}s)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Query coordinator for cluster state
                response = self.client.request_schedule(
                    request_id="readiness-check",
                    model_required="facebook/opt-125m",  # The model we have loaded
                    timeout=1.0
                )
                
                if response and hasattr(response, 'worker_id'):
                    # If we get a response, check if it's from a real worker (not load-generator)
                    if response.worker_id != "load-generator":
                        logger.info(f"Found ready worker: {response.worker_id}")
                        return True
                        
            except Exception as e:
                logger.debug(f"Readiness check failed: {e}")
            
            logger.info("Waiting for workers to be ready...")
            time.sleep(5)
        
        logger.warning(f"Timeout waiting for workers to be ready after {timeout}s")
        return False

    def run(self):
        """Execute the load generation."""
        logger.info("=" * 70)
        logger.info("STARTING LOAD GENERATION")
        logger.info("=" * 70)

        # Start client
        self.client.start()
        time.sleep(1)

        # Wait for workers to be ready
        if not self.wait_for_ready_workers(min_workers=3, timeout=300):
            logger.warning("Proceeding without waiting for workers to be ready")

        logger.info(f"Sending {len(self.trace)} requests over {self.duration}s...")
        logger.info(f"Target RPS: {self.target_rps}, CV: {self.cv}")
        logger.info("=" * 70)

        start_time = time.time()

        for req in self.trace:
            # Wait until it's time for this request
            current_time = time.time() - start_time
            wait_time = req.timestamp - current_time

            if wait_time > 0:
                time.sleep(wait_time)

            # Check if we've exceeded duration
            if time.time() - start_time > self.duration:
                logger.info("Duration exceeded, stopping load generation")
                break

            # Step 1: Send schedule request to coordinator
            schedule_response = self.client.request_schedule(
                request_id=req.request_id,
                model_required=req.model_id,
                timeout=2.0
            )

            self.total_requests += 1

            if schedule_response:
                # Track cache hits/misses based on scheduling action
                if schedule_response.action.value == "serve":
                    self.cache_hits += 1
                elif schedule_response.action.value == "cold_start":
                    self.cache_misses += 1

                # Step 2: Send actual inference request to the assigned worker
                # Get the prompt from the dataset sample
                prompt_text = req.prompt if hasattr(req, 'prompt') else "What is the capital of France?"

                inference_response = self.client.request_inference(
                    worker_host=schedule_response.worker_id,  # Use worker_id as hostname
                    worker_port=8000,
                    request_id=req.request_id,
                    model_id=req.model_id,
                    prompt=prompt_text[:500],  # Truncate prompt to 500 chars for demo
                    max_tokens=30,
                    temperature=0.0,
                    timeout=90.0
                )

                if inference_response and inference_response.success:
                    self.successful_requests += 1

                    # Log every 10th request with full details
                    if self.total_requests % 10 == 0:
                        elapsed = time.time() - start_time
                        logger.info("=" * 80)
                        logger.info(f"[{elapsed:.1f}s] REQUEST {req.request_id} | Model: {req.model_id}")
                        logger.info(f"  Schedule: {schedule_response.worker_id} ({schedule_response.action.value})")
                        logger.info(f"  Prompt: {prompt_text[:100]}...")
                        logger.info(f"  Output: {inference_response.output_text[:150]}...")
                        logger.info(f"  Tokens: {inference_response.num_tokens} | Latency: {inference_response.latency_ms:.0f}ms")
                        logger.info(f"  Stats: Total={self.total_requests}, Success={100*self.successful_requests/self.total_requests:.1f}%")
                        logger.info("=" * 80)
                    elif self.total_requests % 100 == 0:
                        # Summary every 100
                        elapsed = time.time() - start_time
                        logger.info(
                            f"[{elapsed:.1f}s] Milestone {self.total_requests}: "
                            f"Success={100*self.successful_requests/self.total_requests:.1f}%, "
                            f"Avg latency={inference_response.latency_ms:.0f}ms"
                        )
                else:
                    self.failed_requests += 1
                    if self.failed_requests % 10 == 0:
                        logger.warning(f"Failed inference count: {self.failed_requests}")
            else:
                # Schedule request failed
                self.failed_requests += 1
                if self.failed_requests % 10 == 0:
                    logger.warning(f"Failed schedule request count: {self.failed_requests}")

        # Final statistics
        elapsed = time.time() - start_time
        self.print_statistics(elapsed)

        # Stop client
        self.client.stop()

    def print_statistics(self, elapsed: float):
        """Print final statistics."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("LOAD GENERATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {elapsed:.2f}s")
        logger.info(f"Total requests: {self.total_requests}")
        logger.info(f"Successful: {self.successful_requests}")
        logger.info(f"Failed: {self.failed_requests}")
        logger.info(f"Success rate: {100 * self.successful_requests / max(1, self.total_requests):.2f}%")
        logger.info(f"Actual RPS: {self.total_requests / elapsed:.2f}")
        logger.info(f"Cache hits: {self.cache_hits}")
        logger.info(f"Cache misses: {self.cache_misses}")
        logger.info(f"Cache hit rate: {100 * self.cache_hits / max(1, self.cache_hits + self.cache_misses):.2f}%")
        logger.info("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load Generator for Central Coordinator")
    parser.add_argument("--coordinator-host", default="central-coordinator", help="Coordinator host")
    parser.add_argument("--coordinator-port", type=int, default=9000, help="Coordinator port")
    parser.add_argument("--rps", type=float, default=10.0, help="Target requests per second")
    parser.add_argument("--cv", type=float, default=8.0, help="Coefficient of variation")
    parser.add_argument("--duration", type=float, default=300.0, help="Duration in seconds")
    parser.add_argument("--gsm8k", type=str, help="Path to GSM8K dataset")
    parser.add_argument("--sharegpt", type=str, help="Path to ShareGPT dataset")

    args = parser.parse_args()

    generator = LoadGenerator(
        coordinator_host=args.coordinator_host,
        coordinator_port=args.coordinator_port,
        target_rps=args.rps,
        cv=args.cv,
        duration=args.duration,
        gsm8k_path=args.gsm8k,
        sharegpt_path=args.sharegpt
    )

    try:
        generator.run()
    except KeyboardInterrupt:
        logger.info("\nLoad generation interrupted by user")
        generator.print_statistics(time.time())
        sys.exit(0)


if __name__ == "__main__":
    main()
