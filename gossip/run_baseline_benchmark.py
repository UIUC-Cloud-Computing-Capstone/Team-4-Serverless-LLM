#!/usr/bin/env python3
"""
Baseline benchmark runner for serverless LLM scheduling.

This script implements the baseline workload configuration from research papers:
- Datasets: GSM8K + ShareGPT (4K samples each, 2048 tokens)
- Workload: Gamma distribution CV=8 (bursty, Azure Serverless-like)
- Model placement: Popularity-based replication, round-robin

Usage:
    # Start coordinator and workers first
    python centralized_simulation.py --workers 3 --duration 300

    # Run baseline benchmark (in another terminal)
    python run_baseline_benchmark.py --rps 10 --duration 60
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from workload.datasets import load_mixed_dataset
from workload.trace_generator import generate_trace
from workload.model_placement import place_models
from workload.benchmark import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline serverless LLM benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Workload parameters
    parser.add_argument(
        "--duration",
        type=float,
        default=60,
        help="Benchmark duration in seconds"
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=10,
        help="Average requests per second"
    )
    parser.add_argument(
        "--cv",
        type=float,
        default=8.0,
        help="Coefficient of variation (burstiness). CV=8 matches Azure Serverless patterns"
    )

    # Dataset parameters
    parser.add_argument(
        "--gsm8k-samples",
        type=int,
        default=4000,
        help="Number of GSM8K samples (math reasoning)"
    )
    parser.add_argument(
        "--sharegpt-samples",
        type=int,
        default=4000,
        help="Number of ShareGPT samples (chat)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum context length for truncation"
    )

    # Model configuration
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama-7b", "llama-13b", "gpt-neo", "falcon-7b", "mistral-7b"],
        help="List of model IDs to use"
    )
    parser.add_argument(
        "--popularity-alpha",
        type=float,
        default=1.2,
        help="Zipf distribution parameter for model popularity (higher = more skewed)"
    )

    # Placement configuration
    parser.add_argument(
        "--workers",
        nargs="+",
        default=["worker-0", "worker-1", "worker-2"],
        help="List of worker IDs"
    )
    parser.add_argument(
        "--storage-per-worker",
        type=float,
        default=100,
        help="Storage capacity per worker (GB)"
    )
    parser.add_argument(
        "--placement-strategy",
        choices=["round_robin", "random"],
        default="round_robin",
        help="Model placement strategy"
    )

    # Coordinator connection
    parser.add_argument(
        "--coordinator-host",
        default="127.0.0.1",
        help="Central coordinator host"
    )
    parser.add_argument(
        "--coordinator-port",
        type=int,
        default=9000,
        help="Central coordinator port"
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=5.0,
        help="Timeout for scheduling requests (seconds)"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        default="results/baseline",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-trace",
        action="store_true",
        help="Save generated trace to file"
    )
    parser.add_argument(
        "--save-placement",
        action="store_true",
        help="Save placement plan to file"
    )

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-request information"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare workload but don't run benchmark"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("BASELINE SERVERLESS LLM BENCHMARK")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Duration: {args.duration}s")
    print(f"  Target RPS: {args.rps}")
    print(f"  Burstiness (CV): {args.cv}")
    print(f"  Models: {len(args.models)}")
    print(f"  Workers: {len(args.workers)}")
    print(f"  Dataset: GSM8K ({args.gsm8k_samples}) + ShareGPT ({args.sharegpt_samples})")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Coordinator: {args.coordinator_host}:{args.coordinator_port}")
    print(f"  Output: {args.output_dir}")
    print("=" * 70)

    # Step 1: Load datasets
    print("\n[1/4] Loading datasets...")
    dataset = load_mixed_dataset(
        gsm8k_samples=args.gsm8k_samples,
        sharegpt_samples=args.sharegpt_samples,
        max_tokens=args.max_tokens,
        shuffle=True
    )

    # Step 2: Generate trace
    print("\n[2/4] Generating workload trace...")
    trace = generate_trace(
        duration=args.duration,
        mean_rps=args.rps,
        cv=args.cv,
        models=args.models,
        popularity_alpha=args.popularity_alpha,
        seed=args.seed
    )

    if args.save_trace:
        trace_path = f"{args.output_dir}/trace.json"
        trace.save(trace_path)

    # Step 3: Plan model placement
    print("\n[3/4] Planning model placement...")

    # Define model sizes (you can customize these)
    model_sizes = {
        "llama-7b": 13,
        "llama-13b": 26,
        "llama-30b": 60,
        "llama-65b": 120,
        "gpt-neo": 5,
        "gpt-neo-2.7b": 5,
        "falcon-7b": 14,
        "falcon-40b": 80,
        "mistral-7b": 14,
        "mixtral-8x7b": 90,
    }

    placement_plan = place_models(
        models=args.models,
        workers=args.workers,
        popularity=trace.model_popularity,
        storage_per_worker_gb=args.storage_per_worker,
        model_sizes=model_sizes,
        placement_strategy=args.placement_strategy,
        seed=args.seed
    )

    if args.save_placement:
        placement_path = f"{args.output_dir}/placement.json"
        placement_plan.save(placement_path)

    # Print initial placement instructions
    print("\n" + "=" * 70)
    print("INITIAL MODEL PLACEMENT")
    print("=" * 70)
    print("To initialize workers with this placement, load models as follows:\n")

    for worker_id, model_list in sorted(placement_plan.worker_placements.items()):
        print(f"{worker_id}:")
        for model_id in model_list:
            print(f"  worker.load_model('{model_id}')")
        print()

    if args.dry_run:
        print("Dry run complete. Exiting without running benchmark.")
        return

    # Step 4: Run benchmark
    print("\n[4/4] Running benchmark...")

    runner = BenchmarkRunner(
        coordinator_host=args.coordinator_host,
        coordinator_port=args.coordinator_port,
        request_timeout=args.request_timeout
    )

    # Combine trace with dataset
    print("\nCombining trace with dataset prompts...")
    benchmark_requests = []
    dataset_shuffled = dataset.copy()

    import random
    random.seed(args.seed)
    random.shuffle(dataset_shuffled)

    for i, trace_req in enumerate(trace.requests):
        dataset_req = dataset_shuffled[i % len(dataset_shuffled)]

        from workload.benchmark import BenchmarkRequest
        benchmark_req = BenchmarkRequest(
            request_id=trace_req.request_id,
            timestamp=trace_req.timestamp,
            model_id=trace_req.model_id,
            prompt=dataset_req.prompt,
            dataset_source=dataset_req.dataset_source,
            token_count=dataset_req.token_count,
            priority=trace_req.priority
        )
        benchmark_requests.append(benchmark_req)

    print(f"Prepared {len(benchmark_requests)} benchmark requests")

    # Run the benchmark
    try:
        results, metrics = runner.run_benchmark(
            requests=benchmark_requests,
            output_dir=args.output_dir,
            verbose=args.verbose
        )

        # Print summary
        print("\n")
        metrics.print_summary()

        print(f"\nResults saved to: {args.output_dir}/")
        print(f"  - results.csv: Detailed per-request results")
        print(f"  - metrics.json: Aggregated metrics")

        # Print conclusion
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)
        print(f"Cache hit rate: {metrics.cache_hit_rate:.1%}")
        print(f"Mean scheduling latency: {metrics.mean_scheduling_latency*1000:.2f} ms")
        print(f"P95 scheduling latency: {metrics.p95_scheduling_latency*1000:.2f} ms")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running benchmark: {e}")
        print("\nMake sure the coordinator is running:")
        print("  python centralized_simulation.py --workers 3 --duration 300")
        sys.exit(1)


if __name__ == "__main__":
    main()
