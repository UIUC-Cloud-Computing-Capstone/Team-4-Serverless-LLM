"""
Simulation of the central coordinator architecture.

This script creates a central coordinator and multiple worker nodes to demonstrate
and test the centralized scheduling approach.
"""

import time
import random
import signal
import sys
from udp_central_coordinator import UDPCentralCoordinator
from worker_node import WorkerNode


def run_simulation(
    num_workers: int = 5,
    simulation_duration: int = 30,
    coordinator_port: int = 9000,
    base_worker_port: int = 8000
):
    """
    Run a simulation of the centralized coordinator system.

    Args:
        num_workers: Number of worker nodes to create
        simulation_duration: How long to run the simulation (seconds)
        coordinator_port: Port for the coordinator
        base_worker_port: Base port for workers (each gets base + index)
    """
    host = "127.0.0.1"

    print("=" * 70)
    print("CENTRAL COORDINATOR SIMULATION")
    print("=" * 70)
    print(f"Workers: {num_workers}")
    print(f"Duration: {simulation_duration} seconds")
    print(f"Central Coordinator: {host}:{coordinator_port}")
    print("=" * 70)

    # Create central coordinator
    print("\n--- Creating Central Coordinator ---")
    central_coordinator = UDPCentralCoordinator(
        coordinator_id="central-coordinator",
        host=host,
        port=coordinator_port,
        worker_timeout=10.0,
        verbose=False  # Set to True for detailed logs
    )
    central_coordinator.start()
    print(f"Central Coordinator started on {host}:{coordinator_port}")

    # Give central coordinator time to start
    time.sleep(0.5)

    # Create workers
    print(f"\n--- Creating {num_workers} Workers ---")
    workers = []
    available_models = ["llama-7b", "llama-13b", "gpt-neo", "falcon-7b", "mistral-7b"]

    for i in range(num_workers):
        worker = WorkerNode(
            node_id=f"worker-{i}",
            host=host,
            port=base_worker_port + i,
            coordinator_host=host,
            coordinator_port=coordinator_port,
            report_interval=1.0,
            verbose=False  # Set to True for detailed logs
        )

        # Initialize with some random models
        num_models = random.randint(1, 3)
        models_to_load = random.sample(available_models, num_models)

        print(f"Worker {worker.node_id}: Initializing with {models_to_load}...")
        init_time = worker.initialize(models_to_load=models_to_load)
        print(f"Worker {worker.node_id}: Initialized in {init_time:.2f}s")

        # Set initial workload
        worker.set_queue_depth(random.randint(0, 5))
        worker.set_memory_utilization(random.uniform(0.2, 0.7))

        workers.append(worker)
        worker.start()
        print(f"Worker {worker.node_id} started with models: {worker.get_loaded_models()}")

    # Give workers time to send initial reports
    time.sleep(2)

    # Graceful shutdown handler
    def shutdown(sig, frame):
        print("\n\n--- Shutting Down ---")
        print("\nFinal Cluster State:")
        central_coordinator.print_cluster_state()

        print("\nStopping workers...")
        for worker in workers:
            worker.stop()

        print("Stopping central coordinator...")
        central_coordinator.stop()

        print("\nSimulation ended successfully")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    # Main simulation loop
    print(f"\n--- Running Simulation for {simulation_duration} seconds ---")
    print("Simulating workload changes and scheduling requests...")
    print("Press Ctrl+C to stop early\n")

    start_time = time.time()
    request_count = 0

    try:
        while time.time() - start_time < simulation_duration:
            # Simulate some scheduling requests
            if random.random() < 0.3:  # 30% chance each iteration
                request_count += 1
                model = random.choice(available_models)
                requester = random.choice(workers)

                response = requester.request_schedule(
                    request_id=f"req-{request_count}",
                    model_required=model,
                    timeout=2.0
                )

                if response:
                    print(
                        f"Request req-{request_count} for {model}: "
                        f"scheduled to {response.worker_id} "
                        f"(action: {response.action.value})"
                    )

            # Simulate workload changes on workers
            for worker in workers:
                if random.random() < 0.2:  # 20% chance
                    worker.simulate_workload_change()

            # Occasionally print cluster state
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                print(f"\n--- Status at {int(elapsed)}s ---")
                central_coordinator.print_cluster_state()

            time.sleep(1)

    except KeyboardInterrupt:
        shutdown(None, None)

    # Simulation complete
    print(f"\n--- Simulation Complete ({simulation_duration}s) ---")
    print(f"Total scheduling requests: {request_count}")
    central_coordinator.print_cluster_state()

    # Cleanup
    print("\n--- Cleaning Up ---")
    for worker in workers:
        worker.stop()
    central_coordinator.stop()

    print("\nSimulation ended successfully")


def main():
    """Main entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Central Coordinator Simulation"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of worker nodes (default: 5)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Simulation duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--coordinator-port",
        type=int,
        default=9000,
        help="Central Coordinator port (default: 9000)"
    )
    parser.add_argument(
        "--base-worker-port",
        type=int,
        default=8000,
        help="Base port for workers (default: 8000)"
    )

    args = parser.parse_args()

    run_simulation(
        num_workers=args.workers,
        simulation_duration=args.duration,
        coordinator_port=args.coordinator_port,
        base_worker_port=args.base_worker_port
    )


if __name__ == "__main__":
    main()
