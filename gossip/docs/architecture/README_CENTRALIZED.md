# Central Coordinator Implementation

This directory contains the implementation of a central coordinator for comparison against the decentralized gossip-based approach.

## Architecture

The centralized system consists of:

1. **Central Coordinator** (`udp_central_coordinator.py`): Maintains global view of all worker states and makes optimal placement decisions
2. **Worker Nodes** (`worker_node.py`): Report their state to the central coordinator and request scheduling decisions
3. **Data Contracts** (`contracts.py`): Strongly-typed message formats for communication
4. **Optimal Placement Logic** (`central_coordinator.py`): Core scheduling algorithm

## Communication Protocol

All communication uses **UDP with pickle serialization** to match the existing gossip protocol architecture and minimize overhead.

### Message Types

**Worker Load Report:**
```python
{
    "type": "worker_report",
    "payload": {
        "node_id": "worker-1",
        "loaded_models": ["llama-7b", "gpt-neo"],
        "queue_depth": 3,
        "gpu_utilization": 0.65,
        "timestamp": 1234567890.123
    }
}
```

**Schedule Request:**
```python
{
    "type": "schedule_request",
    "payload": {
        "request_id": "req-1",
        "model_required": "llama-7b",
        "priority": 1
    }
}
```

**Schedule Response:**
```python
{
    "type": "schedule_response",
    "payload": {
        "worker_id": "worker-2",
        "action": "serve",  # or "cold_start", "migrate"
        "estimated_wait_time": 2.5,
        "reason": "Worker has model loaded with low queue depth"
    }
}
```

## Scheduling Algorithm

The central coordinator uses the following placement strategy:

1. **Cache Hit Path**: If workers have the required model loaded:
   - Calculate score based on queue depth and GPU utilization
   - Select worker with highest score (lowest load)
   - Return `SERVE` action

2. **Cache Miss Path**: If no worker has the model:
   - Select least loaded worker overall
   - Return `COLD_START` action
   - Account for cold start penalty in estimated wait time

### Scoring Function

```python
score = 1000.0 if has_model else 0.0
score -= (queue_depth * 10)
score -= (gpu_utilization * 100)
```

This heavily favors workers with the model loaded, then optimizes for low queue and GPU utilization.

## Usage

### Running the Simulation

The easiest way to test the system is with the simulation script:

```bash
python3 centralized_simulation.py --workers 5 --duration 30
```

Options:
- `--workers N`: Number of worker nodes (default: 5)
- `--duration S`: Simulation duration in seconds (default: 30)
- `--coordinator-port P`: Coordinator port (default: 9000)
- `--base-worker-port P`: Base port for workers (default: 8000)

### Manual Setup

**1. Start the Central Coordinator:**
```bash
python3 udp_central_coordinator.py --host 127.0.0.1 --port 9000 --verbose
```

**2. Start Workers:**
```bash
# Terminal 1
python3 worker_node.py --node-id worker-1 --port 8001 --coordinator-port 9000 --verbose

# Terminal 2
python3 worker_node.py --node-id worker-2 --port 8002 --coordinator-port 9000 --verbose

# Terminal 3
python3 worker_node.py --node-id worker-3 --port 8003 --coordinator-port 9000 --verbose
```

**3. Interact with Workers (in Python):**
```python
from worker_node import WorkerNode

# Create worker
worker = WorkerNode(
    node_id="test-worker",
    host="127.0.0.1",
    port=8010,
    coordinator_host="127.0.0.1",
    coordinator_port=9000
)
worker.start()

# Load a model
worker.load_model("llama-7b")

# Request scheduling
response = worker.request_schedule(
    request_id="req-1",
    model_required="llama-7b"
)
print(f"Scheduled to: {response.worker_id}")
```

## File Structure

```
gossip/
├── contracts.py                  # Data contracts (WorkerLoadReport, ScheduleRequest, ScheduleResponse)
├── central_coordinator.py        # Core coordinator logic with placement algorithm
├── udp_central_coordinator.py    # UDP server wrapper for central coordinator
├── worker_node.py                # Worker node implementation
├── centralized_simulation.py     # Simulation script
├── messages.proto                # Protobuf schema (documentation)
├── README_CENTRALIZED.md         # This file
│
# Existing gossip implementation:
├── udpnode.py                    # Decentralized gossip node
└── main.py                       # Gossip simulation
```

## Key Metrics for Experiments

The central coordinator tracks:
- **Cache hit rate**: Percentage of requests served without cold start
- **Total requests processed**
- **Active worker count**
- **Average GPU utilization** across cluster
- **Total queue depth** across cluster

Access via:
```python
state = central_coordinator.get_cluster_state()
print(f"Cache hit rate: {state['cache_hit_rate']:.2%}")
```

## Comparison: Centralized vs Decentralized

| Aspect | Centralized | Decentralized (Gossip) |
|--------|-------------|------------------------|
| **State View** | Global, real-time | Local, eventually consistent |
| **Placement** | Optimal given current state | Heuristic based on stale info |
| **Fault Tolerance** | Single point of failure | Highly fault tolerant |
| **Scalability** | Limited by coordinator | High |
| **Latency** | Low (direct communication) | Higher (multi-hop possible) |
| **Overhead** | N workers → coordinator | Random gossip communication |

## Experiments

### Experiment 1: Centralized vs Decentralized Performance

**Hypothesis**: Centralized approach achieves higher cache hit rate due to global state.

**Method**:
1. Run centralized simulation for 60s with 10 workers
2. Run gossip simulation for 60s with 10 workers
3. Compare cache hit rates and latency distributions

### Experiment 2: Fault Tolerance

**Hypothesis**: Decentralized approach recovers faster from node failures.

**Method**:
1. Kill coordinator mid-simulation (centralized)
2. Kill random gossip node mid-simulation (decentralized)
3. Measure time-to-recovery and request success rate

### Experiment 3: Resource Utilization Distribution

**Hypothesis**: Centralized approach achieves more even resource distribution.

**Method**:
1. Track per-worker queue depth and GPU utilization over time
2. Calculate variance/std dev of these metrics
3. Compare distributions between approaches

## Notes

- The protobuf schema in `messages.proto` is provided for documentation and future cross-language support
- Current implementation uses pickle for simplicity and consistency with existing gossip code
- For production, consider implementing actual protobuf serialization for efficiency
- Central coordinator timeout defaults to 10 seconds for detecting dead workers
- Workers report state every 1 second by default

## Future Enhancements

1. **Live Migration Support**: Implement token-based migration as described in thesis point 5
2. **Priority Queues**: Respect request priority in scheduling decisions
3. **Model Affinity**: Track which workers perform best with which models
4. **Predictive Scaling**: Use historical data to anticipate load spikes
5. **Multi-Coordinator**: Add central coordinator replication for fault tolerance
