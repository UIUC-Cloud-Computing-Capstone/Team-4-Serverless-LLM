# Workload Generation Infrastructure

This package provides realistic workload generation for benchmarking serverless LLM scheduling systems, following methodologies from recent research papers (AlpaServe, INFless, etc.).

## Features

### 1. **Real-world Datasets**
- **GSM8K**: Grade school math problems (reasoning tasks)
- **ShareGPT**: Multi-language chat conversations (chat tasks)
- Automatic truncation to max context length (default: 2048 tokens)
- Sampling and mixing capabilities

### 2. **Realistic Arrival Patterns**
- **Gamma distribution** for bursty serverless workloads
- Configurable **CV (Coefficient of Variation)** for burstiness control
  - CV=1: Poisson process (exponential inter-arrivals)
  - CV=8: Highly bursty (typical serverless pattern)
- Adjustable **RPS (Requests Per Second)**

### 3. **Model Popularity Distribution**
- **Zipf distribution** for realistic model popularity
- Top models get majority of requests (80/20 rule)
- Configurable α parameter for skewness

### 4. **Storage-Constrained Placement**
- Popularity-based model replication
- Round-robin or random placement strategies
- Per-worker storage capacity constraints
- Cluster-wide utilization tracking

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**New dependencies added:**
- `datasets>=2.14.0` - HuggingFace datasets library
- `tiktoken>=0.5.0` - Token counting and truncation
- `numpy>=1.24.0` - Distributions and statistics

## Quick Start

### Example 1: Load Mixed Dataset

```python
from workload.datasets import load_mixed_dataset

# Load 4K samples from each dataset (default baseline)
dataset = load_mixed_dataset(
    gsm8k_samples=4000,
    sharegpt_samples=4000,
    max_tokens=2048,
    shuffle=True
)

# Print first request
req = dataset[0]
print(f"Source: {req.dataset_source}")
print(f"Tokens: {req.token_count}")
print(f"Prompt: {req.prompt[:100]}...")
```

### Example 2: Generate Bursty Trace

```python
from workload.trace_generator import generate_trace

# Generate 5-minute trace with CV=8 (highly bursty)
trace = generate_trace(
    duration=300,           # 5 minutes
    mean_rps=10,           # 10 req/s average
    cv=8.0,                # Highly bursty
    models=["llama-7b", "llama-13b", "gpt-neo", "falcon-7b"],
    popularity_alpha=1.2,  # Zipf parameter
    seed=42
)

print(f"Generated {trace.total_requests} requests")
print(f"Actual RPS: {trace.actual_rps:.2f}")
print(f"Top model: {max(trace.model_popularity, key=trace.model_popularity.get)}")
```

### Example 3: Plan Model Placement

```python
from workload.model_placement import place_models

# Define models and sizes
models = ["llama-7b", "llama-13b", "gpt-neo"]
model_sizes = {
    "llama-7b": 13,    # GB
    "llama-13b": 26,
    "gpt-neo": 5
}

# Get popularity from trace
popularity = trace.model_popularity

# Place across 3 workers with 100GB each
plan = place_models(
    models=models,
    workers=["worker-0", "worker-1", "worker-2"],
    popularity=popularity,
    storage_per_worker_gb=100,
    model_sizes=model_sizes,
    placement_strategy="round_robin"
)

plan.print_summary()
```

### Example 4: Run Full Benchmark

```python
from workload.benchmark import BenchmarkRunner

# Create runner
runner = BenchmarkRunner(
    coordinator_host="127.0.0.1",
    coordinator_port=9000,
    request_timeout=5.0
)

# Prepare workload (datasets + trace)
requests, placement = runner.prepare_benchmark(
    duration=60,           # 1 minute
    mean_rps=5,           # 5 req/s
    cv=8.0,               # Bursty
    gsm8k_samples=1000,
    sharegpt_samples=1000,
    seed=42
)

# Run benchmark (requires coordinator running)
results, metrics = runner.run_benchmark(
    requests=requests,
    output_dir="results/benchmark_cv8_rps5",
    verbose=True
)

# Print metrics
metrics.print_summary()
```

## Command-Line Usage

### Test Dataset Loading

```bash
python -m workload.datasets
```

### Test Trace Generation

```bash
python -m workload.trace_generator
```

### Test Model Placement

```bash
python -m workload.model_placement
```

### Run Full Benchmark

```bash
# Start coordinator first (in another terminal)
python centralized_simulation.py --workers 3 --duration 120

# Run benchmark
python -m workload.benchmark \
    --duration 60 \
    --rps 5 \
    --cv 8.0 \
    --coordinator-host 127.0.0.1 \
    --coordinator-port 9000 \
    --output-dir results/test_run \
    --seed 42 \
    --verbose
```

## Baseline Configuration

To match the baseline from research papers:

```python
# Datasets: GSM8K + ShareGPT, 2048 tokens, 4K samples each
dataset = load_mixed_dataset(
    gsm8k_samples=4000,
    sharegpt_samples=4000,
    max_tokens=2048
)

# Workload: Gamma CV=8, scale to desired RPS
trace = generate_trace(
    duration=300,        # Adjust as needed
    mean_rps=10,        # Scale to cluster capacity
    cv=8.0,             # Bursty (Azure Serverless-like)
    popularity_alpha=1.2
)

# Models: Popularity-based replication, round-robin placement
plan = place_models(
    models=YOUR_MODELS,
    workers=YOUR_WORKERS,
    popularity=trace.model_popularity,
    storage_per_worker_gb=100,  # SSD capacity
    placement_strategy="round_robin"
)
```

## Output Files

Benchmark results are saved to the specified output directory:

```
results/
├── results.csv          # Detailed per-request results
└── metrics.json         # Aggregated metrics
```

**metrics.json** contains:
- Total/successful/failed request counts
- Scheduling latency (mean, median, P95, P99)
- Cache hit/miss rates
- Estimated wait times
- Per-model statistics

## Architecture Integration

The workload infrastructure integrates with your existing system:

```
┌─────────────────┐
│ Dataset Loader  │ ← Load GSM8K + ShareGPT
└────────┬────────┘
         ↓
┌─────────────────┐
│ Trace Generator │ ← Generate Gamma-distributed arrivals
└────────┬────────┘
         ↓
┌─────────────────┐
│ Model Placement │ ← Plan initial worker→model mapping
└────────┬────────┘
         ↓
┌─────────────────┐
│ Benchmark Runner│ ← Send requests to central coordinator
│                 │   Measure scheduling latency & cache hits
└─────────────────┘
         ↓
┌─────────────────┐
│Central Coordinator│ ← Your existing scheduler
│   + Workers     │
└─────────────────┘
```

## Customization

### Add Custom Dataset

```python
from workload.datasets import InferenceRequest

def load_custom_dataset():
    requests = []
    # Your dataset loading logic
    for prompt in your_prompts:
        req = InferenceRequest(
            request_id=f"custom_{i}",
            prompt=prompt,
            dataset_source="custom",
            token_count=count_tokens(prompt),
            metadata={}
        )
        requests.append(req)
    return requests
```

### Custom Arrival Distribution

```python
import numpy as np
from workload.trace_generator import TraceGenerator, TraceRequest

class CustomTraceGenerator(TraceGenerator):
    def _generate_arrivals_gamma(self, duration, mean_rps, cv):
        # Your custom inter-arrival time logic
        arrivals = []
        # ... generate timestamps
        return arrivals
```

### Custom Placement Strategy

```python
from workload.model_placement import ModelPlacer

placer = ModelPlacer(model_sizes=your_sizes)

# Implement custom logic
def custom_placement(models, workers, popularity):
    # Your placement algorithm
    return placement_plan
```

## Parameter Tuning Guide

| Parameter | Description | Typical Range | Effect |
|-----------|-------------|---------------|--------|
| `cv` | Burstiness | 1-10 | Higher = more bursty traffic |
| `mean_rps` | Request rate | 1-100+ | Scale to cluster capacity |
| `popularity_alpha` | Zipf skewness | 0.8-2.0 | Higher = more concentrated |
| `max_tokens` | Context length | 512-8192 | Match model capacity |
| `storage_per_worker_gb` | SSD capacity | 50-500 | Worker storage limit |

## Metrics Definitions

- **Scheduling Latency**: Time from request submission to receiving scheduling decision
- **Cache Hit**: Request assigned to worker with model already loaded (SERVE action)
- **Cache Miss**: Request requires cold start (COLD_START action)
- **Estimated Wait Time**: Coordinator's estimate of request processing time
- **Cluster Utilization**: Average storage utilization across workers

## References

This workload generation follows methodologies from:
- AlpaServe: Statistical Multiplexing with Model Parallelism
- INFaaS: Automated Model-less Inference Serving
- Azure Serverless Trace: Representative serverless workload patterns

## Troubleshooting

**Issue**: `datasets` library fails to download
- **Solution**: Check internet connection, or use mock data (fallback enabled)

**Issue**: Token counting errors
- **Solution**: Ensure `tiktoken` is installed: `pip install tiktoken`

**Issue**: Coordinator connection timeout
- **Solution**: Ensure coordinator is running and accessible on specified host:port

**Issue**: Out of memory when loading datasets
- **Solution**: Reduce `gsm8k_samples` and `sharegpt_samples` parameters
