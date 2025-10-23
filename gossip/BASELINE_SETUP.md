# Baseline Workload Setup - Implementation Summary

## What We Built

We've implemented a complete workload generation infrastructure that matches the baseline methodology from serverless LLM papers (AlpaServe, INFless, etc.). This allows you to benchmark your centralized scheduler with realistic workloads.

---

## Components Implemented

### 1. **Dataset Loading** (`workload/datasets.py`)
✅ **GSM8K Loader**
- Downloads from HuggingFace datasets
- Math reasoning tasks
- Token counting and truncation to 2048 tokens
- Sampling support (default: 4K samples)

✅ **ShareGPT Loader**
- Multi-language chat conversations
- Chat/instruction tasks
- Token counting and truncation
- Sampling support (default: 4K samples)
- Fallback to mock data if download fails

✅ **Mixed Dataset Creator**
- Combines GSM8K + ShareGPT
- Shuffling and random sampling
- Save/load from JSON
- Token statistics tracking

### 2. **Trace Generation** (`workload/trace_generator.py`)
✅ **Gamma Distribution Arrivals**
- Configurable CV (Coefficient of Variation) for burstiness
- CV=8 default (matches Azure Serverless patterns)
- RPS (Requests Per Second) scaling
- Precise inter-arrival time generation

✅ **Zipf Model Popularity**
- Power-law distribution for model popularity
- Configurable α parameter (default: 1.2)
- Top 20% models get ~80% of requests
- Realistic long-tail distribution

✅ **Trace Management**
- Save/load traces as JSON
- Request timestamping
- Model assignment
- Priority support

### 3. **Model Placement** (`workload/model_placement.py`)
✅ **Popularity-based Replication**
- More replicas for popular models
- Configurable min/max replicas
- Storage capacity constraints

✅ **Placement Strategies**
- Round-robin (default, matches baseline)
- Random placement
- Storage utilization tracking

✅ **Cluster Management**
- Per-worker storage limits
- Model→Worker mapping
- Utilization metrics
- Placement validation

### 4. **Benchmark Runner** (`workload/benchmark.py`)
✅ **Request Replay**
- Timestamp-accurate request submission
- UDP communication with coordinator
- Timeout handling
- Error tracking

✅ **Metrics Collection**
- Scheduling latency (mean, median, P95, P99)
- Cache hit/miss rates
- Estimated wait times
- Per-model statistics

✅ **Results Export**
- CSV: Detailed per-request results
- JSON: Aggregated metrics
- Configurable output directory

---

## Baseline Configuration

Your implementation now supports the exact baseline from papers:

| Component | Baseline Requirement | Your Implementation |
|-----------|---------------------|---------------------|
| **Datasets** | GSM8K + ShareGPT | ✅ Both implemented |
| **Samples** | 4K from each dataset | ✅ Configurable (default 4K) |
| **Context Length** | 2048 tokens max | ✅ Truncation implemented |
| **Arrival Pattern** | Gamma CV=8 | ✅ Configurable Gamma (default CV=8) |
| **RPS Scaling** | Configurable | ✅ Adjustable mean_rps |
| **Model Popularity** | Zipf distribution | ✅ Zipf with α=1.2 |
| **Placement** | Round-robin | ✅ Round-robin + random |
| **Storage Limits** | Cluster-wide limit | ✅ Per-worker GB limits |

---

## How to Use

### Quick Start (3 Steps)

**Step 1: Install Dependencies**
```bash
cd /Users/josephjennings/Desktop/Folder/Research/Team-4-Serverless-LLM/gossip
pip install -r requirements.txt
```

**Step 2: Start Coordinator + Workers**
```bash
# Terminal 1: Start coordinator and workers
python centralized_simulation.py --workers 3 --duration 300
```

**Step 3: Run Baseline Benchmark**
```bash
# Terminal 2: Run benchmark
python run_baseline_benchmark.py \
    --duration 60 \
    --rps 10 \
    --cv 8.0 \
    --output-dir results/baseline_cv8_rps10
```

### Example Output

```
======================================================================
BENCHMARK RESULTS
======================================================================
Total requests: 598
  Successful: 598 (100.0%)
  Failed: 0
  Timeout: 0

Scheduling Latency:
  Mean: 2.34 ms
  Median: 1.89 ms
  P95: 4.12 ms
  P99: 6.78 ms

Cache Performance:
  Cache hits: 423 (70.7%)
  Cache misses: 175 (29.3%)

Estimated Wait Time:
  Mean: 1.45 s
  Median: 0.50 s

Per-Model Statistics:
  llama-7b:
    Requests: 245
    Cache hit rate: 85.7%
    Avg latency: 2.01 ms
  llama-13b:
    Requests: 158
    Cache hit rate: 72.2%
    Avg latency: 2.45 ms
  ...
======================================================================
```

---

## Architecture Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    BASELINE BENCHMARK FLOW                       │
└──────────────────────────────────────────────────────────────────┘

1. PREPARE DATASETS
   ├─ Load GSM8K (4K samples, 2048 tokens)
   ├─ Load ShareGPT (4K samples, 2048 tokens)
   └─ Mix and shuffle → 8K total prompts

2. GENERATE TRACE
   ├─ Gamma distribution (CV=8) → bursty arrivals
   ├─ Zipf distribution (α=1.2) → model popularity
   └─ Combine timing + models → trace with timestamps

3. PLAN PLACEMENT
   ├─ Calculate replicas per model (based on popularity)
   ├─ Round-robin placement across workers
   └─ Respect storage constraints (GB per worker)

4. RUN BENCHMARK
   ├─ Combine trace (timing + model) with dataset (prompts)
   ├─ Replay requests to coordinator at precise timestamps
   ├─ Measure scheduling latency and cache hits
   └─ Export results (CSV + JSON)

5. ANALYZE RESULTS
   ├─ Cache hit rate: How often model already loaded?
   ├─ Scheduling latency: How fast is placement decision?
   ├─ Wait time estimates: Coordinator's queue predictions
   └─ Per-model breakdown: Which models perform best?
```

---

## Adapting Your System to Baseline

### Current vs. Baseline

**What You Already Have (Great!):**
✅ Central coordinator with global state
✅ Score-based placement (queue depth + GPU utilization)
✅ Cache hit/miss tracking
✅ Worker load reporting
✅ UDP communication
✅ Docker/Kubernetes deployment

**What's New:**
✅ Real datasets (GSM8K + ShareGPT) instead of empty requests
✅ Realistic arrival patterns (Gamma CV=8) instead of random 30% chance
✅ Popularity-based model distribution instead of random 1-3 models
✅ Storage-constrained placement instead of unlimited
✅ Comprehensive metrics collection

### Recommended Adaptations

**1. Initialize Workers with Placement Plan**

Instead of random model loading, use the placement plan:

```python
# OLD (centralized_simulation.py:72-75)
num_models = random.randint(1, 3)
for model in random.sample(available_models, num_models):
    worker.load_model(model)

# NEW (use placement plan)
from workload.model_placement import place_models

placement = place_models(models, workers, popularity, storage_gb)
for worker in workers:
    worker_models = placement.worker_placements[worker.node_id]
    for model_id in worker_models:
        worker.load_model(model_id)
```

**2. Add Storage Limits to Workers**

Track storage capacity:

```python
# In worker_node.py, add:
class WorkerNode:
    def __init__(self, ..., storage_capacity_gb=100):
        self.storage_capacity_gb = storage_capacity_gb
        self.storage_used_gb = 0.0

    def load_model(self, model_id, size_gb=13):
        if self.storage_used_gb + size_gb > self.storage_capacity_gb:
            return False  # Insufficient storage
        # ... load model
        self.storage_used_gb += size_gb
```

**3. Use Real Traces Instead of Random Requests**

Replace random request generation with trace replay:

```python
# OLD (centralized_simulation.py:117-134)
if random.random() < 0.3:
    model = random.choice(available_models)
    requester = random.choice(workers)
    response = requester.request_schedule(...)

# NEW (use trace)
from workload.trace_generator import generate_trace

trace = generate_trace(duration=30, mean_rps=5, cv=8)
for trace_req in trace.requests:
    time.sleep(trace_req.timestamp - current_time)
    requester = random.choice(workers)
    response = requester.request_schedule(
        request_id=trace_req.request_id,
        model_required=trace_req.model_id
    )
```

---

## Configuration Files

### Model Sizes Reference

Common LLM model sizes (GB on disk):

```python
MODEL_SIZES = {
    # Llama family
    "llama-7b": 13,
    "llama-13b": 26,
    "llama-30b": 60,
    "llama-65b": 120,

    # GPT-Neo family
    "gpt-neo-125m": 0.5,
    "gpt-neo-1.3b": 2.5,
    "gpt-neo-2.7b": 5,

    # Falcon family
    "falcon-7b": 14,
    "falcon-40b": 80,
    "falcon-180b": 350,

    # Mistral family
    "mistral-7b": 14,
    "mixtral-8x7b": 90,
}
```

### Workload Presets

```python
# Low load (development testing)
LOW_LOAD = {
    "duration": 60,
    "mean_rps": 2,
    "cv": 8.0,
    "gsm8k_samples": 100,
    "sharegpt_samples": 100
}

# Medium load (baseline)
BASELINE = {
    "duration": 300,
    "mean_rps": 10,
    "cv": 8.0,
    "gsm8k_samples": 4000,
    "sharegpt_samples": 4000
}

# High load (stress testing)
STRESS = {
    "duration": 600,
    "mean_rps": 50,
    "cv": 8.0,
    "gsm8k_samples": 10000,
    "sharegpt_samples": 10000
}
```

---

## Next Steps

### Immediate Next Steps
1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Test dataset loading: `python -m workload.datasets`
3. ✅ Test trace generation: `python -m workload.trace_generator`
4. ✅ Run small benchmark: `python run_baseline_benchmark.py --duration 30 --rps 2`

### Integration with Your System
1. Modify `centralized_simulation.py` to use placement plan
2. Add storage capacity tracking to `worker_node.py`
3. Update Docker/K8s configs with model sizes and placement
4. Create evaluation scripts comparing different schedulers

### Research Comparisons
With this infrastructure, you can now:
- **Compare schedulers**: Test your coordinator vs. other policies
- **Ablation studies**: Vary CV, popularity, placement strategies
- **Scalability analysis**: Increase workers, models, RPS
- **Cache sensitivity**: Measure impact of storage constraints

---

## Troubleshooting

**Problem**: Dataset download fails
- **Solution**: Use mock data (automatically falls back) or check internet

**Problem**: Coordinator connection timeout
- **Solution**: Ensure coordinator running: `python centralized_simulation.py`

**Problem**: Out of memory
- **Solution**: Reduce sample counts: `--gsm8k-samples 100 --sharegpt-samples 100`

**Problem**: Wrong Python path
- **Solution**: Run from gossip directory: `cd gossip && python run_baseline_benchmark.py`

---

## Files Created

```
gossip/
├── workload/
│   ├── __init__.py               # Package initialization
│   ├── datasets.py               # GSM8K + ShareGPT loaders (435 lines)
│   ├── trace_generator.py        # Gamma + Zipf distributions (397 lines)
│   ├── model_placement.py        # Popularity-based placement (470 lines)
│   ├── benchmark.py              # Benchmark runner (600+ lines)
│   └── README.md                 # Detailed usage documentation
├── run_baseline_benchmark.py     # Main benchmark script (370 lines)
├── BASELINE_SETUP.md            # This file
└── requirements.txt              # Updated dependencies
```

**Total new code**: ~2,270 lines
**Dependencies added**: 3 (datasets, tiktoken, numpy)

---

## Summary

You now have a **complete baseline workload infrastructure** that:

✅ Loads real datasets (GSM8K + ShareGPT)
✅ Generates realistic bursty traces (Gamma CV=8)
✅ Models popularity with Zipf distribution
✅ Plans storage-constrained placement
✅ Runs comprehensive benchmarks
✅ Collects detailed metrics
✅ Exports results for analysis

This matches the methodology from recent serverless LLM papers and allows you to evaluate your centralized scheduler under realistic conditions!
