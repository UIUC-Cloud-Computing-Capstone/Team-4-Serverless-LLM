# Quick Start: Baseline Benchmark

## TL;DR

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Terminal 1: Start coordinator + workers
python centralized_simulation.py --workers 3 --duration 300

# 3. Terminal 2: Run baseline benchmark
python run_baseline_benchmark.py --duration 60 --rps 10
```

Results saved to `results/baseline/`

---

## What This Does

✅ Loads **8,000 real prompts** (4K GSM8K + 4K ShareGPT)
✅ Generates **bursty workload** (Gamma CV=8, like Azure Serverless)
✅ Models **popularity** (Zipf: top models get 80% of requests)
✅ Tests **your scheduler** under realistic load
✅ Measures **cache hits, latency, wait times**

---

## Example Commands

### Low Load Test (Development)
```bash
python run_baseline_benchmark.py \
    --duration 30 \
    --rps 2 \
    --gsm8k-samples 100 \
    --sharegpt-samples 100 \
    --output-dir results/dev_test
```

### Baseline (Matches Paper)
```bash
python run_baseline_benchmark.py \
    --duration 300 \
    --rps 10 \
    --cv 8.0 \
    --gsm8k-samples 4000 \
    --sharegpt-samples 4000 \
    --output-dir results/baseline
```

### High Load Stress Test
```bash
python run_baseline_benchmark.py \
    --duration 600 \
    --rps 50 \
    --cv 8.0 \
    --output-dir results/stress_test
```

### Different Burstiness Levels
```bash
# Smooth (Poisson)
python run_baseline_benchmark.py --cv 1.0 --output-dir results/cv1

# Bursty (baseline)
python run_baseline_benchmark.py --cv 8.0 --output-dir results/cv8

# Very bursty
python run_baseline_benchmark.py --cv 15.0 --output-dir results/cv15
```

---

## Output Files

```
results/baseline/
├── results.csv       # Detailed per-request data
└── metrics.json      # Aggregated statistics
```

**metrics.json** contains:
- Cache hit rate (% requests served without cold start)
- Scheduling latency (mean, median, P95, P99)
- Estimated wait times
- Per-model breakdown

---

## Understanding Results

### Cache Hit Rate
- **High (>80%)**: Good model placement, popular models pre-loaded
- **Medium (50-80%)**: Reasonable, some cold starts
- **Low (<50%)**: Poor placement or very diverse requests

### Scheduling Latency
- **<5ms**: Excellent (network + scheduling overhead only)
- **5-20ms**: Good (typical for centralized scheduler)
- **>20ms**: Investigate (coordinator overloaded or network issues)

### Per-Model Stats
- Top models should have **higher cache hit rates**
- Rare models will have **more cold starts**
- Latency should be **consistent** across models

---

## Common Configurations

### Vary Request Rate
```bash
for rps in 5 10 20 50; do
    python run_baseline_benchmark.py \
        --rps $rps \
        --output-dir results/rps_${rps}
done
```

### Vary Burstiness
```bash
for cv in 1 4 8 12; do
    python run_baseline_benchmark.py \
        --cv $cv \
        --output-dir results/cv_${cv}
done
```

### Vary Number of Models
```bash
# Few popular models
python run_baseline_benchmark.py \
    --models llama-7b llama-13b gpt-neo \
    --output-dir results/3_models

# Many diverse models
python run_baseline_benchmark.py \
    --models llama-7b llama-13b llama-30b gpt-neo falcon-7b mistral-7b \
    --output-dir results/6_models
```

---

## Docker Usage

### Using Docker Compose
```bash
# Start coordinator + 3 workers
docker-compose up

# In another terminal, run benchmark
python run_baseline_benchmark.py \
    --coordinator-host central-coordinator \
    --coordinator-port 9000
```

### Using Kubernetes
```bash
# Deploy cluster
kubectl apply -f k8s/

# Port forward coordinator
kubectl port-forward svc/central-coordinator 9000:9000

# Run benchmark
python run_baseline_benchmark.py \
    --coordinator-host localhost \
    --coordinator-port 9000
```

---

## Troubleshooting

**"No module named 'datasets'"**
```bash
pip install datasets tiktoken numpy
```

**"Connection refused"**
```bash
# Make sure coordinator is running
python centralized_simulation.py --workers 3 --duration 300
```

**"Out of memory"**
```bash
# Use smaller sample sizes
python run_baseline_benchmark.py \
    --gsm8k-samples 100 \
    --sharegpt-samples 100
```

---

## Next Steps

1. ✅ Run low-load test: `python run_baseline_benchmark.py --duration 30 --rps 2`
2. ✅ Check results: `cat results/baseline/metrics.json`
3. ✅ Run baseline: `python run_baseline_benchmark.py --duration 300 --rps 10`
4. ✅ Compare schedulers: Modify coordinator logic, re-run benchmark
5. ✅ Analyze: Plot cache hit rate vs. RPS, latency distributions, etc.

See `BASELINE_SETUP.md` for detailed documentation.
See `workload/README.md` for API reference.
