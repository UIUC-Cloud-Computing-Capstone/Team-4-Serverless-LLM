# Serverless LLM Cluster - Quick Setup Guide

## Prerequisites
- Docker and Docker Compose installed
- At least 8GB RAM available
- 5GB disk space for model cache

## Quick Start

### 1. Build and Start the System
```bash
cd config/docker
docker-compose build
docker-compose up -d
```

### 2. Monitor System Startup
```bash
# Check container status
docker-compose ps

# Watch workers load models (~20-30 seconds)
docker logs worker-1 --tail 50 -f
```

**Expected output:**
```
Model loaded in 17.42s
Model parameters: 125,239,296 (125.2M)
Model memory usage: 477.8 MB
========== MODEL LOAD COMPLETE: facebook/opt-125m ==========
Worker ready: True, models loaded: ['facebook/opt-125m']
```

### 3. Monitor Load Generation
```bash
# Watch load generator send inference requests
docker logs load-generator --tail 30 -f
```

**Expected output:**
```
Target RPS: 5.0, CV: 8.0
Sending 1357 requests over 300.0s...
[10.5s] REQUEST req-10 | Model: facebook/opt-125m
  Schedule: worker-2 (serve)
  Prompt: Problem 42: If John has 42 apples...
  Output: Problem 42: If John has 42 apples and gives 21 to Mary...
  Tokens: 58 | Latency: 65000ms
```

### 4. View Inference Results
```bash
# See actual model outputs from workers
docker logs worker-2 | grep -A 3 "Output:"
```

**Expected output:**
```
Output: What is the capital of France?
France is the capital of the French Republic.
I'm not sure what you mean.
```

## System Components

| Service | Port | Description |
|---------|------|-------------|
| `central-coordinator` | 9000/udp | Routes requests to workers |
| `worker-1` | 8000 | Inference worker with OPT-125M |
| `worker-2` | 8000 | Inference worker with OPT-125M |
| `worker-3` | 8000 | Inference worker with OPT-125M |
| `load-generator` | - | Sends inference requests |

## Configuration

### Adjust Load Generation
Edit `config/docker/docker-compose.yml` under `load-generator`:
```yaml
environment:
  - TARGET_RPS=5.0      # Requests per second (default: 5.0)
  - CV=8.0              # Burstiness coefficient (default: 8.0)
  - DURATION=300.0      # Test duration in seconds (default: 300)
```

### Change Model
Edit `docker/worker/entrypoint_worker.py` line 73:
```python
available_models = ["facebook/opt-125m"]  # Change to other HuggingFace models
```

## Performance Expectations

| Metric | Value |
|--------|-------|
| Model Load Time | ~20 seconds |
| Inference Latency (CPU) | ~60-70 seconds per request |
| Success Rate | ~85-95% |
| Memory per Worker | ~1GB |
| Throughput (CPU) | ~0.05-0.1 RPS per worker |

**Note:** CPU inference is slow. For production, use GPU-enabled workers.

## Useful Commands

```bash
# Stop the system
docker-compose down

# View all logs
docker-compose logs -f

# Check resource usage
docker stats

# Restart a specific worker
docker-compose restart worker-1

# Clean up volumes (removes cached models)
docker-compose down -v
```

## Troubleshooting

**Workers not loading models:**
```bash
# Check worker logs for errors
docker logs worker-1 --tail 100
```

**Load generator timing out:**
- Normal on CPU! Inference takes ~60s per request
- Reduce `TARGET_RPS` to 1.0 for CPU testing

**No inference responses:**
```bash
# Verify workers are ready
docker logs worker-1 | grep "Worker ready"

# Check if model is loaded
docker logs worker-1 | grep "MODEL LOAD COMPLETE"
```

## Test Manual Inference

```bash
docker exec worker-1 python3 -c "
from model_loader import ModelLoader
loader = ModelLoader(cache_dir='/tmp/model_cache', device='cpu')
loader.load_model('facebook/opt-125m')
model = loader.loaded_models['facebook/opt-125m']['model']
tokenizer = loader.loaded_models['facebook/opt-125m']['tokenizer']

inputs = tokenizer('Hello, my name is', return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"
```

## Architecture Overview

```
┌─────────────┐
│ Load        │  Generates synthetic workload
│ Generator   │  (GSM8K + ShareGPT datasets)
└──────┬──────┘
       │ 1. Schedule Request
       ▼
┌─────────────┐
│ Central     │  Routes to worker with model
│ Coordinator │  (SERVE, COLD_START, MIGRATE)
└──────┬──────┘
       │ 2. Schedule Response
       ▼
┌─────────────┐
│ Worker-1/2/3│  Performs inference with
│ (OPT-125M)  │  facebook/opt-125m model
└──────┬──────┘
       │ 3. Inference Response
       ▼
    [Logged Output + Metrics]
```
