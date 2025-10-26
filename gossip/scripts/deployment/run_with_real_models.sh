#!/bin/bash
# Script to run the central coordinator with REAL PyTorch models from GCS

set -e

echo "========================================================================"
echo "  Central Coordinator with REAL PyTorch Models"
echo "========================================================================"
echo ""

echo "✓ GCS credentials found"

# Ensure network exists
docker network create sllm-network 2>/dev/null || true
echo "✓ Docker network ready"

# Clean up any existing containers
echo ""
echo "Cleaning up existing containers..."
docker stop central-coordinator worker-0 worker-1 worker-2 load-generator 2>/dev/null || true
docker rm central-coordinator worker-0 worker-1 worker-2 load-generator 2>/dev/null || true

# Start coordinator
echo ""
echo "Starting central coordinator..."
docker run -d \
  --name central-coordinator \
  --network sllm-network \
  -p 9000:9000/udp \
  sllm/central-coordinator:latest

sleep 3
echo "✓ Coordinator started"

# Start workers with real models
echo ""
echo "Starting workers with REAL models..."
echo "Note: First run will download models from GCS (~2-5 minutes)"
echo ""

for i in 0 1 2; do
  echo "Starting worker-$i..."
  docker run -d \
    --name worker-$i \
    --network sllm-network \
    -e WORKER_ID=worker-$i \
    -e COORDINATOR_HOST=central-coordinator \
    -e USE_REAL_MODELS=true \
    -e GCS_BUCKET=remote_model \
    -e MODEL_CACHE_DIR=/tmp/model_cache \
    -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcs-key.json \
    -v ~/.gcp/sllm-key.json:/secrets/gcs-key.json:ro \
    sllm/worker:latest

  sleep 2
done

echo ""
echo "✓ All workers started"
echo ""
echo "Workers are now downloading and loading models from GCS..."
echo "This will take 1-3 minutes for first run (then cached)"
echo ""
echo "Monitoring worker-0 initialization:"
echo ""

# Monitor worker-0 initialization
timeout 180 bash -c '
while ! docker logs worker-0 2>&1 | grep -q "initialization complete"; do
  sleep 5
  echo "  Still loading models..."
done
' || echo "  (Timeout - check logs manually)"

echo ""
echo "========================================================================"
echo "  Workers Initialized!"
echo "========================================================================"
echo ""

# Show initialization summary
echo "Worker initialization summary:"
docker logs worker-0 2>&1 | grep -E "(initialized|memory utilization)" | tail -3
docker logs worker-1 2>&1 | grep -E "(initialized|memory utilization)" | tail -3
docker logs worker-2 2>&1 | grep -E "(initialized|memory utilization)" | tail -3

echo ""
echo "========================================================================"
echo "  Starting Load Generation"
echo "========================================================================"
echo ""

# Start load generation
docker run -d \
  --name load-generator \
  --network sllm-network \
  gossip-loadgen:latest \
  python3 -u load_generator.py \
  --coordinator-host central-coordinator \
  --rps 10.0 \
  --cv 8.0 \
  --duration 60

echo "Load generation running for 60 seconds with:"
echo "  - RPS: 10.0"
echo "  - CV: 8.0 (bursty pattern)"
echo "  - Real models: opt-1.3b and opt-2.7b"
echo ""
echo "Tailing load generator logs..."
echo ""

docker logs -f load-generator

echo ""
echo "========================================================================"
echo "  Load Generation Complete!"
echo "========================================================================"
echo ""

# Show final statistics
echo "Final cluster state:"
docker logs central-coordinator --tail 20

echo ""
echo "To see detailed logs:"
echo "  docker logs worker-0"
echo "  docker logs worker-1"
echo "  docker logs worker-2"
echo "  docker logs central-coordinator"
echo ""
echo "To check memory usage:"
echo "  docker stats worker-0 worker-1 worker-2 --no-stream"
echo ""
echo "To clean up:"
echo "  docker stop central-coordinator worker-0 worker-1 worker-2 load-generator"
echo "  docker rm central-coordinator worker-0 worker-1 worker-2 load-generator"
echo ""
