##  Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Google Cloud Storage credentials (for real models)
- 8GB+ RAM recommended

### 1. Setup Environment

```bash
# Clone and navigate to project
cd gossip

# Install Python dependencies
pip install -r requirements.txt

# Set up GCS credentials (for real models)
export GOOGLE_APPLICATION_CREDENTIALS="path/to/gcs-key.json"
```

### 2. Run Simulation

```bash
# Run with real PyTorch models from GCS
./scripts/deployment/run_with_real_models.sh

# Or run with Docker Compose
docker-compose -f config/docker/docker-compose.yml up
```

### 3. Monitor System

```bash
# Check container status
docker ps

# View logs
docker logs worker-0
docker logs central-coordinator
docker logs load-generator

# Monitor resource usage
docker stats worker-0 worker-1 worker-2
```

### Common Issues

1. **Workers Not Ready**: Check GCS credentials and network connectivity
2. **Download Timeouts**: Increase timeout values in `model_loader.py`
3. **Memory Issues**: Reduce model size or increase container memory limits
4. **Network Issues**: Verify Docker network configuration

### Debug Commands

```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Inspect container logs
docker logs <container-name> --tail 50

# Check network connectivity
docker exec worker-0 ping central-coordinator

# Test GCS connectivity
docker exec worker-0 python -c "from google.cloud import storage; print(storage.Client().bucket('remote_model').exists())"
```
