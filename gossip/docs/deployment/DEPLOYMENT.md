# Deployment Guide

This guide covers deploying the Serverless LLM system in various environments: local, Docker, and Kubernetes.

## Table of Contents
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites
- Python 3.11+
- No external dependencies (uses standard library only)

### Quick Start
```bash
# Run simulation
python3 centralized_simulation.py --workers 5 --duration 30

# Or manually start components
# Terminal 1: Central coordinator
python3 udp_central_coordinator.py --host 127.0.0.1 --port 9000

# Terminal 2-4: Workers
python3 worker_node.py --node-id worker-1 --port 8001 --coordinator-port 9000
python3 worker_node.py --node-id worker-2 --port 8002 --coordinator-port 9000
python3 worker_node.py --node-id worker-3 --port 8003 --coordinator-port 9000
```

---

## Docker Deployment

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+

### Building Images

```bash
# Build coordinator image
docker build -f docker/coordinator/Dockerfile.coordinator -t sllm/central-coordinator:latest .

# Build worker image
docker build -f docker/worker/Dockerfile.worker -t sllm/worker:latest .
```

### Running with Docker Compose

```bash
# Start all services
docker-compose up

# Start in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Scale workers
docker-compose up --scale worker-1=5
```

### Running Individual Containers

```bash
# Create network
docker network create sllm-network

# Start coordinator
docker run -d \
  --name central-coordinator \
  --network sllm-network \
  -p 9000:9000/udp \
  -e DEPLOYMENT_MODE=docker \
  -e LOG_LEVEL=INFO \
  sllm/central-coordinator:latest

# Start workers
docker run -d \
  --name worker-1 \
  --network sllm-network \
  -e WORKER_ID=worker-1 \
  -e COORDINATOR_HOST=central-coordinator \
  -e DEPLOYMENT_MODE=docker \
  sllm/worker:latest

docker run -d \
  --name worker-2 \
  --network sllm-network \
  -e WORKER_ID=worker-2 \
  -e COORDINATOR_HOST=central-coordinator \
  -e DEPLOYMENT_MODE=docker \
  sllm/worker:latest
```

### Docker Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEPLOYMENT_MODE` | `local` | Deployment mode: `local`, `docker`, `kubernetes` |
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `BIND_HOST` | `0.0.0.0` | Host to bind to (use 0.0.0.0 in containers) |
| `COORDINATOR_HOST` | `central-coordinator` | Coordinator hostname (DNS name in Docker) |
| `COORDINATOR_PORT` | `9000` | Coordinator port |
| `WORKER_PORT` | `8000` | Worker port |
| `REPORT_INTERVAL` | `1.0` | Worker report interval (seconds) |
| `WORKER_TIMEOUT` | `10.0` | Timeout for considering workers dead (seconds) |
| `WORKER_ID` | Auto-generated | Worker identifier |

---

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster 1.24+
- kubectl configured
- Container images pushed to registry or loaded locally (for local clusters like kind/minikube)

### Building and Pushing Images

```bash
# For Docker Hub
docker build -f Dockerfile.coordinator -t yourusername/sllm-coordinator:latest .
docker build -f Dockerfile.worker -t yourusername/sllm-worker:latest .

docker push yourusername/sllm-coordinator:latest
docker push yourusername/sllm-worker:latest

# Update k8s/*.yaml files with your image names
```

### Deploying to Kubernetes

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Deploy central coordinator
kubectl apply -f k8s/coordinator-deployment.yaml -n sllm-system

# Deploy workers
kubectl apply -f k8s/worker-statefulset.yaml -n sllm-system

# Verify deployment
kubectl get pods -n sllm-system
kubectl get svc -n sllm-system

# View logs
kubectl logs -f deployment/central-coordinator -n sllm-system
kubectl logs -f statefulset/worker -n sllm-system
```

### Scaling Workers

```bash
# Scale to 10 workers
kubectl scale statefulset worker --replicas=10 -n sllm-system

# Autoscaling (example)
kubectl autoscale statefulset worker \
  --min=3 --max=20 \
  --cpu-percent=80 \
  -n sllm-system
```

### StatefulSet Benefits

Workers are deployed as a **StatefulSet** which provides:
- **Stable network identities**: `worker-0`, `worker-1`, etc.
- **Ordered deployment**: Workers start sequentially
- **Persistent hostnames**: Useful for debugging and monitoring
- **Graceful shutdown**: Proper cleanup on termination

### Testing in Local Kubernetes

For local testing with kind or minikube:

```bash
# Load images into kind cluster
kind load docker-image sllm/central-coordinator:latest
kind load docker-image sllm/worker:latest

# Then deploy normally
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/coordinator-deployment.yaml -n sllm-system
kubectl apply -f k8s/worker-statefulset.yaml -n sllm-system
```

---

## Configuration

### Configuration File

All configuration is centralized in `config.py` and can be overridden with environment variables:

```python
from config import config

# Access configuration
coordinator_host, coordinator_port = config.get_coordinator_address()
worker_id = config.get_worker_id()

# Check if running in container
if config.is_containerized():
    # Container-specific logic
    pass
```

### Environment Variable Precedence

1. **Kubernetes ConfigMap/Secrets** (highest priority)
2. **Docker environment variables** (docker-compose.yml or docker run -e)
3. **Default values in config.py** (lowest priority)

### Network Considerations

**Docker:**
- Uses Docker DNS for service discovery
- `COORDINATOR_HOST=central-coordinator` resolves automatically
- Bridge network with default subnet `172.28.0.0/16`

**Kubernetes:**
- Uses Kubernetes DNS for service discovery
- Service name: `central-coordinator.sllm-system.svc.cluster.local`
- Short name works within namespace: `central-coordinator`
- Headless service for workers enables direct pod-to-pod communication

**Multi-Node Kubernetes:**
- UDP traffic works across nodes
- Ensure firewall rules allow UDP traffic between worker nodes
- Consider pod anti-affinity to spread workers across nodes

---

## Monitoring

### Health Checks

Both coordinator and workers have health check probes:

```bash
# Docker: Check container health
docker ps

# Kubernetes: Check pod health
kubectl get pods -n sllm-system
kubectl describe pod <pod-name> -n sllm-system
```

### Logs

```bash
# Docker
docker-compose logs -f central-coordinator
docker-compose logs -f worker-1

# Kubernetes
kubectl logs -f deployment/central-coordinator -n sllm-system
kubectl logs -f worker-0 -n sllm-system

# Follow all worker logs
kubectl logs -f -l app=worker -n sllm-system
```

### Metrics

The central coordinator tracks:
- Active worker count
- Total requests processed
- Cache hit rate
- Queue depth distribution
- GPU utilization

Access programmatically:
```python
state = central_coordinator.get_cluster_state()
print(state)
```

---

## Troubleshooting

### Workers Can't Connect to Coordinator

**Docker:**
```bash
# Check network
docker network inspect sllm-network

# Check coordinator is running
docker ps | grep central-coordinator

# Test connectivity from worker container
docker exec -it worker-1 ping central-coordinator
```

**Kubernetes:**
```bash
# Check service
kubectl get svc central-coordinator -n sllm-system

# Check DNS resolution from worker pod
kubectl exec -it worker-0 -n sllm-system -- nslookup central-coordinator

# Check coordinator logs
kubectl logs deployment/central-coordinator -n sllm-system
```

### UDP Port Issues

**Symptom:** Workers report timeout errors

**Solutions:**
1. Verify UDP port is exposed: `-p 9000:9000/udp`
2. Check firewall rules allow UDP
3. Verify coordinator is binding to `0.0.0.0` not `127.0.0.1`
4. In Kubernetes, ensure service type is correct (ClusterIP/LoadBalancer)

### Container Keeps Restarting

```bash
# Docker: Check exit code and logs
docker ps -a
docker logs <container-id>

# Kubernetes: Check events and logs
kubectl describe pod <pod-name> -n sllm-system
kubectl logs <pod-name> -n sllm-system --previous
```

Common causes:
- Missing environment variables
- Port already in use
- Insufficient resources (memory/CPU)
- Image pull errors

### Permission Denied Errors

Both Dockerfiles use non-root users for security. If you see permission errors:

```dockerfile
# In Dockerfile, ensure proper ownership
RUN chown -R worker:worker /app
USER worker
```

### High Memory Usage

Workers are limited to 1Gi by default. Adjust in k8s manifests:

```yaml
resources:
  limits:
    memory: "2Gi"  # Increase as needed
    cpu: "2000m"
```

---

## Best Practices

### Security

1. **Use non-root users** ✅ Already implemented
2. **Scan images** for vulnerabilities:
   ```bash
   docker scan sllm/worker:latest
   ```
3. **Use secrets** for sensitive data:
   ```bash
   kubectl create secret generic sllm-secrets \
     --from-literal=api-key=xxx \
     -n sllm-system
   ```
4. **Network policies** to restrict traffic:
   ```yaml
   # Example: Only allow worker -> coordinator traffic
   ```

### Resource Management

1. **Set resource requests/limits** in Kubernetes
2. **Use horizontal pod autoscaling** for workers
3. **Monitor resource usage**:
   ```bash
   kubectl top pods -n sllm-system
   ```

### High Availability

1. **For Coordinator**: Currently single instance (SPOF by design for experiments)
   - For production, consider active-passive replication
   - Use persistent volume for state

2. **For Workers**: Already stateless and horizontally scalable
   - Set pod disruption budgets:
   ```yaml
   apiVersion: policy/v1
   kind: PodDisruptionBudget
   metadata:
     name: worker-pdb
   spec:
     minAvailable: 2
     selector:
       matchLabels:
         app: worker
   ```

### Production Readiness Checklist

- [ ] Images scanned for vulnerabilities
- [ ] Resource limits configured
- [ ] Health checks passing
- [ ] Monitoring and alerting setup
- [ ] Backup/restore procedures tested
- [ ] Load testing completed
- [ ] Security policies applied
- [ ] Documentation updated
- [ ] Runbooks created for common issues

---

## Next Steps

1. **Test locally** with docker-compose
2. **Deploy to staging** Kubernetes cluster
3. **Run experiments** comparing centralized vs decentralized
4. **Scale workers** to test performance at scale
5. **Add metrics exporter** (Prometheus/Grafana)
6. **Implement actual model loading** (replace simulation)
7. **Add authentication** between components
8. **Setup CI/CD** pipeline for automated deployments
