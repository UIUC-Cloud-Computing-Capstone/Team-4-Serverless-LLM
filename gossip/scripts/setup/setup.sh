#!/bin/bash
# Setup script for Gossip: Centralized LLM Load Balancing System

set -e

echo "🚀 Setting up Gossip: Centralized LLM Load Balancing System"
echo "=========================================================="

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION found"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed"
    exit 1
fi
echo "✅ Docker found"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required but not installed"
    exit 1
fi
echo "✅ Docker Compose found"

# Note: May need to have run virtual environment setup prior to this
pip install -r requirements.txt
echo "✅ Python dependencies installed"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data/cache
mkdir -p data/models
echo "✅ Directories created"

# Build Docker images
echo "🐳 Building Docker images..."
docker build -f docker/worker/Dockerfile.worker -t gossip-worker .
docker build -f docker/coordinator/Dockerfile.coordinator -t gossip-coordinator .
docker build -f docker/loadgen/Dockerfile.loadgen -t gossip-loadgen .
echo "✅ Docker images built"

# Create Docker network
echo "🌐 Creating Docker network..."
docker network create gossip-network 2>/dev/null || echo "✅ Docker network already exists"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run the simulation: ./scripts/deployment/run_with_real_models.sh"
echo "2. Or use Docker Compose: docker-compose -f config/docker/docker-compose.yml up"
echo "3. Check logs: docker logs worker-0"
echo ""
echo "For more information, see README.md"

