# Makefile for Gossip: Centralized LLM Load Balancing System

.PHONY: help setup build test clean run stop logs status

# Default target
help:
	@echo "Gossip: Centralized LLM Load Balancing System"
	@echo "=============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  setup     - Set up the development environment"
	@echo "  build     - Build Docker images"
	@echo "  test      - Run all tests"
	@echo "  run       - Run the simulation"
	@echo "  stop      - Stop all containers"
	@echo "  clean     - Clean up containers and images"
	@echo "  logs      - Show container logs"
	@echo "  status    - Show container status"
	@echo "  install   - Install Python dependencies"
	@echo "  lint      - Run code linting"
	@echo ""

# Setup development environment
setup:
	@echo "🚀 Setting up Gossip system..."
	@./scripts/setup/setup.sh

# Install Python dependencies
install:
	@echo "📦 Installing Python dependencies..."
	@pip install -r requirements.txt

# Build Docker images
build:
	@echo " Building Docker images..."
	@docker build -f docker/worker/Dockerfile.worker -t gossip-worker .
	@docker build -f docker/coordinator/Dockerfile.coordinator -t gossip-coordinator .
	@docker build -f docker/loadgen/Dockerfile.loadgen -t gossip-loadgen .
	@echo " Docker images built"


# Run the simulation
run:
	@echo " Running Gossip simulation..."
	@./scripts/deployment/run_with_real_models.sh

# Stop all containers
stop:
	@echo " Stopping containers..."
	@docker stop $$(docker ps -q --filter "name=worker-\|central-coordinator\|load-generator") 2>/dev/null || true
	@echo " Containers stopped"

# Clean up containers and images
clean:
	@echo " Cleaning up..."
	@docker stop $$(docker ps -q --filter "name=worker-\|central-coordinator\|load-generator") 2>/dev/null || true
	@docker rm $$(docker ps -aq --filter "name=worker-\|central-coordinator\|load-generator") 2>/dev/null || true
	@docker rmi gossip-worker gossip-coordinator gossip-loadgen 2>/dev/null || true
	@echo " Cleanup complete"

# Show container logs
logs:
	@echo " Container logs:"
	@echo "=================="
	@docker logs central-coordinator --tail 20 2>/dev/null || echo "Coordinator not running"
	@echo ""
	@docker logs worker-0 --tail 20 2>/dev/null || echo "Worker-0 not running"
	@echo ""
	@docker logs load-generator --tail 20 2>/dev/null || echo "Load generator not running"

# Show container status
status:
	@echo " Container status:"
	@echo "===================="
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" --filter "name=worker-\|central-coordinator\|load-generator"

# Quick development cycle
dev: clean build run

# Full test cycle
test-full: clean build test

