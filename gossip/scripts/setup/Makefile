# Makefile for SLLM Gossip System

.PHONY: help build-docker run-docker stop-docker clean-docker \
        build-k8s deploy-k8s delete-k8s logs-k8s \
        test lint format

# Variables
COORDINATOR_IMAGE := sllm/central-coordinator:latest
WORKER_IMAGE := sllm/worker:latest
NAMESPACE := sllm-system

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# Docker targets
build-docker: ## Build Docker images
	docker build -f Dockerfile.coordinator -t $(COORDINATOR_IMAGE) .
	docker build -f Dockerfile.worker -t $(WORKER_IMAGE) .

run-docker: ## Run with docker-compose
	docker-compose up

run-docker-detached: ## Run with docker-compose in background
	docker-compose up -d

stop-docker: ## Stop docker-compose services
	docker-compose down

logs-docker: ## View docker-compose logs
	docker-compose logs -f

clean-docker: ## Remove Docker images and containers
	docker-compose down -v
	docker rmi $(COORDINATOR_IMAGE) $(WORKER_IMAGE) || true

# Kubernetes targets
build-k8s: build-docker ## Build images for Kubernetes

load-kind: build-docker ## Load images into kind cluster
	kind load docker-image $(COORDINATOR_IMAGE)
	kind load docker-image $(WORKER_IMAGE)

deploy-k8s: ## Deploy to Kubernetes
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/coordinator-deployment.yaml -n $(NAMESPACE)
	kubectl apply -f k8s/worker-statefulset.yaml -n $(NAMESPACE)

delete-k8s: ## Delete Kubernetes resources
	kubectl delete -f k8s/worker-statefulset.yaml -n $(NAMESPACE) --ignore-not-found
	kubectl delete -f k8s/coordinator-deployment.yaml -n $(NAMESPACE) --ignore-not-found
	kubectl delete -f k8s/namespace.yaml --ignore-not-found

logs-k8s: ## View Kubernetes logs
	kubectl logs -f -l app=central-coordinator -n $(NAMESPACE) --tail=50

logs-k8s-workers: ## View worker logs
	kubectl logs -f -l app=worker -n $(NAMESPACE) --tail=50

status-k8s: ## Check Kubernetes status
	kubectl get all -n $(NAMESPACE)

scale-workers: ## Scale workers (usage: make scale-workers REPLICAS=10)
	kubectl scale statefulset worker --replicas=$(REPLICAS) -n $(NAMESPACE)

# Testing targets
test-local: ## Run local simulation
	python3 centralized_simulation.py --workers 5 --duration 30

test-gossip: ## Run gossip simulation
	python3 main.py

# Development targets
lint: ## Run linting (requires pylint)
	@command -v pylint >/dev/null 2>&1 || { echo "pylint not installed. Install with: pip install pylint"; exit 1; }
	pylint *.py --disable=C0111,R0913,R0914

format: ## Format code (requires black)
	@command -v black >/dev/null 2>&1 || { echo "black not installed. Install with: pip install black"; exit 1; }
	black *.py

# Quick deployment workflows
quick-docker: build-docker run-docker-detached ## Quick Docker deployment

quick-k8s: build-k8s load-kind deploy-k8s status-k8s ## Quick Kubernetes deployment (kind)

# Default target
.DEFAULT_GOAL := help
