#!/bin/bash
# Test runner for Gossip system

set -e

echo "🧪 Running Gossip System Tests"
echo "=============================="

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run unit tests
echo "📋 Running unit tests..."
python -m pytest tests/unit/ -v --tb=short

# Run integration tests
echo "📋 Running integration tests..."
python -m pytest tests/integration/ -v --tb=short

# Run specific model loading test
echo "📋 Testing model loading..."
python tests/unit/test_model_loading.py

echo ""
echo "✅ All tests completed!"
