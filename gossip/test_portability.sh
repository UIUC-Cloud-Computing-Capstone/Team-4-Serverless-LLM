#!/bin/bash
# Test script for portability improvements

echo "======================================"
echo "Testing Portability Improvements"
echo "======================================"
echo ""

# Add Docker to PATH if needed
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

# Test 1: Configuration module
echo "Test 1: Configuration module..."
python3 -c "from config import config; config.print_config()" && echo " PASS" || echo "! FAIL"
echo ""

# Test 2: Worker entrypoint import
echo "Test 2: Worker entrypoint import..."
python3 -c "import entrypoint_worker" && echo " PASS" || echo "! FAIL"
echo ""

# Test 3: Coordinator entrypoint import
echo "Test 3: Coordinator entrypoint import..."
python3 -c "import entrypoint_coordinator" && echo " PASS" || echo "! FAIL"
echo ""

# Test 4: Docker availability
echo "Test 4: Docker CLI..."
if docker --version > /dev/null 2>&1; then
    docker --version
    echo " PASS"
else
    echo "! FAIL - Docker not in PATH"
    echo "Run: sudo ln -sf /Applications/Docker.app/Contents/Resources/bin/docker /usr/local/bin/docker"
fi
echo ""

# Test 5: Docker daemon
echo "Test 5: Docker daemon..."
if docker ps > /dev/null 2>&1; then
    echo "Docker daemon running"
    echo " PASS"
else
    echo "! FAIL - Docker daemon not running or credentials issue"
fi
echo ""

# Test 6: Build coordinator image (if Docker works)
if docker ps > /dev/null 2>&1; then
    echo "Test 6: Building coordinator image..."
    if docker build -f Dockerfile.coordinator -t sllm/central-coordinator:latest . > /tmp/docker_build.log 2>&1; then
        echo " PASS"
    else
        echo "! FAIL"
        tail -10 /tmp/docker_build.log
    fi
    echo ""
fi

echo "======================================"
echo "Test Summary Complete"
echo "======================================"
