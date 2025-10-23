#!/usr/bin/env python3
"""
Quick test to verify GCS model loading works.
"""

import os
import sys
from model_loader import ModelLoader

# Set GCS credentials
credentials_path = os.path.expanduser("~/.gcp/sllm-key.json")
if os.path.exists(credentials_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    print(f"✓ Using credentials: {credentials_path}")
else:
    print(f"✗ Credentials not found at: {credentials_path}")
    print("  Run the setup commands to create service account key")
    sys.exit(1)

# Initialize model loader
print("\n=== Testing Model Loader ===\n")
loader = ModelLoader(
    gcs_bucket="remote_model",
    cache_dir="/tmp/model_cache_test",
    device="cpu"
)

# Test loading opt-1.3b
print("Testing opt-1.3b download and load...")
try:
    success = loader.load_model("opt-1.3b")
    if success:
        print("✓ opt-1.3b loaded successfully!")
        print(f"  Loaded models: {loader.get_loaded_models()}")
        print(f"  Memory usage: {loader.get_model_memory_usage('opt-1.3b') / (1024**3):.2f} GB")
    else:
        print("✗ Failed to load opt-1.3b")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error loading opt-1.3b: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test unloading
print("\nTesting model unload...")
loader.unload_model("opt-1.3b")
print(f"✓ Model unloaded. Loaded models: {loader.get_loaded_models()}")

print("\n=== All Tests Passed! ===")
print("\nYour GCS setup is working correctly.")
print("You can now run Docker containers with USE_REAL_MODELS=true")
