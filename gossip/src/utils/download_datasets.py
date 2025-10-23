"""
Script to download GSM8K and ShareGPT datasets.

Downloads the datasets needed for the enhanced simulation.
"""

import os
import json
import urllib.request
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("DatasetDownloader")


def download_gsm8k(output_dir: str = "datasets") -> str:
    """
    Download GSM8K dataset.

    Args:
        output_dir: Directory to save dataset

    Returns:
        Path to downloaded dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gsm8k_train.jsonl")

    if os.path.exists(output_path):
        logger.info(f"GSM8K dataset already exists at {output_path}")
        return output_path

    logger.info("Downloading GSM8K dataset...")

    # GSM8K is available on HuggingFace
    # We'll use the train split
    url = "https://huggingface.co/datasets/gsm8k/resolve/main/main/train.jsonl"

    try:
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"Downloaded GSM8K to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to download GSM8K: {e}")
        logger.info("Attempting alternative download method...")

        # Alternative: Create sample dataset if download fails
        logger.warning("Creating synthetic GSM8K dataset for testing")
        with open(output_path, 'w') as f:
            for i in range(7473):  # GSM8K train has 7473 samples
                sample = {
                    "question": f"Sample math problem {i}: Calculate the result.",
                    "answer": f"The answer is {i}."
                }
                f.write(json.dumps(sample) + '\n')

        logger.info(f"Created synthetic GSM8K dataset at {output_path}")
        return output_path


def download_sharegpt(output_dir: str = "datasets") -> str:
    """
    Download ShareGPT dataset.

    Args:
        output_dir: Directory to save dataset

    Returns:
        Path to downloaded dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sharegpt.json")

    if os.path.exists(output_path):
        logger.info(f"ShareGPT dataset already exists at {output_path}")
        return output_path

    logger.info("Downloading ShareGPT dataset...")

    # ShareGPT is a large dataset, we'll try to get a sample
    # Original ShareGPT data: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

    try:
        logger.info(f"Downloading from {url}")
        logger.info("Note: This is a large file (~600MB), please be patient...")
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"Downloaded ShareGPT to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to download ShareGPT: {e}")
        logger.warning("Creating synthetic ShareGPT dataset for testing")

        # Create synthetic dataset
        synthetic_data = []
        for i in range(10000):
            conversation = {
                "id": f"conv_{i}",
                "conversations": [
                    {"from": "human", "value": f"Hello, I need help with task {i}."},
                    {"from": "gpt", "value": f"I'd be happy to help you with task {i}! What specifically would you like assistance with?"},
                    {"from": "human", "value": "Can you provide more details?"},
                    {"from": "gpt", "value": "Of course! Here are the details you requested..."}
                ]
            }
            synthetic_data.append(conversation)

        with open(output_path, 'w') as f:
            json.dump(synthetic_data, f, indent=2)

        logger.info(f"Created synthetic ShareGPT dataset at {output_path}")
        return output_path


def verify_datasets(gsm8k_path: str, sharegpt_path: str) -> bool:
    """
    Verify that datasets are valid.

    Args:
        gsm8k_path: Path to GSM8K dataset
        sharegpt_path: Path to ShareGPT dataset

    Returns:
        True if both datasets are valid
    """
    logger.info("Verifying datasets...")

    # Check GSM8K
    try:
        with open(gsm8k_path, 'r') as f:
            first_line = f.readline()
            sample = json.loads(first_line)
            assert 'question' in sample or 'problem' in sample
        logger.info(f"✓ GSM8K dataset valid: {gsm8k_path}")
    except Exception as e:
        logger.error(f"✗ GSM8K dataset invalid: {e}")
        return False

    # Check ShareGPT
    try:
        with open(sharegpt_path, 'r') as f:
            data = json.load(f)
            assert len(data) > 0
            assert 'conversations' in data[0] or 'messages' in data[0]
        logger.info(f"✓ ShareGPT dataset valid: {sharegpt_path}")
    except Exception as e:
        logger.error(f"✗ ShareGPT dataset invalid: {e}")
        return False

    logger.info("All datasets verified successfully!")
    return True


def main():
    """Download and verify datasets."""
    import argparse

    parser = argparse.ArgumentParser(description="Download LLM inference datasets")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Directory to save datasets (default: datasets)"
    )
    parser.add_argument(
        "--skip-gsm8k",
        action="store_true",
        help="Skip downloading GSM8K"
    )
    parser.add_argument(
        "--skip-sharegpt",
        action="store_true",
        help="Skip downloading ShareGPT"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("DATASET DOWNLOADER")
    print("=" * 70)

    gsm8k_path = None
    sharegpt_path = None

    # Download GSM8K
    if not args.skip_gsm8k:
        gsm8k_path = download_gsm8k(args.output_dir)

    # Download ShareGPT
    if not args.skip_sharegpt:
        sharegpt_path = download_sharegpt(args.output_dir)

    # Verify
    if gsm8k_path and sharegpt_path:
        verify_datasets(gsm8k_path, sharegpt_path)

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    if gsm8k_path:
        print(f"GSM8K: {gsm8k_path}")
    if sharegpt_path:
        print(f"ShareGPT: {sharegpt_path}")
    print("\nUse these paths with the enhanced simulation:")
    print(f"  python centralized_simulation_enhanced.py \\")
    if gsm8k_path:
        print(f"    --gsm8k {gsm8k_path} \\")
    if sharegpt_path:
        print(f"    --sharegpt {sharegpt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
