"""
Dataset loader for GSM8K and ShareGPT datasets.

This module handles loading, truncation, and sampling of datasets for
creating realistic LLM inference workloads.
"""

import json
import random
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')


class DatasetLoader:
    """
    Loads and processes GSM8K and ShareGPT datasets.

    Handles:
    - Loading datasets from various sources
    - Truncating inputs to max token length (2048)
    - Random sampling (4K samples from each dataset)
    - Creating mixed workloads
    """

    def __init__(
        self,
        max_tokens: int = 2048,
        samples_per_dataset: int = 4000,
        seed: Optional[int] = 42
    ):
        """
        Initialize the dataset loader.

        Args:
            max_tokens: Maximum number of tokens for input truncation
            samples_per_dataset: Number of samples to randomly select from each dataset
            seed: Random seed for reproducibility
        """
        self.max_tokens = max_tokens
        self.samples_per_dataset = samples_per_dataset
        self.logger = logging.getLogger("DatasetLoader")

        if seed is not None:
            random.seed(seed)

        self.gsm8k_data: List[Dict[str, Any]] = []
        self.sharegpt_data: List[Dict[str, Any]] = []
        self.mixed_workload: List[Dict[str, Any]] = []

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.

        Uses a simple heuristic: ~4 characters per token on average.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to maximum token length.

        Args:
            text: Input text
            max_tokens: Maximum number of tokens

        Returns:
            Truncated text
        """
        estimated_tokens = self._estimate_tokens(text)
        if estimated_tokens <= max_tokens:
            return text

        # Truncate by character count (roughly 4 chars per token)
        max_chars = max_tokens * 4
        return text[:max_chars]

    def load_gsm8k(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load GSM8K dataset.

        GSM8K is a dataset of grade school math word problems.
        Format: {"question": "...", "answer": "..."}

        Args:
            file_path: Path to GSM8K JSON file. If None, uses dummy data for testing.

        Returns:
            List of GSM8K samples
        """
        if file_path and Path(file_path).exists():
            self.logger.info(f"Loading GSM8K from {file_path}")
            with open(file_path, 'r') as f:
                data = [json.loads(line) for line in f]
        else:
            # Generate dummy GSM8K data for testing
            self.logger.warning("GSM8K file not found, generating dummy data")
            data = []
            for i in range(10000):
                question = f"Problem {i}: If John has {i} apples and gives {i//2} to Mary, how many does he have left?"
                answer = f"{i - i//2}"
                data.append({
                    "question": question,
                    "answer": answer
                })

        # Sample and truncate
        if len(data) > self.samples_per_dataset:
            data = random.sample(data, self.samples_per_dataset)

        # Truncate questions to max tokens
        for sample in data:
            sample['question'] = self._truncate_text(sample['question'], self.max_tokens)

        self.gsm8k_data = data
        self.logger.info(f"Loaded {len(self.gsm8k_data)} GSM8K samples")
        return self.gsm8k_data

    def load_sharegpt(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load ShareGPT dataset.

        ShareGPT contains real conversation data from ChatGPT.
        Format: {"conversations": [{"from": "human/gpt", "value": "..."}]}

        Args:
            file_path: Path to ShareGPT JSON file. If None, uses dummy data for testing.

        Returns:
            List of ShareGPT samples
        """
        if file_path and Path(file_path).exists():
            self.logger.info(f"Loading ShareGPT from {file_path}")
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            # Generate dummy ShareGPT data for testing
            self.logger.warning("ShareGPT file not found, generating dummy data")
            data = []
            for i in range(10000):
                conversations = [
                    {"from": "human", "value": f"Hello, can you help me with task {i}?"},
                    {"from": "gpt", "value": f"Of course! I'd be happy to help you with task {i}."},
                    {"from": "human", "value": f"Great! Here are the details..."},
                ]
                data.append({
                    "id": f"conv_{i}",
                    "conversations": conversations
                })

        # Sample and truncate
        if len(data) > self.samples_per_dataset:
            data = random.sample(data, self.samples_per_dataset)

        # Truncate conversations to max tokens
        for sample in data:
            if 'conversations' in sample:
                # Concatenate all conversation turns
                full_text = " ".join([turn.get('value', '') for turn in sample['conversations']])
                # Truncate the full conversation
                truncated = self._truncate_text(full_text, self.max_tokens)
                # Store truncated version
                sample['truncated_text'] = truncated

        self.sharegpt_data = data
        self.logger.info(f"Loaded {len(self.sharegpt_data)} ShareGPT samples")
        return self.sharegpt_data

    def create_mixed_workload(self) -> List[Dict[str, Any]]:
        """
        Create a mixed workload from GSM8K and ShareGPT datasets.

        Combines both datasets and shuffles them to emulate real-world
        inference workloads with diverse request types.

        Returns:
            Shuffled list of all samples with dataset labels
        """
        if not self.gsm8k_data:
            self.logger.warning("GSM8K data not loaded, loading with defaults")
            self.load_gsm8k()

        if not self.sharegpt_data:
            self.logger.warning("ShareGPT data not loaded, loading with defaults")
            self.load_sharegpt()

        # Tag each sample with its source dataset
        mixed = []

        for sample in self.gsm8k_data:
            mixed.append({
                'dataset': 'gsm8k',
                'data': sample,
                'text': sample['question']
            })

        for sample in self.sharegpt_data:
            mixed.append({
                'dataset': 'sharegpt',
                'data': sample,
                'text': sample.get('truncated_text', '')
            })

        # Shuffle to create realistic mixed workload
        random.shuffle(mixed)

        self.mixed_workload = mixed
        self.logger.info(f"Created mixed workload with {len(self.mixed_workload)} samples")
        self.logger.info(f"  - GSM8K: {len(self.gsm8k_data)} samples")
        self.logger.info(f"  - ShareGPT: {len(self.sharegpt_data)} samples")

        return self.mixed_workload

    def get_sample(self, index: Optional[int] = None) -> Dict[str, Any]:
        """
        Get a sample from the mixed workload.

        Args:
            index: Index of sample to retrieve. If None, returns random sample.

        Returns:
            Sample dictionary
        """
        if not self.mixed_workload:
            self.create_mixed_workload()

        if index is None:
            return random.choice(self.mixed_workload)

        return self.mixed_workload[index % len(self.mixed_workload)]

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistics about loaded datasets.

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'gsm8k_count': len(self.gsm8k_data),
            'sharegpt_count': len(self.sharegpt_data),
            'mixed_count': len(self.mixed_workload),
            'max_tokens': self.max_tokens,
            'samples_per_dataset': self.samples_per_dataset
        }

        if self.gsm8k_data:
            avg_len = sum(len(s['question']) for s in self.gsm8k_data) / len(self.gsm8k_data)
            stats['gsm8k_avg_length'] = avg_len

        if self.sharegpt_data:
            avg_len = sum(len(s.get('truncated_text', '')) for s in self.sharegpt_data) / len(self.sharegpt_data)
            stats['sharegpt_avg_length'] = avg_len

        return stats


def main():
    """Example usage of DatasetLoader."""
    import argparse

    parser = argparse.ArgumentParser(description="Dataset Loader Test")
    parser.add_argument("--gsm8k", type=str, help="Path to GSM8K dataset")
    parser.add_argument("--sharegpt", type=str, help="Path to ShareGPT dataset")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens")
    parser.add_argument("--samples", type=int, default=4000, help="Samples per dataset")

    args = parser.parse_args()

    # Create loader
    loader = DatasetLoader(
        max_tokens=args.max_tokens,
        samples_per_dataset=args.samples
    )

    # Load datasets
    loader.load_gsm8k(args.gsm8k)
    loader.load_sharegpt(args.sharegpt)

    # Create mixed workload
    mixed = loader.create_mixed_workload()

    # Print statistics
    stats = loader.get_dataset_stats()
    print("\n=== Dataset Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Show some samples
    print("\n=== Sample Data ===")
    for i in range(3):
        sample = loader.get_sample(i)
        print(f"\nSample {i} ({sample['dataset']}):")
        print(f"  Text: {sample['text'][:100]}...")


if __name__ == "__main__":
    main()
