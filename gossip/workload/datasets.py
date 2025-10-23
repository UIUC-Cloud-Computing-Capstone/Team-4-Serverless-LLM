"""
Dataset loaders for realistic LLM inference workloads.

Implements loading and preprocessing for:
- GSM8K: Grade school math word problems (reasoning tasks)
- ShareGPT: Multi-language chat conversations (chat tasks)

Both datasets are truncated to max context length and sampled to create
mixed workloads that emulate real-world inference patterns.
"""

import json
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    from datasets import load_dataset
    import tiktoken
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("Warning: 'datasets' or 'tiktoken' not installed. Install with: pip install datasets tiktoken")


@dataclass
class InferenceRequest:
    """Represents a single inference request with input prompt."""

    request_id: str
    prompt: str
    dataset_source: str  # "gsm8k" or "sharegpt"
    token_count: int
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'request_id': self.request_id,
            'prompt': self.prompt,
            'dataset_source': self.dataset_source,
            'token_count': self.token_count,
            'metadata': self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'InferenceRequest':
        """Create from dictionary."""
        return cls(
            request_id=data['request_id'],
            prompt=data['prompt'],
            dataset_source=data['dataset_source'],
            token_count=data['token_count'],
            metadata=data.get('metadata')
        )


class TokenCounter:
    """Token counting using tiktoken (OpenAI's tokenizer)."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize token counter.

        Args:
            encoding_name: Tokenizer encoding to use
                - cl100k_base: GPT-4, GPT-3.5-turbo
                - p50k_base: Older GPT-3 models
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("tiktoken is required for token counting")

        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to maximum token count."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)


class GSM8KLoader:
    """Loader for GSM8K dataset (Grade School Math 8K)."""

    def __init__(self, max_tokens: int = 2048, cache_dir: Optional[str] = None):
        """
        Initialize GSM8K loader.

        Args:
            max_tokens: Maximum context length (truncate if exceeded)
            cache_dir: Directory to cache downloaded dataset
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("datasets library is required. Install with: pip install datasets")

        self.max_tokens = max_tokens
        self.cache_dir = cache_dir
        self.token_counter = TokenCounter()
        self._dataset = None

    def load(self, split: str = "train", num_samples: Optional[int] = None) -> List[InferenceRequest]:
        """
        Load GSM8K dataset.

        Args:
            split: Dataset split ("train" or "test")
            num_samples: Number of samples to randomly select (None = all)

        Returns:
            List of inference requests with math problems
        """
        print(f"Loading GSM8K dataset (split={split})...")

        # Load dataset from HuggingFace
        dataset = load_dataset("gsm8k", "main", split=split, cache_dir=self.cache_dir)

        # Sample if requested
        if num_samples and num_samples < len(dataset):
            indices = random.sample(range(len(dataset)), num_samples)
            dataset = dataset.select(indices)

        # Convert to inference requests
        requests = []
        for idx, example in enumerate(dataset):
            # GSM8K format: {"question": "...", "answer": "..."}
            prompt = example['question']

            # Truncate to max tokens
            token_count = self.token_counter.count_tokens(prompt)
            if token_count > self.max_tokens:
                prompt = self.token_counter.truncate_to_tokens(prompt, self.max_tokens)
                token_count = self.max_tokens

            request = InferenceRequest(
                request_id=f"gsm8k_{split}_{idx}",
                prompt=prompt,
                dataset_source="gsm8k",
                token_count=token_count,
                metadata={
                    'split': split,
                    'original_answer': example['answer'],
                    'index': idx
                }
            )
            requests.append(request)

        print(f"Loaded {len(requests)} GSM8K samples (avg {sum(r.token_count for r in requests) / len(requests):.1f} tokens)")
        return requests


class ShareGPTLoader:
    """Loader for ShareGPT dataset (multi-language chat conversations)."""

    def __init__(self, max_tokens: int = 2048, cache_dir: Optional[str] = None):
        """
        Initialize ShareGPT loader.

        Args:
            max_tokens: Maximum context length (truncate if exceeded)
            cache_dir: Directory to cache downloaded dataset
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("datasets library is required. Install with: pip install datasets")

        self.max_tokens = max_tokens
        self.cache_dir = cache_dir
        self.token_counter = TokenCounter()

    def load(self, num_samples: Optional[int] = None) -> List[InferenceRequest]:
        """
        Load ShareGPT dataset.

        Note: ShareGPT is a community dataset with conversations.
        Using anon8231489123/ShareGPT_Vicuna_unfiltered as the source.

        Args:
            num_samples: Number of samples to randomly select (None = all)

        Returns:
            List of inference requests with chat prompts
        """
        print("Loading ShareGPT dataset...")

        # Load ShareGPT dataset from HuggingFace
        # This is a common ShareGPT source used in LLM research
        try:
            dataset = load_dataset(
                "anon8231489123/ShareGPT_Vicuna_unfiltered",
                split="train",
                cache_dir=self.cache_dir
            )
        except Exception as e:
            print(f"Warning: Could not load ShareGPT dataset: {e}")
            print("Falling back to mock ShareGPT data for testing...")
            return self._load_mock_sharegpt(num_samples or 100)

        # Sample if requested
        if num_samples and num_samples < len(dataset):
            indices = random.sample(range(len(dataset)), num_samples)
            dataset = dataset.select(indices)

        # Convert to inference requests
        requests = []
        for idx, example in enumerate(dataset):
            # ShareGPT format: {"conversations": [{"from": "human", "value": "..."}]}
            conversations = example.get('conversations', [])

            # Extract first user message as prompt
            prompt = ""
            for conv in conversations:
                if conv.get('from') in ['human', 'user']:
                    prompt = conv.get('value', '')
                    break

            if not prompt:
                continue  # Skip if no user message found

            # Truncate to max tokens
            token_count = self.token_counter.count_tokens(prompt)
            if token_count > self.max_tokens:
                prompt = self.token_counter.truncate_to_tokens(prompt, self.max_tokens)
                token_count = self.max_tokens

            request = InferenceRequest(
                request_id=f"sharegpt_{idx}",
                prompt=prompt,
                dataset_source="sharegpt",
                token_count=token_count,
                metadata={
                    'conversation_length': len(conversations),
                    'index': idx
                }
            )
            requests.append(request)

        print(f"Loaded {len(requests)} ShareGPT samples (avg {sum(r.token_count for r in requests) / len(requests):.1f} tokens)")
        return requests

    def _load_mock_sharegpt(self, num_samples: int) -> List[InferenceRequest]:
        """Create mock ShareGPT data for testing when dataset unavailable."""
        mock_prompts = [
            "Can you explain how neural networks work?",
            "What's the difference between supervised and unsupervised learning?",
            "Write a Python function to calculate fibonacci numbers.",
            "Translate this to Spanish: Hello, how are you?",
            "What are the main causes of climate change?",
            "Explain quantum computing to a 10-year-old.",
            "How do I make chocolate chip cookies?",
            "What's the capital of France and its population?",
            "Debug this code: for i in range(10) print(i)",
            "Summarize the plot of Shakespeare's Hamlet.",
        ]

        requests = []
        for i in range(num_samples):
            prompt = random.choice(mock_prompts)
            token_count = self.token_counter.count_tokens(prompt)

            request = InferenceRequest(
                request_id=f"mock_sharegpt_{i}",
                prompt=prompt,
                dataset_source="sharegpt_mock",
                token_count=token_count,
                metadata={'mock': True, 'index': i}
            )
            requests.append(request)

        print(f"Created {len(requests)} mock ShareGPT samples")
        return requests


def load_mixed_dataset(
    gsm8k_samples: int = 4000,
    sharegpt_samples: int = 4000,
    max_tokens: int = 2048,
    cache_dir: Optional[str] = None,
    shuffle: bool = True,
    save_path: Optional[str] = None
) -> List[InferenceRequest]:
    """
    Load mixed dataset combining GSM8K and ShareGPT.

    This creates a realistic mixed workload with:
    - Math reasoning tasks (GSM8K)
    - Chat/instruction tasks (ShareGPT)

    Args:
        gsm8k_samples: Number of GSM8K samples to load
        sharegpt_samples: Number of ShareGPT samples to load
        max_tokens: Maximum context length for truncation
        cache_dir: Directory to cache downloaded datasets
        shuffle: Whether to shuffle the combined dataset
        save_path: Optional path to save dataset as JSON

    Returns:
        List of mixed inference requests
    """
    print("=" * 70)
    print("LOADING MIXED DATASET")
    print("=" * 70)

    # Load GSM8K
    gsm8k_loader = GSM8KLoader(max_tokens=max_tokens, cache_dir=cache_dir)
    gsm8k_requests = gsm8k_loader.load(split="train", num_samples=gsm8k_samples)

    # Load ShareGPT
    sharegpt_loader = ShareGPTLoader(max_tokens=max_tokens, cache_dir=cache_dir)
    sharegpt_requests = sharegpt_loader.load(num_samples=sharegpt_samples)

    # Combine datasets
    all_requests = gsm8k_requests + sharegpt_requests

    if shuffle:
        random.shuffle(all_requests)

    # Print statistics
    print("\n" + "=" * 70)
    print(f"Total samples: {len(all_requests)}")
    print(f"  - GSM8K: {len(gsm8k_requests)} ({len(gsm8k_requests)/len(all_requests)*100:.1f}%)")
    print(f"  - ShareGPT: {len(sharegpt_requests)} ({len(sharegpt_requests)/len(all_requests)*100:.1f}%)")

    token_counts = [r.token_count for r in all_requests]
    print(f"\nToken statistics:")
    print(f"  - Mean: {sum(token_counts) / len(token_counts):.1f}")
    print(f"  - Min: {min(token_counts)}")
    print(f"  - Max: {max(token_counts)}")
    print("=" * 70)

    # Save if requested
    if save_path:
        save_dataset(all_requests, save_path)

    return all_requests


def save_dataset(requests: List[InferenceRequest], save_path: str):
    """Save dataset to JSON file."""
    data = [r.to_dict() for r in requests]

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved dataset to {save_path}")


def load_dataset_from_file(load_path: str) -> List[InferenceRequest]:
    """Load dataset from JSON file."""
    with open(load_path, 'r') as f:
        data = json.load(f)

    requests = [InferenceRequest.from_dict(item) for item in data]
    print(f"Loaded {len(requests)} requests from {load_path}")
    return requests


if __name__ == "__main__":
    # Example usage
    print("Testing dataset loaders...\n")

    # Test with small samples
    requests = load_mixed_dataset(
        gsm8k_samples=10,
        sharegpt_samples=10,
        max_tokens=2048,
        shuffle=True
    )

    # Print first few examples
    print("\nFirst 3 examples:")
    for i, req in enumerate(requests[:3]):
        print(f"\n{i+1}. {req.dataset_source} ({req.token_count} tokens)")
        print(f"   Prompt: {req.prompt[:100]}...")
