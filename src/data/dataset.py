"""
Language Modeling Dataset for FORDE LLM Pretraining.

Supports:
- Streaming datasets from Hugging Face (e.g., fineweb, slimpajama)
- Dummy data for testing
- Token packing and sequence length management
"""

import numpy as np

try:
    import datasets

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class DummyDataset:
    """Simple dummy dataset for testing training loop."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.rng = np.random.RandomState(42)
        self._index = 0

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= self.num_samples:
            raise StopIteration

        self._index += 1
        return {"input_ids": self.rng.randint(0, self.vocab_size, size=(self.seq_len,))}

    def __len__(self):
        return self.num_samples

    def batch(self, batch_size: int):
        """Return batched iterator."""
        return DummyBatchedDataset(self, batch_size)


class DummyBatchedDataset:
    """Batched wrapper for DummyDataset."""

    def __init__(self, dataset: DummyDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.vocab_size = dataset.vocab_size
        self.seq_len = dataset.seq_len
        self.num_samples = dataset.num_samples
        self._index = 0
        self.rng = np.random.RandomState(42)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= self.num_samples:
            raise StopIteration

        batch_samples = min(self.batch_size, self.num_samples - self._index)
        self._index += batch_samples

        return {
            "input_ids": self.rng.randint(
                0, self.vocab_size, size=(batch_samples, self.seq_len)
            )
        }


def create_dummy_dataset(
    vocab_size: int = 32000, seq_len: int = 512, num_samples: int = 10000
) -> DummyDataset:
    """
    Create a dummy dataset for testing.

    Args:
        vocab_size: Vocabulary size for random tokens
        seq_len: Sequence length
        num_samples: Number of samples to generate

    Returns:
        DummyDataset instance
    """
    return DummyDataset(vocab_size, seq_len, num_samples)


def create_lm_dataset(
    dataset_name: str = "HuggingFaceFW/fineweb",
    split: str = "train",
    vocab_size: int = 32000,
    max_seq_len: int = 512,
    streaming: bool = True,
):
    """
    Create a language modeling dataset from Hugging Face.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split
        vocab_size: Vocabulary size (for tokenizer compatibility)
        max_seq_len: Maximum sequence length
        streaming: Whether to stream the dataset

    Returns:
        Dataset that yields tokenized examples
    """
    if not HAS_DATASETS:
        print("Warning: 'datasets' package not available. Using dummy data.")
        return create_dummy_dataset(vocab_size, max_seq_len)

    try:
        from transformers import AutoTokenizer

        # Load dataset
        # For FineWeb, we usually want to use a specific subset or sample if not specified,
        # but 'HuggingFaceFW/fineweb' requires a subset configuration usually.
        # Fallback to 'sample-10BT' if it is the main repository.
        subset = None
        if dataset_name == "HuggingFaceFW/fineweb":
            subset = "sample-10BT"

        dataset = datasets.load_dataset(
            dataset_name, name=subset, streaming=streaming, split=split
        )

        # Load tokenizer (using GPT-2 style tokenizer)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Streaming friendly tokenization
        if streaming:

            def streaming_tokenize(examples):
                text_field = "text" if "text" in examples else list(examples.keys())[0]
                texts = examples[text_field]
                return tokenizer(
                    texts,
                    truncation=True,
                    max_length=max_seq_len,
                    padding="max_length",
                    return_tensors="np",
                )

            # For streaming, we map and then format
            tokenized_dataset = dataset.map(
                streaming_tokenize,
                batched=True,
                remove_columns=[
                    "text",
                    "id",
                    "dump",
                    "url",
                    "date",
                    "file_path",
                    "language",
                    "language_score",
                    "token_count",
                ],
            )
            # Since map on streaming returns an IterableDataset, we wrap it
            return StreamingLMDataset(tokenized_dataset)

        else:
            # Logic for non-streaming (existing logic)
            def tokenize_function(examples):
                text_field = "text" if "text" in examples else list(examples.keys())[0]
                texts = examples[text_field]

                tokenized = tokenizer(
                    texts,
                    truncation=True,
                    max_length=max_seq_len,
                    padding="max_length",
                    return_tensors="np",
                )

                return {"input_ids": tokenized["input_ids"]}

            tokenized_dataset = dataset.map(
                tokenize_function, batched=True, remove_columns=dataset.column_names
            )

            return tokenized_dataset

    except Exception as e:
        print(f"Warning: Failed to load dataset '{dataset_name}': {e}")
        print("Falling back to dummy data.")
        return create_dummy_dataset(vocab_size, max_seq_len)


class StreamingLMDataset:
    """Wrapper for streaming datasets with batching support."""

    def __init__(self, hf_dataset, batch_size: int = 8):
        self.dataset = hf_dataset
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for example in self.dataset:
            batch.append(example["input_ids"])
            if len(batch) >= self.batch_size:
                yield {"input_ids": np.stack(batch)}
                batch = []

        # Yield remaining items
        if batch:
            yield {"input_ids": np.stack(batch)}

    def batch(self, batch_size: int):
        """Configure batch size and return self."""
        self.batch_size = batch_size
        return self


if __name__ == "__main__":
    print("--- Testing Dataset Loaders ---")

    # Test dummy dataset
    print("\n1. Testing DummyDataset:")
    dummy = create_dummy_dataset(vocab_size=1000, seq_len=64, num_samples=100)
    batched = dummy.batch(8)

    batch = next(iter(batched))
    print(f"   Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"   Batch input_ids dtype: {batch['input_ids'].dtype}")
    print(f"   Sample tokens: {batch['input_ids'][0, :10]}")

    # Count batches
    dummy2 = create_dummy_dataset(vocab_size=1000, seq_len=64, num_samples=100)
    batched2 = dummy2.batch(8)
    num_batches = sum(1 for _ in batched2)
    print(f"   Total batches: {num_batches}")

    print("\n--- Dataset tests passed! ---")
