"""TinyStories dataset loader with GPT-2 BPE tokenization.

Used for smoke testing the training pipeline on CPU.
Pre-tokenizes the entire corpus into a flat tensor of token IDs,
then serves non-overlapping sequences of length seq_len.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


class TinyStoriesDataset(Dataset):
    """TinyStories dataset — children's stories with simple vocabulary.

    Pre-tokenizes the entire dataset into a flat array of token IDs,
    then samples fixed-length sequences of length ``seq_len`` during training.

    Args:
        seq_len:    Length of each training sequence (tokens).
        max_tokens: Total tokens to load (caps dataset size for faster CPU runs).
        split:      "train" or "validation".
        cache_dir:  Where to cache the HuggingFace dataset download.
    """

    def __init__(
        self,
        seq_len: int = 512,
        max_tokens: int = 500_000,
        split: str = "train",
        cache_dir: Optional[str] = None,
    ) -> None:
        self.seq_len    = seq_len
        self.max_tokens = max_tokens

        # Lazy imports — keep base package light
        import tiktoken
        from datasets import load_dataset

        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab  # 50257

        print(f"[Dataset] Loading TinyStories ({split})...")
        ds = load_dataset(
            "roneneldan/TinyStories",
            split=split,
            cache_dir=cache_dir,
        )

        print(f"[Dataset] Tokenizing up to {max_tokens:,} tokens...")
        token_ids: list[int] = []
        for example in ds:
            tokens = self.tokenizer.encode(example["text"])
            tokens.append(50256)          # end-of-text marker
            token_ids.extend(tokens)
            if len(token_ids) >= max_tokens:
                break

        token_ids = token_ids[:max_tokens]
        self.tokens = torch.tensor(token_ids, dtype=torch.long)

        print(f"[Dataset] Loaded {len(self.tokens):,} tokens")
        print(f"[Dataset] Sequences available: {len(self):,}")

    def __len__(self) -> int:
        # Number of non-overlapping sequences; -1 so labels never go out of bounds
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return one training example.

        Returns:
            input_ids: shape (seq_len,) — token IDs fed to the model.
            labels:    shape (seq_len,) — same as input_ids; model.forward
                       shifts by 1 internally for causal LM loss.
        """
        start = idx * self.seq_len
        end   = start + self.seq_len
        input_ids = self.tokens[start:end]
        labels    = self.tokens[start:end].clone()
        return {"input_ids": input_ids, "labels": labels}


def create_dataloader(
    dataset: TinyStoriesDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Standard DataLoader factory for TinyStoriesDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,   # CPU training
        drop_last=True,     # consistent batch size
    )
