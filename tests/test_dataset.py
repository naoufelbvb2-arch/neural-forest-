"""Tests for TinyStoriesDataset and create_dataloader.

Tests marked @pytest.mark.slow download data from the internet and are
excluded from the regular CI run:
    pytest tests/ -v -m "not slow"
"""
import pytest
import torch


@pytest.mark.slow
def test_dataset_builds() -> None:
    """Dataset loads and tokenises without error."""
    from forest.training.dataset import TinyStoriesDataset

    ds = TinyStoriesDataset(seq_len=64, max_tokens=10_000)
    assert len(ds) > 0
    assert ds.vocab_size == 50_257


@pytest.mark.slow
def test_dataset_item_shape() -> None:
    """Each item has shape (seq_len,) with dtype long."""
    from forest.training.dataset import TinyStoriesDataset

    ds   = TinyStoriesDataset(seq_len=64, max_tokens=10_000)
    item = ds[0]

    assert "input_ids" in item
    assert "labels"    in item
    assert item["input_ids"].shape == (64,)
    assert item["labels"].shape    == (64,)
    assert item["input_ids"].dtype == torch.long
    assert item["labels"].dtype    == torch.long


@pytest.mark.slow
def test_dataloader_batches() -> None:
    """DataLoader produces batches with correct shape."""
    from forest.training.dataset import TinyStoriesDataset, create_dataloader

    ds    = TinyStoriesDataset(seq_len=64, max_tokens=10_000)
    dl    = create_dataloader(ds, batch_size=4)
    batch = next(iter(dl))

    assert batch["input_ids"].shape == (4, 64)
    assert batch["labels"].shape    == (4, 64)
