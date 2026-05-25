"""Dataset utilities for loading and streaming pre-training corpora."""

from __future__ import annotations

from torch.utils.data import Dataset


class ForestDataset(Dataset):
    """Streaming dataset for language model pre-training.

    Supports FineWeb-Edu, FineWeb-Edu-Ar, OpenWebMath, The Stack v2, etc.

    TODO: implement in PROMPT 3
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("ForestDataset — implement in PROMPT 3")
