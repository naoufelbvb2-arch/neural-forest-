"""Trainer: orchestrates the pre-training loop with gradient tracking."""

from __future__ import annotations


class ForestTrainer:
    """Training loop for NeuralForest models.

    Handles: optimizer setup, gradient accumulation, checkpoint saving,
    W&B logging, and VRAM monitoring.

    TODO: implement in PROMPT 3
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("ForestTrainer — implement in PROMPT 3")

    def train(self) -> None:
        """Run the full training loop.

        TODO: implement in PROMPT 3
        """
        raise NotImplementedError("ForestTrainer.train — implement in PROMPT 3")
