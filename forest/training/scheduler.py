"""Learning rate scheduler and load-balance loss decay for Neural Forest."""
from __future__ import annotations

import math


class CosineWithWarmup:
    """Cosine learning rate schedule with linear warmup.

    Linear warmup from 0 to ``peak_lr`` over ``warmup_steps``,
    then cosine decay from ``peak_lr`` to ``min_lr`` over the remaining steps.

    Args:
        optimizer:     PyTorch optimizer to update.
        peak_lr:       Target LR after warmup.
        warmup_steps:  Number of linear warmup steps.
        max_steps:     Total training steps (warmup + decay).
        min_lr:        Minimum LR at end of schedule. Defaults to ``peak_lr * 0.1``.
    """

    def __init__(
        self,
        optimizer,
        peak_lr:      float,
        warmup_steps: int,
        max_steps:    int,
        min_lr:       float | None = None,
    ) -> None:
        self.optimizer     = optimizer
        self.peak_lr       = peak_lr
        self.warmup_steps  = warmup_steps
        self.max_steps     = max_steps
        self.min_lr        = min_lr if min_lr is not None else peak_lr * 0.1
        self.step_count    = 0

    def step(self) -> float:
        """Advance one training step, update optimizer lr, return new lr."""
        self.step_count += 1
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def get_lr(self) -> float:
        """Return the lr for ``step_count`` without advancing the counter."""
        t = self.step_count
        if t <= self.warmup_steps:
            return self.peak_lr * (t / max(1, self.warmup_steps))
        decay_steps = self.max_steps - self.warmup_steps
        progress    = (t - self.warmup_steps) / max(1, decay_steps)
        progress    = min(progress, 1.0)
        cos_val     = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cos_val


class LinearLoadBalanceDecay:
    """Linearly decay the load-balance loss coefficient over training.

    Starts at ``start_weight`` and interpolates to ``end_weight`` by
    ``max_steps``.  Allows the router to enforce load balance early, then
    relax as zones specialise.

    Args:
        start_weight:  Initial lb coefficient (default 0.01).
        end_weight:    Final lb coefficient (default 0.003).
        max_steps:     Steps over which decay occurs.
    """

    def __init__(
        self,
        start_weight: float = 0.01,
        end_weight:   float = 0.003,
        max_steps:    int   = 5_000,
    ) -> None:
        self.start_weight = start_weight
        self.end_weight   = end_weight
        self.max_steps    = max_steps
        self.step_count   = 0

    def step(self) -> float:
        """Advance one step and return the new weight."""
        self.step_count += 1
        return self.get_weight()

    def get_weight(self) -> float:
        """Return the weight for ``step_count`` without advancing the counter."""
        progress = min(self.step_count / max(1, self.max_steps), 1.0)
        return self.start_weight + (self.end_weight - self.start_weight) * progress
