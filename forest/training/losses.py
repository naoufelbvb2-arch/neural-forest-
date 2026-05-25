"""Loss functions: language modeling loss + router auxiliary losses."""

from __future__ import annotations

import torch


def language_modeling_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy language modeling loss.

    Args:
        logits: Shape (batch, seq_len, vocab_size).
        labels: Shape (batch, seq_len), -100 for ignored positions.

    Returns:
        Scalar loss tensor.

    TODO: implement in PROMPT 3
    """
    raise NotImplementedError("language_modeling_loss — implement in PROMPT 3")


def router_load_balance_loss(
    router_logits: torch.Tensor,
    num_zones: int,
) -> torch.Tensor:
    """Auxiliary load-balancing loss to prevent zone collapse.

    Args:
        router_logits: Shape (batch * seq_len, num_zones).
        num_zones: Total number of zones.

    Returns:
        Scalar auxiliary loss tensor.

    TODO: implement in PROMPT 3
    """
    raise NotImplementedError("router_load_balance_loss — implement in PROMPT 3")
