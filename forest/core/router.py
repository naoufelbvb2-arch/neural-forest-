"""Router: assigns each token to one (or two) specialty zones."""

from __future__ import annotations

import torch
import torch.nn as nn

from forest.config import ForestConfig


class Router(nn.Module):
    """Token-to-zone routing with optional Skip Zone detection.

    During training: Gumbel-Softmax hard routing (Top-1).
    During inference: dynamic Top-1 or Top-2 based on confidence gap.

    Zone 0 is the Skip Zone — tokens routed there bypass all FFN computation.

    Args:
        config: ForestConfig instance.

    TODO: implement in PROMPT 2
    """

    def __init__(self, config: ForestConfig) -> None:
        super().__init__()
        self.config = config
        # TODO: implement linear projection + routing logic

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute zone assignments for each token.

        Args:
            hidden_states: Output from Spine, shape (batch, seq_len, embed_dim).

        Returns:
            zone_indices: Long tensor of shape (batch, seq_len) with zone index per token.
            router_logits: Float tensor of shape (batch, seq_len, num_zones) for aux loss.

        TODO: implement in PROMPT 2
        """
        raise NotImplementedError("Router.forward — implement in PROMPT 2")
