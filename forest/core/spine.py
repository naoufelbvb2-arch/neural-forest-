"""Shared Spine: multi-layer attention backbone, always resident in VRAM."""

from __future__ import annotations

import torch
import torch.nn as nn

from forest.config import ForestConfig


class Spine(nn.Module):
    """Shared attention backbone shared across all tokens and zones.

    The Spine runs first for every forward pass. It maintains a coherent KV
    cache because attention is never fragmented across zone switches.

    Args:
        config: ForestConfig instance.

    TODO: implement in PROMPT 2
    """

    def __init__(self, config: ForestConfig) -> None:
        super().__init__()
        self.config = config
        # TODO: implement attention layers, layer norms, positional encodings

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the shared spine.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).
            attention_mask: Optional causal or padding mask.

        Returns:
            Hidden states of shape (batch, seq_len, embed_dim).

        TODO: implement in PROMPT 2
        """
        raise NotImplementedError("Spine.forward — implement in PROMPT 2")
