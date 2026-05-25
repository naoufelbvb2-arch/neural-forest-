"""Specialty Zone: domain-specific FFN expert loaded from RAM on demand."""

from __future__ import annotations

import torch
import torch.nn as nn

from forest.config import ForestConfig


class Zone(nn.Module):
    """Multi-layer SwiGLU FFN expert — one instance per domain specialty.

    Zones live in RAM (WARM tier) and are loaded into VRAM only when activated
    by the Router. Zone 0 (Skip) is a no-op and never instantiated as a Zone.

    Architecture per zone:
        Linear(embed_dim, hidden) + SiLU   # gating
        Linear(embed_dim, hidden)           # value
        element-wise multiply (GLU)
        [optional: Linear(hidden, hidden) + SiLU — middle layer]
        Linear(hidden, embed_dim)           # project back

    Args:
        config: ForestConfig instance.
        zone_id: Zone index (1-10). Used for naming and specialization.

    TODO: implement in PROMPT 2
    """

    def __init__(self, config: ForestConfig, zone_id: int) -> None:
        super().__init__()
        self.config = config
        self.zone_id = zone_id
        # TODO: implement SwiGLU FFN layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process tokens assigned to this zone.

        Args:
            x: Token hidden states, shape (num_tokens, embed_dim).

        Returns:
            Transformed hidden states, same shape as input.

        TODO: implement in PROMPT 2
        """
        raise NotImplementedError(f"Zone({self.zone_id}).forward — implement in PROMPT 2")
