"""NeuralForest: top-level model assembling Spine + Router + Zones."""

from __future__ import annotations

import torch
import torch.nn as nn

from forest.config import ForestConfig


class NeuralForest(nn.Module):
    """Sparse MoE language model combining Spine, Router, and Zones.

    Forward pass:
        1. Embed tokens
        2. Run Shared Spine (attention, always in VRAM)
        3. Router assigns each token to a zone
        4. Each assigned zone processes its tokens (FFN)
        5. LM Head produces logits

    Args:
        config: ForestConfig instance.

    TODO: implement in PROMPT 2
    """

    def __init__(self, config: ForestConfig) -> None:
        super().__init__()
        self.config = config
        # TODO: implement embedding, spine, router, zones, lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            input_ids: Token IDs, shape (batch, seq_len).
            attention_mask: Optional mask, shape (batch, seq_len).
            labels: Optional shifted token IDs for language modeling loss.

        Returns:
            Dictionary with keys: logits, loss (if labels provided), router_logits.

        TODO: implement in PROMPT 2
        """
        raise NotImplementedError("NeuralForest.forward — implement in PROMPT 2")
