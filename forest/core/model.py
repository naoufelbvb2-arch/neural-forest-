"""NeuralForest: top-level model assembling Spine + Router + ZoneCollection + LM head.

Full forward pass:
    input_ids
        → SharedSpine   (embed + L causal-attention layers + final_norm)
        → ForestRouter  (assign each token to a zone; Gumbel in training, top-k in eval)
        → ZoneCollection (apply domain-specific SwiGLU FFN; skip zone is a no-op)
        → lm_head       (linear projection to vocab_size; weight-tied with embedding)
        → logits  [+ cross-entropy loss + load-balance auxiliary loss when labels given]

KV-cache protocol: pass kv_cache=None for training / first inference step;
    pass the returned kv_cache list for each subsequent decoding step.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from forest.config import ForestConfig
from forest.core.spine import SharedSpine
from forest.core.router import ForestRouter, RoutingDecision
from forest.core.zone import ZoneCollection


class NeuralForest(nn.Module):
    """Sparse MoE language model: Spine → Router → Zones → LM head.

    The model respects ``self.training`` to switch the Router between
    stochastic Gumbel-Softmax (training) and deterministic top-k (inference).

    Weight tying (``config.tie_word_embeddings=True``, default):
        ``lm_head.weight`` is set to ``spine.token_embedding.weight``,
        saving ``vocab_size × embed_dim`` parameters with no quality loss.

    Args:
        config: ForestConfig instance.
    """

    def __init__(self, config: ForestConfig) -> None:
        super().__init__()
        self.config = config

        self.spine  = SharedSpine(config)
        self.router = ForestRouter(config)
        self.zones  = ZoneCollection(config)

        # LM head: (embed_dim → vocab_size), no bias
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying: share the embedding matrix with the output projection.
        # This assignment makes lm_head.weight the *same object* as the embedding
        # weight — they share storage and receive the same gradient.
        if config.tie_word_embeddings:
            self.lm_head.weight = self.spine.token_embedding.weight

        # Coefficient for the load-balance auxiliary loss.
        # Stored as a plain float so the trainer (or lb_decay) can update it
        # without touching router internals.
        self.load_balance_weight: float = 0.01

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding (for external weight manipulation)."""
        return self.spine.get_input_embeddings()

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count actual model parameters (not the config estimate)."""
        params = self.parameters() if not trainable_only else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)

    def count_parameters(self) -> dict[str, int]:
        """Breakdown of actual trainable parameters by component.

        Returns:
            Dict with keys: spine, router, zones, lm_head, total.
            When weight tying is on, lm_head=0 (shared with spine embedding).
        """
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        spine_count  = _count(self.spine)
        router_count = _count(self.router)
        zones_count  = _count(self.zones)

        # lm_head shares weights with spine embedding when tied → count as 0
        if self.config.tie_word_embeddings:
            lm_head_count = 0
        else:
            lm_head_count = _count(self.lm_head)

        total = spine_count + router_count + zones_count + lm_head_count
        return {
            "spine":   spine_count,
            "router":  router_count,
            "zones":   zones_count,
            "lm_head": lm_head_count,
            "total":   total,
        }

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> dict[str, Any]:
        """Full NeuralForest forward pass.

        Args:
            input_ids:  Token IDs, shape (batch, seq_len).
            labels:     Optional target IDs for language modelling loss,
                        shape (batch, seq_len). Positions labelled -100 are ignored.
                        Loss is computed as CE(logits[:, :-1], labels[:, 1:])
                        + routing load-balance auxiliary loss.
            kv_cache:   Per-layer (K, V) tensors from a prior step.
                        None → full causal pass (training or first decoding step).
                        List of length spine_layers → incremental decoding.

        Returns:
            dict with keys:
                "logits"            – (batch, seq_len, vocab_size)
                "routing_decision"  – RoutingDecision (zone assignments, entropy, …)
                "kv_cache"          – updated list of (K, V) for next step
                "loss"              – scalar total loss (only when labels is not None)
        """
        # ── 1. Shared Spine ────────────────────────────────────────────
        hidden_states, new_kv_cache = self.spine(input_ids, kv_cache)
        # hidden_states: (B, T, embed_dim)

        # ── 2. Router ──────────────────────────────────────────────────
        # training_mode matches the nn.Module training flag so that
        # model.train() / model.eval() automatically selects the right strategy.
        routing_decision: RoutingDecision = self.router(
            hidden_states, training_mode=self.training
        )

        # ── 3. Zone dispatch ───────────────────────────────────────────
        # ZoneCollection applies each token's assigned SwiGLU zone.
        # Skip-zone tokens (zone 0) pass through unchanged (zero FFN contribution).
        zone_output = self.zones(hidden_states, routing_decision)
        # zone_output: (B, T, embed_dim)

        # ── 4. LM head ─────────────────────────────────────────────────
        logits = self.lm_head(zone_output)  # (B, T, vocab_size)

        out: dict[str, Any] = {
            "logits":           logits,
            "routing_decision": routing_decision,
            "kv_cache":         new_kv_cache,
        }

        # ── 5. Language modelling loss (optional) ──────────────────────
        if labels is not None:
            # Causal LM: position i predicts position i+1.
            # Shift by one so that each token predicts the next.
            shift_logits = logits[:, :-1].contiguous()          # (B, T-1, V)
            shift_labels = labels[:, 1:].contiguous()           # (B, T-1)

            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Weight the raw load-balance loss by self.load_balance_weight.
            # The trainer (or LinearLoadBalanceDecay) can adjust this coefficient.
            out["lm_loss"] = lm_loss
            out["loss"]    = lm_loss + routing_decision.load_balance_loss * self.load_balance_weight

        return out
