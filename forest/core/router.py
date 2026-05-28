"""ForestRouter: token-to-zone routing with Gumbel-Softmax, load balancing, and skip detection.

Routing protocol:
  Training  → Top-1 Gumbel-Softmax (hard=True) for discrete zone assignments.
              Gradient flows via the Straight-Through Estimator (STE).
  Inference → Dynamic Top-1/Top-2 based on confidence gap between top-2 probabilities.
              High confidence (gap > threshold) → Top-1.
              Low  confidence (gap ≤ threshold) → Top-2 with normalised weights.

Zone 0 is the Skip Zone: tokens routed there bypass all FFN computation entirely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from forest.config import ForestConfig


# ---------------------------------------------------------------------------
# RoutingDecision
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    """Container for all outputs of one ForestRouter forward pass.

    Attributes:
        zone_indices:       Primary zone per token, shape (batch, seq). Long tensor.
        zone_weights:       Weights for zone outputs.
                            Training → (batch, seq, 1), always 1.0 in the forward pass
                            but with STE gradient through Gumbel-Softmax.
                            Inference → (batch, seq, 2) with normalised weights; the
                            second weight is 0 for high-confidence tokens.
        top_k_indices:      Zone indices for each selected expert.
                            Training → (batch, seq, 1).
                            Inference → (batch, seq, 2).
        logits:             Raw gate logits, shape (batch, seq, num_zones).
        load_balance_loss:  Switch-Transformer load-balancing auxiliary loss (scalar).
                            Raw unweighted value (range ~[1, num_zones]).
                            Multiply by ``NeuralForest.load_balance_weight`` before
                            adding to the cross-entropy loss.
        entropy:            Mean routing entropy (scalar). For logging only — not
                            added to loss. Low entropy = sharp specialisation.
        skip_mask:          True for tokens routed to Zone 0 (skip), shape (batch, seq).
    """

    zone_indices:       torch.Tensor
    zone_weights:       torch.Tensor
    top_k_indices:      torch.Tensor
    logits:             torch.Tensor
    load_balance_loss:  torch.Tensor
    entropy:            torch.Tensor
    skip_mask:          torch.Tensor


# ---------------------------------------------------------------------------
# ForestRouter
# ---------------------------------------------------------------------------

class ForestRouter(nn.Module):
    """Learnable token-to-zone router for Neural Forest.

    A single linear gate maps each hidden state to zone logits.
    The routing strategy differs between training and inference (see module docstring).

    Args:
        config: ForestConfig instance.

    Hyperparameters (not in ForestConfig — routing-specific):
        confidence_threshold:  Gap between top-2 probs that triggers Top-2 routing
                               during inference (default 0.3).

    Note:
        ``load_balance_loss`` in RoutingDecision is the *raw* unweighted loss.
        Multiply by ``NeuralForest.load_balance_weight`` before adding to CE loss.
    """

    confidence_threshold: float = 0.3

    def __init__(self, config: ForestConfig) -> None:
        super().__init__()
        self.num_zones      = config.num_zones       # 11 (0 = skip + 10 specialty)
        self.top_k          = config.router_top_k    # 1 by default
        self.use_skip_zone  = config.use_skip_zone

        # Single linear projection: no bias (avoids constant zone preference)
        self.gate = nn.Linear(config.embed_dim, config.num_zones, bias=False)

        # Small init: start near-uniform so all zones are explored early in training
        nn.init.normal_(self.gate.weight, std=0.01)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        training_mode: bool = True,
    ) -> RoutingDecision:
        """Compute zone assignments for every token in the batch.

        Args:
            hidden_states: Spine output, shape (batch, seq, embed_dim).
            training_mode: If True, use Gumbel-Softmax (differentiable, stochastic).
                           If False, use softmax + dynamic top-k (deterministic).

        Returns:
            RoutingDecision with all routing information.
        """
        B, T, _ = hidden_states.shape

        logits = self.gate(hidden_states)           # (B, T, num_zones)
        probs  = F.softmax(logits, dim=-1)          # (B, T, num_zones) — for aux losses

        if training_mode:
            zone_indices, zone_weights, top_k_indices = self._gumbel_routing(logits)
        else:
            zone_indices, zone_weights, top_k_indices = self._inference_routing(probs)

        load_balance_loss = self._compute_load_balance_loss(probs, zone_indices)
        entropy           = self._compute_entropy(probs)

        if self.use_skip_zone:
            skip_mask = zone_indices == 0
        else:
            skip_mask = torch.zeros(B, T, dtype=torch.bool, device=hidden_states.device)

        return RoutingDecision(
            zone_indices      = zone_indices,
            zone_weights      = zone_weights,
            top_k_indices     = top_k_indices,
            logits            = logits,
            load_balance_loss = load_balance_loss,
            entropy           = entropy,
            skip_mask         = skip_mask,
        )

    # ------------------------------------------------------------------
    # Routing strategies
    # ------------------------------------------------------------------

    def _gumbel_routing(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-1 Gumbel-Softmax routing for training.

        Uses hard=True so the forward pass is discrete (one-hot) while the
        backward pass uses the soft Gumbel-Softmax (STE gradient).

        Args:
            logits: Shape (batch, seq, num_zones).

        Returns:
            zone_indices:  (batch, seq) long — primary zone per token.
            zone_weights:  (batch, seq, 1) — 1.0 forward, STE gradient backward.
            top_k_indices: (batch, seq, 1) long.
        """
        # hard=True: forward gives one-hot; backward uses soft Gumbel-Softmax (STE)
        one_hot = F.gumbel_softmax(logits, tau=1.0, hard=True)  # (B, T, num_zones)

        zone_indices  = one_hot.argmax(dim=-1)                              # (B, T)
        # Gather preserves the STE gradient from one_hot
        zone_weights  = one_hot.gather(-1, zone_indices.unsqueeze(-1))      # (B, T, 1)
        top_k_indices = zone_indices.unsqueeze(-1)                          # (B, T, 1)

        return zone_indices, zone_weights, top_k_indices

    def _inference_routing(
        self,
        probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dynamic Top-1/Top-2 routing for inference.

        Returns top-2 indices and weights for every token.
        For tokens where the confidence gap exceeds `confidence_threshold`,
        the second weight is set to 0 (effective Top-1).

        Args:
            probs: Softmax probabilities, shape (batch, seq, num_zones).

        Returns:
            zone_indices:  (batch, seq) long — primary (top-1) zone.
            zone_weights:  (batch, seq, 2) — normalised; second may be 0.
            top_k_indices: (batch, seq, 2) long.
        """
        top2_probs, top2_indices = probs.topk(2, dim=-1)      # (B, T, 2)
        diff = top2_probs[..., 0] - top2_probs[..., 1]        # (B, T)

        # Tokens where top-1 is clearly dominant → zero out the second weight
        high_confidence = diff >= self.confidence_threshold    # (B, T) bool

        # Normalise so weights sum to 1 per token
        norm = top2_probs / top2_probs.sum(dim=-1, keepdim=True)

        # For high-confidence tokens: weight = [1.0, 0.0]
        # For low-confidence tokens:  weight = [norm1, norm2]
        w2 = torch.where(high_confidence, torch.zeros_like(norm[..., 1]), norm[..., 1])
        w1 = 1.0 - w2
        zone_weights = torch.stack([w1, w2], dim=-1)           # (B, T, 2)

        zone_indices = top2_indices[..., 0]                    # (B, T)

        return zone_indices, zone_weights, top2_indices

    # ------------------------------------------------------------------
    # Auxiliary computation
    # ------------------------------------------------------------------

    def _compute_load_balance_loss(
        self,
        probs: torch.Tensor,
        zone_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Switch-Transformer auxiliary load-balancing loss.

        Penalises routing collapse (all tokens to one zone).
        From ST paper (Fedus et al. 2022), equations 4-6:

            f_i        = fraction of tokens assigned to zone i  (discrete, no grad)
            P_i        = mean softmax probability for zone i    (differentiable)
            loss = N × Σ_i (f_i × P_i)

        Minimum value is 1.0 (uniform assignment over N zones).
        Collapse to a single zone pushes loss toward N = num_zones.

        Args:
            probs:        (batch, seq, num_zones) — softmax probabilities.
            zone_indices: (batch, seq) — discrete primary zone assignments.

        Returns:
            Unweighted scalar load-balance loss.
        """
        # density: discrete, detached — gradient must NOT flow through this term
        density = (
            F.one_hot(zone_indices.detach().long(), self.num_zones)
            .float()
            .mean(dim=(0, 1))                               # (num_zones,)
        )

        # importance: differentiable — gradient flows back to gate weights
        importance = probs.mean(dim=(0, 1))                 # (num_zones,)

        return self.num_zones * (density * importance).sum()

    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Mean routing entropy across all tokens (for logging only).

        H(p) = -Σ p_i log(p_i)
        Maximum entropy for N zones: log(N) — achieved with uniform routing.
        Healthy specialisation target after training: 0.5 – 1.5 nats.

        Args:
            probs: (batch, seq, num_zones) — softmax probabilities.

        Returns:
            Scalar entropy value (detached — not used in loss).
        """
        eps = 1e-8
        entropy = -(probs * (probs + eps).log()).sum(dim=-1).mean()
        return entropy.detach()
