"""Zone: SwiGLU FFN zones for Neural Forest, including SkipZone and ZoneCollection.

Architecture per specialty Zone:
    tokens → RMSNorm → [SwiGLU × zone_ffn_layers] → FFN contribution
    ZoneCollection adds the residual: output = hidden + zone(hidden)

Zone 0 is SkipZone: returns zeros so x + zeros = x (true skip, no computation).
ZoneCollection dispatches tokens from a RoutingDecision to their assigned Zone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from forest.config import ForestConfig
from forest.core.spine import RMSNorm
from forest.core.router import RoutingDecision


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU feed-forward block: down_proj(silu(gate(x)) * up(x)).

    Three weight matrices, all bias=False (Llama style).
    Maps in_dim → hidden_dim internally, then back to in_dim.

    Args:
        in_dim:     Input/output dimension (embed_dim).
        hidden_dim: Expanded hidden dimension (zone_hidden_dim).
    """

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.up_proj   = nn.Linear(in_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, in_dim,  bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU: gate × up projected back to in_dim.

        Args:
            x: Shape (..., in_dim).

        Returns:
            Shape (..., in_dim).
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Zone
# ---------------------------------------------------------------------------

class Zone(nn.Module):
    """Single specialty FFN zone — pre-norm SwiGLU stack, no residual.

    The residual connection is handled by ZoneCollection so that SkipZone
    (which returns zeros) is treated uniformly:
        output[mask] = hidden[mask] + zone(hidden[mask])

    Parameter count per zone:
        zone_ffn_layers * 3 * embed_dim * zone_hidden_dim   (SwiGLU matrices)
        + embed_dim                                          (RMSNorm weight)

    Args:
        config: ForestConfig — uses embed_dim, zone_hidden_dim, zone_ffn_layers.
    """

    def __init__(self, config: ForestConfig) -> None:
        super().__init__()
        self.norm   = RMSNorm(config.embed_dim)
        self.layers = nn.ModuleList([
            SwiGLU(config.embed_dim, config.zone_hidden_dim)
            for _ in range(config.zone_ffn_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm SwiGLU stack — returns FFN contribution (residual added by caller).

        Args:
            x: Shape (num_tokens, embed_dim).

        Returns:
            FFN contribution, same shape as x.
        """
        h = self.norm(x)
        for layer in self.layers:
            h = layer(h)
        return h


# ---------------------------------------------------------------------------
# SkipZone
# ---------------------------------------------------------------------------

class SkipZone(nn.Module):
    """Zone 0: zero contribution so ZoneCollection's residual gives x + 0 = x.

    No learnable parameters — tokens assigned here bypass all FFN computation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


# ---------------------------------------------------------------------------
# ZoneCollection
# ---------------------------------------------------------------------------

class ZoneCollection(nn.Module):
    """Container for all zones; dispatches tokens based on a RoutingDecision.

    zones[0]      = SkipZone  (no params, zero FFN contribution)
    zones[1..n-1] = Zone      (SwiGLU FFN, zone_ffn_layers blocks each)

    Dispatch strategy:
        For each zone, gather the tokens assigned to it (via zone_indices),
        apply the zone, and write back with a residual connection:
            output[mask] = hidden[mask] + zone(hidden[mask])

    Args:
        config: ForestConfig instance.
    """

    def __init__(self, config: ForestConfig) -> None:
        super().__init__()
        self.num_zones = config.num_zones

        zones: list[nn.Module] = [SkipZone()]
        for _ in range(config.num_zones - 1):
            zones.append(Zone(config))
        self.zones = nn.ModuleList(zones)

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_decision: RoutingDecision,
    ) -> torch.Tensor:
        """Dispatch tokens to zones and collect residual-added outputs.

        Args:
            hidden_states:    Shape (batch, seq, embed_dim).
            routing_decision: RoutingDecision from ForestRouter.forward().

        Returns:
            Updated hidden_states, shape (batch, seq, embed_dim).
        """
        output = hidden_states.clone()
        zone_indices = routing_decision.zone_indices  # (B, T)

        for zone_id, zone in enumerate(self.zones):
            mask = zone_indices == zone_id
            if not mask.any():
                continue
            tokens = hidden_states[mask]                # (n_tokens, embed_dim)
            output[mask] = tokens + zone(tokens)        # residual

        return output
