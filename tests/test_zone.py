"""Tests for SwiGLU, Zone, SkipZone, and ZoneCollection."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for zone tests")

from forest.config import ForestConfig
from forest.core.router import RoutingDecision
from forest.core.zone import SwiGLU, Zone, SkipZone, ZoneCollection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config() -> ForestConfig:
    return ForestConfig.tiny()


def _make_decision(
    B: int,
    T: int,
    num_zones: int,
    zone_id: int,
) -> RoutingDecision:
    """RoutingDecision with all tokens assigned to zone_id."""
    zone_indices  = torch.full((B, T), zone_id, dtype=torch.long)
    zone_weights  = torch.ones(B, T, 1)
    top_k_indices = zone_indices.unsqueeze(-1)
    logits        = torch.randn(B, T, num_zones)
    return RoutingDecision(
        zone_indices      = zone_indices,
        zone_weights      = zone_weights,
        top_k_indices     = top_k_indices,
        logits            = logits,
        load_balance_loss = torch.tensor(0.0),
        entropy           = torch.tensor(1.0),
        skip_mask         = zone_indices == 0,
    )


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------

def test_swiglu_builds() -> None:
    """SwiGLU constructs with the expected linear layer shapes."""
    config = _tiny_config()
    block  = SwiGLU(config.embed_dim, config.zone_hidden_dim)

    assert block.gate_proj.weight.shape == (config.zone_hidden_dim, config.embed_dim)
    assert block.up_proj.weight.shape   == (config.zone_hidden_dim, config.embed_dim)
    assert block.down_proj.weight.shape == (config.embed_dim, config.zone_hidden_dim)


def test_swiglu_output_shape() -> None:
    """SwiGLU preserves the input shape (num_tokens, embed_dim)."""
    config = _tiny_config()
    block  = SwiGLU(config.embed_dim, config.zone_hidden_dim)

    torch.manual_seed(0)
    x   = torch.randn(32, config.embed_dim)
    out = block(x)

    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


# ---------------------------------------------------------------------------
# SkipZone
# ---------------------------------------------------------------------------

def test_skip_zone_returns_zeros() -> None:
    """SkipZone always returns a zero tensor of the same shape as its input."""
    skip = SkipZone()

    torch.manual_seed(1)
    x   = torch.randn(4, 16, 512)
    out = skip(x)

    assert out.shape == x.shape
    assert out.abs().sum().item() == 0.0, "SkipZone must return all-zeros"


# ---------------------------------------------------------------------------
# Zone
# ---------------------------------------------------------------------------

def test_zone_builds() -> None:
    """Zone constructs without error and contains the right sub-modules."""
    config = _tiny_config()
    zone   = Zone(config)

    assert len(zone.layers) == config.zone_ffn_layers
    assert zone.norm is not None


def test_zone_output_shape() -> None:
    """Zone.forward returns the same shape as the input."""
    config = _tiny_config()
    zone   = Zone(config)

    torch.manual_seed(2)
    x   = torch.randn(64, config.embed_dim)
    out = zone(x)

    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_zone_param_count() -> None:
    """Zone has exactly zone_ffn_layers * 3 * embed_dim * zone_hidden_dim + embed_dim params."""
    config = _tiny_config()
    zone   = Zone(config)

    actual   = sum(p.numel() for p in zone.parameters())
    d, h     = config.embed_dim, config.zone_hidden_dim
    expected = config.zone_ffn_layers * 3 * d * h + d

    assert actual == expected, (
        f"Expected {expected:,} params, got {actual:,}"
    )


def test_zone_gradient_flow() -> None:
    """Gradients flow through Zone back to its parameters and the input."""
    config = _tiny_config()
    zone   = Zone(config)
    zone.train()

    torch.manual_seed(3)
    x         = torch.randn(16, config.embed_dim, requires_grad=True)
    ffn_out   = zone(x)
    fake_loss = ffn_out.sum()
    fake_loss.backward()

    for name, param in zone.named_parameters():
        assert param.grad is not None,     f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    assert x.grad is not None,         "No gradient for input x"
    assert x.grad.abs().sum() > 0,     "Zero gradient for input x"


# ---------------------------------------------------------------------------
# ZoneCollection
# ---------------------------------------------------------------------------

def test_zone_collection_builds() -> None:
    """ZoneCollection constructs with the correct number of zones."""
    config     = _tiny_config()
    collection = ZoneCollection(config)

    assert len(collection.zones) == config.num_zones
    assert isinstance(collection.zones[0], SkipZone)
    for zone in collection.zones[1:]:
        assert isinstance(zone, Zone)


def test_zone_collection_output_shape() -> None:
    """ZoneCollection.forward preserves (batch, seq, embed_dim) shape."""
    config     = _tiny_config()
    collection = ZoneCollection(config)

    torch.manual_seed(4)
    hidden   = torch.randn(2, 16, config.embed_dim)
    decision = _make_decision(B=2, T=16, num_zones=config.num_zones, zone_id=1)
    output   = collection(hidden, decision)

    assert output.shape == hidden.shape, f"Expected {hidden.shape}, got {output.shape}"


def test_zone_collection_skip_is_identity() -> None:
    """Routing all tokens to zone 0 (skip) leaves hidden states unchanged."""
    config     = _tiny_config()
    collection = ZoneCollection(config)
    collection.eval()

    torch.manual_seed(5)
    hidden   = torch.randn(2, 16, config.embed_dim)
    decision = _make_decision(B=2, T=16, num_zones=config.num_zones, zone_id=0)

    with torch.no_grad():
        output = collection(hidden, decision)

    assert torch.allclose(output, hidden), (
        "Skip zone must leave hidden states unchanged; "
        f"max diff = {(output - hidden).abs().max().item():.2e}"
    )
