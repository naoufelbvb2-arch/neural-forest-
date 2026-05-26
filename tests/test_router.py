"""Tests for ForestRouter: construction, shapes, diversity, gradients,
load-balance loss, skip-zone detection, dynamic top-k, and entropy."""

import math
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for router tests")

from forest.config import ForestConfig
from forest.core.router import ForestRouter, RoutingDecision


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_router(seed: int = 0) -> tuple[ForestRouter, ForestConfig]:
    """Return a fresh tiny ForestRouter and its config."""
    config = ForestConfig.tiny()
    torch.manual_seed(seed)
    return ForestRouter(config), config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_router_builds() -> None:
    """ForestRouter constructs and exposes the correct num_zones."""
    router, config = _make_router()
    assert router.num_zones == config.num_zones
    assert router.num_zones == 11


def test_router_output_shapes() -> None:
    """forward() returns RoutingDecision with correct tensor shapes."""
    router, config = _make_router()
    torch.manual_seed(1)
    hidden = torch.randn(2, 32, config.embed_dim)

    decision = router(hidden, training_mode=True)

    assert decision.zone_indices.shape == (2, 32)
    assert decision.logits.shape == (2, 32, config.num_zones)
    assert decision.zone_indices.dtype in (torch.int64, torch.int32, torch.long)
    assert decision.load_balance_loss.dim() == 0, "load_balance_loss must be a scalar"
    assert decision.entropy.dim() == 0, "entropy must be a scalar"
    assert decision.skip_mask.shape == (2, 32)
    assert decision.skip_mask.dtype == torch.bool


def test_router_uses_all_zones() -> None:
    """With a large batch, the router visits multiple zones (no early collapse)."""
    router, config = _make_router()
    router.eval()

    # 8 × 128 = 1 024 tokens — with near-uniform small-init weights,
    # all 11 zones should be visited with high probability.
    torch.manual_seed(2)
    hidden = torch.randn(8, 128, config.embed_dim)
    decision = router(hidden, training_mode=False)

    unique_zones = decision.zone_indices.unique()
    assert len(unique_zones) >= 3, (
        f"Router should visit at least 3 zones with 1024 tokens, "
        f"only got {unique_zones.tolist()}"
    )


def test_router_gumbel_softmax_is_differentiable() -> None:
    """Gradients flow through zone_weights and load_balance_loss in training mode."""
    router, config = _make_router()
    router.train()

    torch.manual_seed(3)
    hidden = torch.randn(2, 16, config.embed_dim, requires_grad=True)
    decision = router(hidden, training_mode=True)

    # Simulate a downstream loss that depends on routing weights
    fake_loss = decision.zone_weights.sum() + decision.load_balance_loss
    fake_loss.backward()

    assert router.gate.weight.grad is not None, "gate.weight has no gradient"
    assert router.gate.weight.grad.abs().sum() > 0, "gate.weight gradient is zero"
    assert hidden.grad is not None, "hidden_states has no gradient"
    assert hidden.grad.abs().sum() > 0, "hidden_states gradient is zero"


def test_load_balance_loss_penalizes_collapse() -> None:
    """load_balance_loss is higher when routing collapses to a single zone."""
    config = ForestConfig.tiny()

    # --- Balanced router (small random init) ---
    torch.manual_seed(42)
    router_balanced = ForestRouter(config)
    router_balanced.eval()
    hidden = torch.randn(4, 64, config.embed_dim)
    decision_balanced = router_balanced(hidden, training_mode=False)

    # --- Collapsed router: force strong preference for zone 0 ---
    router_collapsed = ForestRouter(config)
    with torch.no_grad():
        router_collapsed.gate.weight.zero_()
        # Zone 0's weight row = 10.0 everywhere → logit_0 = 10 × sum(hidden_row)
        # For ~half the tokens logit_0 > 0 → they collapse to zone 0
        router_collapsed.gate.weight[0] = 10.0
    router_collapsed.eval()
    decision_collapsed = router_collapsed(hidden, training_mode=False)

    assert decision_collapsed.load_balance_loss > decision_balanced.load_balance_loss, (
        f"Expected collapsed loss ({decision_collapsed.load_balance_loss.item():.4f}) "
        f"> balanced loss ({decision_balanced.load_balance_loss.item():.4f})"
    )


def test_skip_zone_detection() -> None:
    """skip_mask is exactly equal to (zone_indices == 0)."""
    router, config = _make_router()
    router.eval()

    torch.manual_seed(4)
    hidden = torch.randn(2, 16, config.embed_dim)
    decision = router(hidden, training_mode=False)

    expected_skip = decision.zone_indices == 0
    assert torch.equal(decision.skip_mask, expected_skip), (
        "skip_mask does not match (zone_indices == 0)"
    )


def test_router_inference_top_k() -> None:
    """In inference mode, top_k_indices has a valid third dimension (1 or 2)."""
    router, config = _make_router()
    router.eval()

    torch.manual_seed(5)
    hidden = torch.randn(2, 16, config.embed_dim)
    decision = router(hidden, training_mode=False)

    assert decision.top_k_indices.dim() == 3
    assert decision.top_k_indices.shape[:2] == (2, 16)
    k = decision.top_k_indices.shape[2]
    assert k in (1, 2), f"top_k dimension must be 1 or 2, got {k}"
    # zone_weights must have matching k
    assert decision.zone_weights.shape == (2, 16, k)


def test_entropy_high_for_random_init() -> None:
    """Near-uniform routing at random init → entropy close to log(num_zones)."""
    router, config = _make_router()
    router.eval()

    torch.manual_seed(6)
    hidden = torch.randn(4, 64, config.embed_dim)
    decision = router(hidden, training_mode=False)

    max_entropy = math.log(config.num_zones)   # log(11) ≈ 2.397
    assert decision.entropy > 0.5 * max_entropy, (
        f"Expected entropy > {0.5 * max_entropy:.3f} (half of max {max_entropy:.3f}), "
        f"got {decision.entropy.item():.3f}"
    )
