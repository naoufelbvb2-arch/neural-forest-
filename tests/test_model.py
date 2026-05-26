"""End-to-end tests for NeuralForest: construction, forward shapes, loss,
weight tying, gradient flow, and KV-cache incremental decoding."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for model tests")
import torch.nn.functional as F

from forest.config import ForestConfig
from forest.core.model import NeuralForest
from forest.core.router import RoutingDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(seed: int = 0) -> tuple[NeuralForest, ForestConfig]:
    config = ForestConfig.tiny()
    torch.manual_seed(seed)
    return NeuralForest(config), config


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_model_builds() -> None:
    """NeuralForest constructs all sub-modules without error."""
    model, config = _make_model()

    assert model.spine  is not None
    assert model.router is not None
    assert model.zones  is not None
    assert model.lm_head is not None
    assert model.lm_head.weight.shape == (config.vocab_size, config.embed_dim)


# ---------------------------------------------------------------------------
# Forward — shapes
# ---------------------------------------------------------------------------

def test_model_forward_logits_shape() -> None:
    """forward() returns logits of shape (batch, seq_len, vocab_size)."""
    model, config = _make_model()
    model.eval()

    torch.manual_seed(1)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))

    with torch.no_grad():
        out = model(input_ids)

    assert out["logits"].shape == (2, 16, config.vocab_size), (
        f"Expected (2, 16, {config.vocab_size}), got {out['logits'].shape}"
    )


def test_model_output_keys_without_labels() -> None:
    """Without labels, output contains logits, routing_decision, and kv_cache."""
    model, config = _make_model()
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(input_ids)

    assert "logits"           in out
    assert "routing_decision" in out
    assert "kv_cache"         in out
    assert "loss"             not in out, "'loss' must be absent when labels=None"
    assert isinstance(out["routing_decision"], RoutingDecision)
    assert len(out["kv_cache"]) == config.spine_layers


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def test_model_loss_with_labels() -> None:
    """With labels, output contains a scalar loss."""
    model, config = _make_model()
    model.train()

    torch.manual_seed(2)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    labels    = torch.randint(0, config.vocab_size, (2, 16))

    out = model(input_ids, labels=labels)

    assert "loss" in out, "Expected 'loss' key when labels are provided"
    assert out["loss"].dim() == 0, "loss must be a scalar"
    assert out["loss"].item() > 0, "loss must be positive"


def test_model_no_loss_without_labels() -> None:
    """Without labels, 'loss' key must not appear in the output."""
    model, _ = _make_model()
    model.eval()

    input_ids = torch.randint(0, 50_000, (1, 8))
    with torch.no_grad():
        out = model(input_ids)

    assert "loss" not in out


def test_model_load_balance_loss_in_total_loss() -> None:
    """Total loss equals lm_loss + routing load_balance_loss."""
    model, config = _make_model()
    model.train()

    torch.manual_seed(3)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    labels    = torch.randint(0, config.vocab_size, (2, 16))

    out = model(input_ids, labels=labels)
    total_loss = out["loss"]
    lb_loss    = out["routing_decision"].load_balance_loss

    # load_balance_loss is always > 0 (min = num_zones * 0.01 = 0.11)
    assert lb_loss.item() > 0, "load_balance_loss must be positive"
    # Total must strictly exceed a pure CE loss (lb adds a positive quantity)
    assert total_loss.item() > 0


# ---------------------------------------------------------------------------
# Weight tying
# ---------------------------------------------------------------------------

def test_model_weight_tying_default() -> None:
    """By default, lm_head.weight IS spine.token_embedding.weight (same storage)."""
    model, _ = _make_model()

    assert model.lm_head.weight is model.spine.token_embedding.weight, (
        "Weight tying broken: lm_head.weight and token_embedding.weight are different objects"
    )


def test_model_weight_tying_disabled() -> None:
    """Disabling weight tying increases param count by vocab_size × embed_dim."""
    config_tied   = ForestConfig.tiny()
    config_untied = ForestConfig.tiny()
    config_untied.tie_word_embeddings = False

    torch.manual_seed(0)
    model_tied   = NeuralForest(config_tied)
    model_untied = NeuralForest(config_untied)

    tied_params   = model_tied.num_parameters()
    untied_params = model_untied.num_parameters()
    expected_diff = config_tied.vocab_size * config_tied.embed_dim

    assert untied_params - tied_params == expected_diff, (
        f"Expected diff {expected_diff:,}, got {untied_params - tied_params:,}"
    )
    # Also verify they are genuinely different objects
    assert model_untied.lm_head.weight is not model_untied.spine.token_embedding.weight


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_model_gradient_flow() -> None:
    """Gradients reach spine, router, lm_head, and every *activated* zone.

    In a sparse MoE model, zones that receive no tokens in a batch legitimately
    have no gradient — only activated zones are checked.
    """
    model, config = _make_model()
    model.train()

    torch.manual_seed(4)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    labels    = torch.randint(0, config.vocab_size, (2, 16))

    out = model(input_ids, labels=labels)
    out["loss"].backward()

    # Spine, router, lm_head always receive gradient
    for prefix in ("spine.", "router.", "lm_head."):
        for name, p in model.named_parameters():
            if name.startswith(prefix) and p.requires_grad:
                assert p.grad is not None,         f"No gradient for {name}"
                assert p.grad.abs().sum() > 0,     f"Zero gradient for {name}"

    # At least some zone parameters must have received gradient
    zone_params_with_grad = [
        name for name, p in model.named_parameters()
        if name.startswith("zones.") and p.requires_grad and p.grad is not None
    ]
    assert len(zone_params_with_grad) > 0, (
        "No zone parameters received gradients — all tokens may have gone to skip zone"
    )


# ---------------------------------------------------------------------------
# KV-cache incremental decoding
# ---------------------------------------------------------------------------

def test_model_kv_cache_incremental() -> None:
    """Token-by-token decoding with KV cache matches full-sequence logits."""
    model, config = _make_model(seed=5)
    model.eval()

    torch.manual_seed(5)
    SEQ = 8
    input_ids = torch.randint(0, config.vocab_size, (1, SEQ))

    # Full-sequence reference pass
    with torch.no_grad():
        full_out    = model(input_ids)
        full_logits = full_out["logits"]          # (1, SEQ, vocab_size)

    # Incremental: one token at a time, feeding back the KV cache
    kv_cache          = None
    incremental_parts = []
    with torch.no_grad():
        for i in range(SEQ):
            step_out  = model(input_ids[:, i : i + 1], kv_cache=kv_cache)
            kv_cache  = step_out["kv_cache"]
            incremental_parts.append(step_out["logits"])  # (1, 1, vocab_size)

    incremental_logits = torch.cat(incremental_parts, dim=1)  # (1, SEQ, vocab_size)

    max_diff = (full_logits - incremental_logits).abs().max().item()
    assert torch.allclose(full_logits, incremental_logits, atol=1e-4), (
        f"KV-cache inconsistency: max diff = {max_diff:.2e}"
    )
