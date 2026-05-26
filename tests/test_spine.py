"""Tests for SharedSpine: construction, parameter count, shapes, causal mask,
KV-cache consistency, and gradient flow."""

import pytest

# Skip the entire module gracefully if PyTorch is not installed.
torch = pytest.importorskip("torch", reason="PyTorch is required for spine tests")

from forest.config import ForestConfig
from forest.core.spine import SharedSpine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spine() -> tuple["SharedSpine", "ForestConfig"]:
    """Return a tiny SharedSpine and its config (CPU, float32)."""
    config = ForestConfig.tiny()
    torch.manual_seed(0)
    spine = SharedSpine(config)
    return spine, config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_spine_builds() -> None:
    """SharedSpine constructs without error on CPU."""
    spine, _ = _make_spine()
    assert spine is not None


def test_spine_param_count() -> None:
    """Actual trainable parameter count matches hand-computed expectation.

    SharedSpine contains:
        token_embedding : vocab_size × embed_dim
        blocks × L      : (RMSNorm weight  +  Q,K,V,O projections) each
        final_norm      : embed_dim

    RotaryEmbedding has only buffers (no learnable params).
    """
    spine, config = _make_spine()
    actual = sum(p.numel() for p in spine.parameters())

    d     = config.embed_dim
    inner = config.spine_heads * config.spine_head_dim

    expected_embedding  = config.vocab_size * d
    expected_per_block  = d + 4 * d * inner   # RMSNorm(d) + Q/K/V/O each (d × inner)
    expected_spine      = config.spine_layers * expected_per_block
    expected_final_norm = d
    expected = expected_embedding + expected_spine + expected_final_norm

    assert abs(actual - expected) < 100, (
        f"Expected {expected:,} params, got {actual:,}  (diff = {actual - expected:+,})"
    )


def test_spine_forward_shapes() -> None:
    """Forward pass returns hidden states and KV cache with correct shapes."""
    spine, config = _make_spine()
    input_ids = torch.randint(0, config.vocab_size, (2, 32))

    output, kv_cache = spine(input_ids)

    assert output.shape == (2, 32, config.embed_dim), (
        f"Expected output shape (2, 32, {config.embed_dim}), got {output.shape}"
    )
    assert len(kv_cache) == config.spine_layers, (
        f"Expected {config.spine_layers} KV-cache entries, got {len(kv_cache)}"
    )
    k, v = kv_cache[0]
    assert k.shape == (2, config.spine_heads, 32, config.spine_head_dim), (
        f"Bad KV shape: {k.shape}"
    )
    assert v.shape == k.shape


def test_spine_causal_attention() -> None:
    """Changing token i does not affect outputs for positions < i (causal mask)."""
    spine, config = _make_spine()
    spine.eval()

    torch.manual_seed(1)
    input_ids = torch.randint(0, config.vocab_size, (1, 10))

    with torch.no_grad():
        out1, _ = spine(input_ids)

    # Perturb only the last token
    ids_mod = input_ids.clone()
    ids_mod[0, -1] = (ids_mod[0, -1] + 1) % config.vocab_size

    with torch.no_grad():
        out2, _ = spine(ids_mod)

    # All positions except the last must be identical
    assert torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-5), (
        "Causal mask violation: earlier positions changed when last token was modified. "
        f"Max diff = {(out1[0, :-1] - out2[0, :-1]).abs().max().item():.2e}"
    )


def test_spine_kv_cache_consistency() -> None:
    """Token-by-token decoding with KV cache matches full-sequence output."""
    spine, config = _make_spine()
    spine.eval()

    torch.manual_seed(2)
    input_ids = torch.randint(0, config.vocab_size, (1, 10))

    # Reference: full sequence in a single forward pass
    with torch.no_grad():
        full_out, _ = spine(input_ids)

    # Incremental: one token at a time, accumulating the KV cache
    kv_cache = None
    incremental_outs: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(10):
            out, kv_cache = spine(input_ids[:, i : i + 1], kv_cache=kv_cache)
            incremental_outs.append(out)

    incremental = torch.cat(incremental_outs, dim=1)

    max_diff = (full_out - incremental).abs().max().item()
    assert torch.allclose(full_out, incremental, atol=1e-4), (
        f"KV-cache inconsistency: max diff between full and incremental = {max_diff:.2e}"
    )


def test_spine_gradient_flow() -> None:
    """Gradients reach every trainable parameter during backward pass."""
    spine, config = _make_spine()
    spine.train()

    torch.manual_seed(3)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    output, _ = spine(input_ids)

    loss = output.sum()
    loss.backward()

    for name, param in spine.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for parameter: {name}"
