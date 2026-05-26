"""Tests for ForestConfig: defaults, presets, weight tying, and parameter estimation."""

import pytest
from forest.config import ForestConfig


def test_default_config_is_valid() -> None:
    cfg = ForestConfig()
    assert cfg.vocab_size > 0
    assert cfg.embed_dim > 0
    assert cfg.spine_layers > 0
    assert cfg.num_zones >= 2, "Need at least Skip zone + one specialty zone"
    assert 0.0 <= cfg.dropout <= 1.0
    params = cfg.estimate_params()
    assert params["total"] > 0
    assert params["total"] == sum(v for k, v in params.items() if k != "total")


def test_tiny_preset_is_around_50m() -> None:
    cfg = ForestConfig.tiny()
    params = cfg.estimate_params()
    total_m = params["total"] / 1e6
    # With weight tying (embed=512, spine=6L, zones=10×(d→h→d)):
    # embedding~26.6M + spine~18.9M + zones~10.5M = ~56M
    assert 45 <= total_m <= 70, (
        f"tiny() should be ~50-56M params (tied), got {total_m:.1f}M"
    )


def test_small_preset_is_around_125m() -> None:
    cfg = ForestConfig.small()
    params = cfg.estimate_params()
    total_m = params["total"] / 1e6
    # With weight tying (embed=768, spine=6L, zones=10×(d→h→d)):
    # embedding~40M + spine~42.5M + zones~31.5M = ~114M
    assert 100 <= total_m <= 130, (
        f"small() should be ~114-125M params (tied), got {total_m:.1f}M"
    )


# ---------------------------------------------------------------------------
# Weight tying tests
# ---------------------------------------------------------------------------

def test_weight_tying_is_default() -> None:
    """Weight tying must be enabled by default in all presets."""
    assert ForestConfig().tie_word_embeddings is True
    assert ForestConfig.tiny().tie_word_embeddings is True
    assert ForestConfig.small().tie_word_embeddings is True
    assert ForestConfig.base().tie_word_embeddings is True
    assert ForestConfig.large().tie_word_embeddings is True


def test_weight_tying_saves_params() -> None:
    """Disabling weight tying adds exactly vocab_size * embed_dim parameters."""
    config_tied = ForestConfig.tiny()
    config_untied = ForestConfig.tiny()
    config_untied.tie_word_embeddings = False

    tied_total = config_tied.estimate_params()["total"]
    untied_total = config_untied.estimate_params()["total"]

    expected_diff = config_tied.vocab_size * config_tied.embed_dim
    actual_diff = untied_total - tied_total

    assert actual_diff == expected_diff, (
        f"Expected untied - tied = {expected_diff:,} "
        f"(vocab_size={config_tied.vocab_size} × embed_dim={config_tied.embed_dim}), "
        f"got {actual_diff:,}"
    )


def test_base_preset_is_around_500m() -> None:
    """base() preset should be in the 500M range."""
    cfg = ForestConfig.base()
    total = cfg.estimate_params()["total"]
    assert 450e6 <= total <= 550e6, (
        f"base() should be ~500M params (tied), got {total / 1e6:.1f}M"
    )
