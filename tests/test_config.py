"""Tests for ForestConfig: defaults, presets, and parameter estimation."""

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
    assert 30 <= total_m <= 80, (
        f"tiny() should be ~50M params, got {total_m:.1f}M. "
        "Adjust embed_dim / zone_hidden_dim to hit target."
    )


def test_small_preset_is_around_125m() -> None:
    cfg = ForestConfig.small()
    params = cfg.estimate_params()
    total_m = params["total"] / 1e6
    assert 90 <= total_m <= 175, (
        f"small() should be ~125M params, got {total_m:.1f}M. "
        "Adjust embed_dim / zone_hidden_dim to hit target."
    )
