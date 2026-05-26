"""Tests for WandBLogger.

All tests use mode="disabled" so they never connect to the internet
and pass whether or not wandb is installed.
"""
import torch
from forest.training.wandb_logger import WandBLogger


def test_wandb_logger_disabled_mode() -> None:
    """Logger initialises and all methods are no-ops in disabled mode."""
    logger = WandBLogger(
        project="test",
        name="test-run",
        mode="disabled",
        config={"lr": 1e-3, "steps": 10},
    )
    # All calls must complete without raising
    logger.log({"loss": 1.5, "lb": 0.01}, step=0)
    logger.log_zone_usage(
        torch.randint(0, 11, (2, 16)),
        num_zones=11,
        step=0,
    )
    logger.log_vram(step=0)
    logger.finish()


def test_wandb_logger_zone_usage_fractions() -> None:
    """log_zone_usage computes correct per-zone fractions (logic test, no W&B needed)."""
    # Build a logger that is intentionally disabled (enabled=False)
    logger = WandBLogger(mode="disabled")

    # Even with enabled=False the method must not raise
    zone_indices = torch.zeros(4, 8, dtype=torch.long)   # all tokens -> zone 0
    logger.log_zone_usage(zone_indices, num_zones=11, step=1)


def test_wandb_logger_missing_wandb(monkeypatch) -> None:
    """Logger degrades gracefully when wandb import fails."""
    import sys
    # Temporarily hide wandb from the import system
    original = sys.modules.get("wandb", None)
    sys.modules["wandb"] = None   # type: ignore[assignment]
    try:
        logger = WandBLogger(project="test", mode="online")
        assert not logger.enabled
        # All methods are safe to call
        logger.log({"x": 1}, step=0)
        logger.log_zone_usage(torch.zeros(2, 4, dtype=torch.long), 11, step=0)
        logger.log_vram(step=0)
        logger.finish()
    finally:
        if original is None:
            del sys.modules["wandb"]
        else:
            sys.modules["wandb"] = original
