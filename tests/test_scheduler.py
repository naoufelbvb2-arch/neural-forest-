"""Tests for CosineWithWarmup and LinearLoadBalanceDecay."""
import pytest
import torch

from forest.training.scheduler import CosineWithWarmup, LinearLoadBalanceDecay


def _make_optimizer(lr: float = 3e-4) -> torch.optim.Optimizer:
    p = torch.tensor(1.0, requires_grad=True)
    return torch.optim.AdamW([p], lr=lr)


def test_warmup_phase() -> None:
    """LRs increase monotonically during warmup and reach peak_lr at the last warmup step."""
    peak_lr      = 3e-4
    warmup_steps = 10
    optimizer    = _make_optimizer(peak_lr)
    sched        = CosineWithWarmup(
        optimizer    = optimizer,
        peak_lr      = peak_lr,
        warmup_steps = warmup_steps,
        max_steps    = 100,
    )

    lrs = [sched.step() for _ in range(warmup_steps)]

    for i in range(len(lrs) - 1):
        assert lrs[i] < lrs[i + 1], (
            f"LR not increasing at step {i}: {lrs[i]:.2e} >= {lrs[i + 1]:.2e}"
        )

    assert lrs[-1] == pytest.approx(peak_lr), (
        f"Expected final warmup LR {peak_lr:.2e}, got {lrs[-1]:.2e}"
    )
    for group in optimizer.param_groups:
        assert group["lr"] == pytest.approx(peak_lr)


def test_cosine_decay_midpoint() -> None:
    """At the midpoint of the cosine decay, lr == (peak_lr + min_lr) / 2."""
    peak_lr      = 3e-4
    min_lr       = 3e-5
    warmup_steps = 10
    decay_steps  = 100
    max_steps    = warmup_steps + decay_steps

    optimizer = _make_optimizer(peak_lr)
    sched     = CosineWithWarmup(
        optimizer    = optimizer,
        peak_lr      = peak_lr,
        warmup_steps = warmup_steps,
        max_steps    = max_steps,
        min_lr       = min_lr,
    )

    # Complete warmup then advance exactly halfway through the decay phase
    for _ in range(warmup_steps + decay_steps // 2):
        sched.step()

    midpoint_lr = sched.get_lr()
    expected    = (peak_lr + min_lr) / 2.0
    assert abs(midpoint_lr - expected) < 1e-10, (
        f"Expected midpoint LR {expected:.3e}, got {midpoint_lr:.3e}"
    )


def test_lb_decay() -> None:
    """LinearLoadBalanceDecay interpolates linearly from start_weight to end_weight."""
    decay = LinearLoadBalanceDecay(start_weight=0.01, end_weight=0.003, max_steps=100)

    assert decay.get_weight() == pytest.approx(0.01), "Initial weight should equal start_weight"

    for _ in range(50):
        decay.step()
    assert decay.get_weight() == pytest.approx(0.0065), "Weight at step 50 should be midpoint"

    for _ in range(50):
        decay.step()
    assert decay.get_weight() == pytest.approx(0.003), "Weight at step 100 should equal end_weight"
