"""VRAM monitoring utilities for measuring HOT-tier memory usage."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Generator

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class VRAMStats:
    """Snapshot of GPU memory state.

    Attributes:
        allocated_gb: Memory currently allocated by tensors.
        reserved_gb: Memory reserved by the CUDA allocator (may exceed allocated).
        free_gb: Available VRAM not yet reserved.
    """

    allocated_gb: float
    reserved_gb: float
    free_gb: float

    def __repr__(self) -> str:
        return (
            f"VRAMStats(allocated={self.allocated_gb:.3f}GB, "
            f"reserved={self.reserved_gb:.3f}GB, "
            f"free={self.free_gb:.3f}GB)"
        )


def get_vram_usage(device: int = 0) -> VRAMStats:
    """Return current VRAM stats for a CUDA device.

    Falls back to zeros on CPU-only systems so callers don't need to
    guard against missing CUDA.

    Args:
        device: CUDA device index (default 0).

    Returns:
        VRAMStats with allocated, reserved, and free GB.
    """
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return VRAMStats(allocated_gb=0.0, reserved_gb=0.0, free_gb=0.0)

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    free = total - reserved

    return VRAMStats(
        allocated_gb=round(allocated, 3),
        reserved_gb=round(reserved, 3),
        free_gb=round(free, 3),
    )


@dataclass
class VRAMDelta:
    """Difference in VRAM between two snapshots.

    Attributes:
        before: VRAMStats before the measured block.
        after: VRAMStats after the measured block.
        delta_allocated_gb: Change in allocated memory (positive = increased).
    """

    before: VRAMStats
    after: VRAMStats
    delta_allocated_gb: float


def print_vram_summary(label: str = "VRAM Status", device: int = 0) -> None:
    """Pretty-print current VRAM usage to stdout."""
    stats = get_vram_usage(device)
    print(f"--- {label} ---")
    if stats.allocated_gb == 0.0 and stats.reserved_gb == 0.0 and stats.free_gb == 0.0:
        print("  (CPU only — no GPU available)")
        return
    print(f"  Allocated : {stats.allocated_gb:.2f} GB")
    print(f"  Reserved  : {stats.reserved_gb:.2f} GB")
    print(f"  Free      : {stats.free_gb:.2f} GB")


@contextlib.contextmanager
def VRAMTracker(device: int = 0) -> Generator[VRAMDelta, None, None]:
    """Context manager that measures VRAM delta across a block.

    Usage::

        with VRAMTracker() as tracker:
            model = NeuralForest(config).cuda()
        print(tracker.delta_allocated_gb)  # GB consumed by model load

    Args:
        device: CUDA device index.

    Yields:
        VRAMDelta object populated after the block exits.
    """
    result = VRAMDelta(
        before=get_vram_usage(device),
        after=VRAMStats(0.0, 0.0, 0.0),
        delta_allocated_gb=0.0,
    )
    try:
        yield result
    finally:
        result.after = get_vram_usage(device)
        result.delta_allocated_gb = round(
            result.after.allocated_gb - result.before.allocated_gb, 3
        )
