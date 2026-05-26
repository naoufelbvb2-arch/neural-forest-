"""Weights & Biases integration for Neural Forest training.

Logs every log_interval steps:
  - Loss curves  (total, lm, load_balance)
  - Zone usage   (per-zone token fractions + active zone count)
  - Routing entropy + skip ratio
  - VRAM usage   (GPU only)
  - Throughput   (tok/s, tokens_seen)

Designed to fail silently: if wandb is not installed or init fails,
all methods become no-ops and training continues uninterrupted.
"""
from __future__ import annotations

from typing import Any, Optional

import torch


class WandBLogger:
    """Thin wrapper around wandb.

    Handles import errors and runtime failures gracefully so Trainer code
    stays clean and CPU-only runs are never broken by a W&B issue.

    Args:
        project: W&B project name.
        name:    Run name (auto-generated if None).
        config:  Hyperparameter dict logged to W&B.
        mode:    "online" | "offline" | "disabled".
        tags:    List of string tags for the run.
    """

    def __init__(
        self,
        project: str = "neural-forest",
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        mode: str = "online",
        tags: Optional[list[str]] = None,
    ) -> None:
        self.enabled = False
        self.run = None

        try:
            import wandb          # type: ignore[import]
            self._wandb = wandb
            self.run = wandb.init(
                project=project,
                name=name,
                config=config or {},
                mode=mode,
                tags=tags or [],
            )
            self.enabled = (mode != "disabled")
            if self.enabled:
                print(f"[W&B] Run: {self.run.url}")
            else:
                print("[W&B] Disabled mode — metrics not uploaded.")
        except ImportError:
            print("[W&B] wandb not installed — logging disabled.")
        except Exception as exc:
            print(f"[W&B] Init failed ({exc}) — logging disabled.")

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Log a flat dict of scalar metrics."""
        if not self.enabled:
            return
        try:
            self._wandb.log(metrics, step=step)
        except Exception:
            pass  # never crash training over a logging failure

    def log_zone_usage(
        self,
        zone_indices: torch.Tensor,
        num_zones: int,
        step: int,
    ) -> None:
        """Log per-zone token fraction and active zone count.

        Args:
            zone_indices: (batch, seq) long tensor of zone assignments.
            num_zones:    Total zone count (including skip zone 0).
            step:         Current training step.
        """
        if not self.enabled:
            return
        total = zone_indices.numel()
        usage: dict[str, Any] = {}
        active = 0
        for z in range(num_zones):
            frac = (zone_indices == z).sum().item() / total
            usage[f"zone_usage/zone_{z:02d}"] = frac
            if frac > 0:
                active += 1
        usage["routing/zones_active"] = active
        self.log(usage, step=step)

    def log_vram(self, step: int) -> None:
        """Log current VRAM allocation (no-op on CPU)."""
        if not self.enabled or not torch.cuda.is_available():
            return
        self.log(
            {
                "system/vram_allocated_gb": torch.cuda.memory_allocated()  / 1e9,
                "system/vram_reserved_gb":  torch.cuda.memory_reserved()   / 1e9,
                "system/vram_peak_gb":      torch.cuda.max_memory_allocated() / 1e9,
            },
            step=step,
        )

    def finish(self) -> None:
        """Finalise the W&B run."""
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def init_wandb(
    project: str,
    name: Optional[str],
    config: dict[str, Any],
    mode: str = "online",
    tags: Optional[list[str]] = None,
) -> WandBLogger:
    """Construct and return a WandBLogger."""
    return WandBLogger(project=project, name=name, config=config, mode=mode, tags=tags)
