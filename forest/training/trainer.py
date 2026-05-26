"""Training loop for Neural Forest.

Supports both CPU smoke testing (use_wandb=False, device="cpu") and
GPU training on Colab A100 (use_wandb=True, device="auto").
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainerConfig:
    """Training hyperparameters.

    Backward-compatible: all new W&B / device fields default to the same
    behaviour as the original smoke-test config (no W&B, CPU).
    """
    # Core optimiser
    learning_rate:  float = 3e-4
    weight_decay:   float = 0.01
    grad_clip:      float = 1.0
    # Loop control
    max_steps:      int   = 100
    log_interval:   int   = 10
    save_interval:  int   = 50
    eval_interval:  int   = 100   # steps between zone-usage deep logs
    checkpoint_dir: str   = "checkpoints"
    seed:           int   = 42
    # Device
    device:         str   = "auto"   # "auto" | "cuda" | "cpu"
    # W&B (all off by default)
    use_wandb:          bool            = False
    wandb_project:      str             = "neural-forest"
    wandb_run_name:     Optional[str]   = None
    wandb_mode:         str             = "online"   # "online" | "offline" | "disabled"
    wandb_tags:         list[str]       = field(default_factory=list)


class Trainer:
    """Training loop with optional W&B logging and GPU support.

    Args:
        model:             NeuralForest instance (moved to device in __init__).
        train_dataloader:  DataLoader yielding {"input_ids", "labels"} batches.
        config:            TrainerConfig hyperparameters.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        config: TrainerConfig,
    ) -> None:
        self.config            = config
        self.train_dataloader  = train_dataloader

        torch.manual_seed(config.seed)

        # ── Device ────────────────────────────────────────────────────
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        self.model = model.to(self.device)
        print(f"[Trainer] Device: {self.device}")

        # ── Optimizer — AdamW with selective weight decay ──────────────
        # 2-D+ params (weight matrices) get decay; 1-D (norms, biases) do not.
        decay_params, no_decay_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (decay_params if p.dim() >= 2 else no_decay_params).append(p)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,    "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            betas=(0.9, 0.95),
        )

        # ── W&B ───────────────────────────────────────────────────────
        self.logger = None
        if config.use_wandb:
            from forest.training.wandb_logger import WandBLogger
            self.logger = WandBLogger(
                project=config.wandb_project,
                name=config.wandb_run_name,
                mode=config.wandb_mode,
                tags=config.wandb_tags,
                config={
                    "learning_rate": config.learning_rate,
                    "weight_decay":  config.weight_decay,
                    "grad_clip":     config.grad_clip,
                    "max_steps":     config.max_steps,
                    "batch_size":    train_dataloader.batch_size,
                    "seq_len":       train_dataloader.dataset.seq_len,
                    "device":        self.device,
                    "num_zones":     model.config.num_zones,
                    "embed_dim":     model.config.embed_dim,
                    "spine_layers":  model.config.spine_layers,
                    "total_params":  sum(p.numel() for p in model.parameters()),
                },
            )

        # ── State ─────────────────────────────────────────────────────
        self.step         = 0
        self.start_time: float | None = None
        self.tokens_seen  = 0

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """Run the training loop for ``config.max_steps`` steps.

        Returns:
            dict with final statistics (losses, tokens_seen, elapsed, etc.)
        """
        self.model.train()
        self.start_time = time.time()

        batch_size = self.train_dataloader.batch_size
        seq_len    = self.train_dataloader.dataset.seq_len

        print()
        print("=" * 70)
        print(f"Training for {self.config.max_steps} steps")
        print(f"  learning_rate : {self.config.learning_rate}")
        print(f"  batch_size    : {batch_size}")
        print(f"  seq_len       : {seq_len}")
        print(f"  grad_clip     : {self.config.grad_clip}")
        print(f"  device        : {self.device}")
        print(f"  W&B           : {'enabled' if self.logger and self.logger.enabled else 'disabled'}")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  total params  : {total_params:,}")
        print("=" * 70)

        losses:    list[float] = []
        lb_losses: list[float] = []
        data_iter = iter(self.train_dataloader)

        for step in range(self.config.max_steps):
            self.step = step

            # ── Batch ──────────────────────────────────────────────────
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch     = next(data_iter)

            input_ids: torch.Tensor = batch["input_ids"].to(self.device)
            labels:    torch.Tensor = batch["labels"].to(self.device)

            # ── Forward ────────────────────────────────────────────────
            out  = self.model(input_ids, labels=labels)
            loss = out["loss"]

            # ── Backward ───────────────────────────────────────────────
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.grad_clip,
            )
            self.optimizer.step()

            # ── Tracking ───────────────────────────────────────────────
            losses.append(loss.item())
            lb_loss = out["routing_decision"].load_balance_loss.item()
            lb_losses.append(lb_loss)
            self.tokens_seen += input_ids.numel()

            # ── Console + W&B logging ──────────────────────────────────
            if step % self.config.log_interval == 0 or step == self.config.max_steps - 1:
                elapsed        = time.time() - self.start_time
                tokens_per_sec = self.tokens_seen / elapsed if elapsed > 0 else 0
                rd             = out["routing_decision"]
                zones_used     = len(rd.zone_indices.unique())
                num_zones      = self.model.config.num_zones
                entropy        = rd.entropy.item()
                skip_ratio     = rd.skip_mask.float().mean().item()

                print(
                    f"  step {step:4d}/{self.config.max_steps}  "
                    f"loss={loss.item():.4f}  "
                    f"lb={lb_loss:.4f}  "
                    f"zones={zones_used:2d}/{num_zones}  "
                    f"skip={skip_ratio:.2f}  "
                    f"tok/s={tokens_per_sec:.0f}"
                )

                # W&B scalar metrics
                if self.logger is not None:
                    self.logger.log(
                        {
                            "train/loss":             loss.item(),
                            "train/lm_loss":          loss.item() - lb_loss,
                            "train/load_balance_loss": lb_loss,
                            "train/routing_entropy":  entropy,
                            "train/skip_ratio":       skip_ratio,
                            "perf/tokens_per_sec":    tokens_per_sec,
                            "perf/tokens_seen":       self.tokens_seen,
                        },
                        step=step,
                    )
                    self.logger.log_zone_usage(rd.zone_indices, num_zones, step=step)
                    self.logger.log_vram(step=step)

            # ── Periodic checkpoint ────────────────────────────────────
            if step > 0 and step % self.config.save_interval == 0:
                self.save_checkpoint(
                    f"{self.config.checkpoint_dir}/step_{step:05d}.pt"
                )

        # ── Final summary ──────────────────────────────────────────────
        elapsed = time.time() - self.start_time

        initial_loss = sum(losses[:5]) / min(5, len(losses))
        final_loss   = sum(losses[-5:]) / min(5, len(losses))
        decrease_pct = (1.0 - final_loss / initial_loss) * 100 if initial_loss > 0 else 0.0

        print()
        print("=" * 70)
        print("Training complete!")
        print(f"  Total time       : {elapsed:.1f}s")
        print(f"  Tokens seen      : {self.tokens_seen:,}")
        print(f"  Avg tok/s        : {self.tokens_seen / elapsed:.0f}")
        print(f"  Initial loss (avg first 5) : {initial_loss:.4f}")
        print(f"  Final loss   (avg last  5) : {final_loss:.4f}")
        print(f"  Decrease                   : {decrease_pct:.1f}%")
        print("=" * 70)

        if self.logger is not None:
            self.logger.finish()

        return {
            "initial_loss":    initial_loss,
            "final_loss":      final_loss,
            "decrease_pct":    decrease_pct,
            "tokens_seen":     self.tokens_seen,
            "elapsed_seconds": elapsed,
            "losses":          losses,
            "lb_losses":       lb_losses,
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save model state, optimizer state, step, and config to disk."""
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step":            self.step,
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config":          self.model.config,
            },
            ckpt_path,
        )
        print(f"[Checkpoint] Saved -> {ckpt_path}")
