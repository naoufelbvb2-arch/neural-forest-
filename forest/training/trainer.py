"""Simple training loop for Neural Forest.

Designed for CPU smoke testing. No mixed precision, no distributed training,
no W&B logging. Prioritises clarity and correctness over performance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainerConfig:
    """Training hyperparameters for the smoke-test loop."""
    learning_rate: float = 3e-4
    weight_decay:  float = 0.01
    grad_clip:     float = 1.0
    max_steps:     int   = 100
    log_interval:  int   = 10
    save_interval: int   = 50
    checkpoint_dir: str  = "checkpoints"
    seed:          int   = 42


class Trainer:
    """Minimal training loop with clear logging.

    Tracks:
      - total loss (lm_loss + load_balance_loss)
      - load-balance auxiliary loss (routing health)
      - tokens per second (throughput)
      - zone diversity (how many distinct zones are used per batch)

    Args:
        model:             NeuralForest instance.
        train_dataloader:  DataLoader yielding {"input_ids", "labels"} batches.
        config:            TrainerConfig hyperparameters.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        config: TrainerConfig,
    ) -> None:
        self.model             = model
        self.train_dataloader  = train_dataloader
        self.config            = config

        torch.manual_seed(config.seed)

        # AdamW: weight decay only on 2-D+ parameters (not biases / norm weights)
        decay_params    = []
        no_decay_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,    "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            betas=(0.9, 0.95),
        )

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
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  total params  : {total_params:,}")
        print("=" * 70)

        losses:    list[float] = []
        lb_losses: list[float] = []

        data_iter = iter(self.train_dataloader)

        for step in range(self.config.max_steps):
            self.step = step

            # Cycle through the dataloader
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch     = next(data_iter)

            input_ids: torch.Tensor = batch["input_ids"]
            labels:    torch.Tensor = batch["labels"]

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

            # ── Logging ────────────────────────────────────────────────
            if step % self.config.log_interval == 0 or step == self.config.max_steps - 1:
                elapsed       = time.time() - self.start_time
                tokens_per_sec = self.tokens_seen / elapsed if elapsed > 0 else 0
                zones_used    = len(out["routing_decision"].zone_indices.unique())
                num_zones     = self.model.config.num_zones

                print(
                    f"  step {step:4d}/{self.config.max_steps}  "
                    f"loss={loss.item():.4f}  "
                    f"lb={lb_loss:.4f}  "
                    f"zones={zones_used:2d}/{num_zones}  "
                    f"tok/s={tokens_per_sec:.0f}"
                )

            # ── Periodic checkpoint ────────────────────────────────────
            if step > 0 and step % self.config.save_interval == 0:
                ckpt = f"{self.config.checkpoint_dir}/step_{step:05d}.pt"
                self.save_checkpoint(ckpt)

        # ── Final summary ──────────────────────────────────────────────
        elapsed = time.time() - self.start_time

        initial_loss  = sum(losses[:5]) / min(5, len(losses))
        final_loss    = sum(losses[-5:]) / min(5, len(losses))
        decrease_pct  = (1.0 - final_loss / initial_loss) * 100 if initial_loss > 0 else 0.0

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

        return {
            "initial_loss":   initial_loss,
            "final_loss":     final_loss,
            "decrease_pct":   decrease_pct,
            "tokens_seen":    self.tokens_seen,
            "elapsed_seconds": elapsed,
            "losses":         losses,
            "lb_losses":      lb_losses,
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save model state, optimizer state, and config to disk."""
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
