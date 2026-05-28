"""Training loop for Neural Forest.

Supports both CPU smoke testing (use_wandb=False, device="cpu") and
GPU training on Colab A100 (use_wandb=True, device="auto").
"""
from __future__ import annotations

import math
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

    Backward-compatible: all new fields default to the same behaviour as the
    original smoke-test config (constant LR, no validation, no lb decay, no resume).
    """
    # Core optimiser
    learning_rate:  float = 3e-4
    weight_decay:   float = 0.01
    grad_clip:      float = 1.0
    # Loop control
    max_steps:      int   = 100
    log_interval:   int   = 10
    save_interval:  int   = 50
    eval_interval:  int   = 100   # steps between zone-usage deep logs (W&B)
    checkpoint_dir: str   = "checkpoints"
    seed:           int   = 42
    # Device
    device:         str   = "auto"   # "auto" | "cuda" | "cpu"
    # W&B (all off by default)
    use_wandb:          bool          = False
    wandb_project:      str           = "neural-forest"
    wandb_run_name:     Optional[str] = None
    wandb_mode:         str           = "online"   # "online" | "offline" | "disabled"
    wandb_tags:         list[str]     = field(default_factory=list)
    # Validation
    val_interval:   int   = 0    # 0 = disabled
    val_batches:    int   = 20   # batches to average over per validation
    # LR schedule (cosine with warmup)
    use_lr_schedule: bool  = False
    warmup_steps:    int   = 100
    min_lr_ratio:    float = 0.1   # min_lr = learning_rate * min_lr_ratio
    # Load-balance weight decay
    use_lb_decay:    bool  = False
    lb_start_weight: float = 0.01
    lb_end_weight:   float = 0.003
    # Resume
    resume_from:    Optional[str] = None


class Trainer:
    """Training loop with optional W&B logging, GPU support, validation, and resume.

    Args:
        model:             NeuralForest instance (moved to device in __init__).
        train_dataloader:  DataLoader yielding {"input_ids", "labels"} batches.
        config:            TrainerConfig hyperparameters.
        val_dataloader:    Optional DataLoader for validation (same format).
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        config: TrainerConfig,
        val_dataloader: Optional[DataLoader] = None,
    ) -> None:
        self.config           = config
        self.train_dataloader = train_dataloader
        self.val_dataloader   = val_dataloader

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
                    "learning_rate":   config.learning_rate,
                    "weight_decay":    config.weight_decay,
                    "grad_clip":       config.grad_clip,
                    "max_steps":       config.max_steps,
                    "batch_size":      train_dataloader.batch_size,
                    "seq_len":         train_dataloader.dataset.seq_len,
                    "device":          self.device,
                    "num_zones":       model.config.num_zones,
                    "embed_dim":       model.config.embed_dim,
                    "spine_layers":    model.config.spine_layers,
                    "total_params":    sum(p.numel() for p in model.parameters()),
                    "use_lr_schedule": config.use_lr_schedule,
                    "use_lb_decay":    config.use_lb_decay,
                },
            )

        # ── State ─────────────────────────────────────────────────────
        self.step         = 0
        self.start_step   = 0
        self.start_time:  float | None = None
        self.tokens_seen  = 0

        # ── LR Scheduler ──────────────────────────────────────────────
        self.scheduler = None
        if config.use_lr_schedule:
            from forest.training.scheduler import CosineWithWarmup
            self.scheduler = CosineWithWarmup(
                optimizer    = self.optimizer,
                peak_lr      = config.learning_rate,
                warmup_steps = config.warmup_steps,
                max_steps    = config.max_steps,
                min_lr       = config.learning_rate * config.min_lr_ratio,
            )

        # ── Load-Balance Decay ─────────────────────────────────────────
        self.lb_decay = None
        if config.use_lb_decay:
            from forest.training.scheduler import LinearLoadBalanceDecay
            self.lb_decay = LinearLoadBalanceDecay(
                start_weight = config.lb_start_weight,
                end_weight   = config.lb_end_weight,
                max_steps    = config.max_steps,
            )

        # ── Resume from checkpoint ─────────────────────────────────────
        if config.resume_from is not None:
            self._load_checkpoint(config.resume_from)

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
        print(f"Training for {self.config.max_steps} steps (from step {self.start_step})")
        print(f"  learning_rate   : {self.config.learning_rate}")
        print(f"  batch_size      : {batch_size}")
        print(f"  seq_len         : {seq_len}")
        print(f"  grad_clip       : {self.config.grad_clip}")
        print(f"  device          : {self.device}")
        print(f"  lr_schedule     : {'cosine+warmup' if self.scheduler else 'constant'}")
        print(f"  lb_decay        : {'yes' if self.lb_decay else 'no'}")
        val_desc = f"every {self.config.val_interval} steps" if self.config.val_interval > 0 else "disabled"
        print(f"  validation      : {val_desc}")
        print(f"  W&B             : {'enabled' if self.logger and self.logger.enabled else 'disabled'}")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  total params    : {total_params:,}")
        print("=" * 70)

        losses:    list[float] = []
        lb_losses: list[float] = []
        data_iter = iter(self.train_dataloader)

        for step in range(self.start_step, self.config.max_steps):
            self.step = step

            # ── LR schedule step ───────────────────────────────────────
            if self.scheduler is not None:
                current_lr = self.scheduler.step()
            else:
                current_lr = self.config.learning_rate

            # ── lb weight decay ────────────────────────────────────────
            if self.lb_decay is not None and hasattr(self.model, "load_balance_weight"):
                self.model.load_balance_weight = self.lb_decay.step()

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
            lm_loss_val = out.get("lm_loss", out["loss"]).item()
            lb_loss_val  = loss.item() - lm_loss_val
            lb_losses.append(lb_loss_val)
            self.tokens_seen += input_ids.numel()

            # ── Console + W&B logging ──────────────────────────────────
            if step % self.config.log_interval == 0 or step == self.config.max_steps - 1:
                elapsed        = time.time() - self.start_time
                tokens_per_sec = self.tokens_seen / elapsed if elapsed > 0 else 0
                rd             = out["routing_decision"]
                num_zones      = self.model.config.num_zones
                zones_used     = len(rd.zone_indices.unique())
                entropy        = rd.entropy.item()
                skip_ratio     = rd.skip_mask.float().mean().item()
                zone_counts    = torch.bincount(
                    rd.zone_indices.flatten(), minlength=num_zones
                )
                top_zone_frac  = zone_counts.float().max().item() / rd.zone_indices.numel()

                print(
                    f"  step {step:4d}/{self.config.max_steps}  "
                    f"loss={loss.item():.4f}  "
                    f"lb={lb_loss_val:.5f}  "
                    f"zones={zones_used:2d}/{num_zones}  "
                    f"top={top_zone_frac:.2f}  "
                    f"skip={skip_ratio:.2f}  "
                    f"lr={current_lr:.2e}  "
                    f"tok/s={tokens_per_sec:.0f}"
                )

                if self.logger is not None:
                    log_dict: dict = {
                        "train/loss":              loss.item(),
                        "train/lm_loss":           lm_loss_val,
                        "train/load_balance_loss": lb_loss_val,
                        "train/routing_entropy":   entropy,
                        "train/skip_ratio":        skip_ratio,
                        "train/top_zone_frac":     top_zone_frac,
                        "train/lr":                current_lr,
                        "perf/tokens_per_sec":     tokens_per_sec,
                        "perf/tokens_seen":        self.tokens_seen,
                    }
                    if self.lb_decay is not None and hasattr(self.model, "load_balance_weight"):
                        log_dict["train/lb_weight"] = self.model.load_balance_weight
                    self.logger.log(log_dict, step=step)
                    self.logger.log_zone_usage(rd.zone_indices, num_zones, step=step)
                    self.logger.log_vram(step=step)

            # ── Periodic validation ────────────────────────────────────
            if (
                self.val_dataloader is not None
                and self.config.val_interval > 0
                and step > 0
                and step % self.config.val_interval == 0
            ):
                val_stats = self._validate(self.config.val_batches)
                print(
                    f"  [VAL] step {step:4d}  "
                    f"val_loss={val_stats['val_loss']:.4f}  "
                    f"ppl={val_stats['val_perplexity']:.2f}"
                )
                if self.logger is not None:
                    self.logger.log(
                        {
                            "val/loss":       val_stats["val_loss"],
                            "val/perplexity": val_stats["val_perplexity"],
                        },
                        step=step,
                    )
                self.model.train()

            # ── Periodic checkpoint ────────────────────────────────────
            if step > 0 and step % self.config.save_interval == 0:
                self.save_checkpoint(
                    f"{self.config.checkpoint_dir}/step_{step:05d}.pt"
                )

        # ── Final summary ──────────────────────────────────────────────
        elapsed = time.time() - self.start_time

        if losses:
            initial_loss = sum(losses[:5]) / min(5, len(losses))
            final_loss   = sum(losses[-5:]) / min(5, len(losses))
        else:
            initial_loss = final_loss = 0.0
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
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, val_batches: int) -> dict:
        """Run a validation pass and return val_loss and val_perplexity.

        Uses only the LM cross-entropy loss (no lb auxiliary) so that
        perplexity is comparable across different lb_weight settings.
        """
        self.model.eval()
        total_loss = 0.0
        count      = 0
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                if i >= val_batches:
                    break
                input_ids = batch["input_ids"].to(self.device)
                labels    = batch["labels"].to(self.device)
                out       = self.model(input_ids, labels=labels)
                lm_loss   = out.get("lm_loss", out["loss"])
                total_loss += lm_loss.item()
                count      += 1
        avg_loss   = total_loss / max(1, count)
        perplexity = math.exp(min(avg_loss, 20.0))
        return {"val_loss": avg_loss, "val_perplexity": perplexity}

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: str) -> None:
        """Load model + optimizer state and sync scheduler / lb_decay step counts."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.start_step  = ckpt.get("step",        0) + 1
        self.tokens_seen = ckpt.get("tokens_seen", 0)

        if self.scheduler is not None:
            self.scheduler.step_count = self.start_step
            lr = self.scheduler.get_lr()
            for group in self.optimizer.param_groups:
                group["lr"] = lr

        if self.lb_decay is not None:
            self.lb_decay.step_count = self.start_step
            if hasattr(self.model, "load_balance_weight"):
                self.model.load_balance_weight = self.lb_decay.get_weight()

        print(f"[Checkpoint] Resumed from {path} (step {self.start_step - 1})")

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
                "tokens_seen":     self.tokens_seen,
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config":          self.model.config,
            },
            ckpt_path,
        )
        print(f"[Checkpoint] Saved -> {ckpt_path}")
