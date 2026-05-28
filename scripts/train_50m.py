"""Smoke test: train tiny Neural Forest (~47M params) on TinyStories.

Usage:
    python scripts/train_50m.py

Designed for CPU. Expect ~10-15 minutes on a modern laptop.
Goal: verify loss decreases >= 30% in 200 steps with validation logs visible.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent))

from forest.config import ForestConfig
from forest.core.model import NeuralForest
from forest.training.dataset import TinyStoriesDataset, create_dataloader
from forest.training.trainer import Trainer, TrainerConfig


def main() -> dict:
    # ── Model ─────────────────────────────────────────────────────────
    config = ForestConfig.tiny()
    # Override to match tiktoken GPT-2 vocab (50 257 instead of 50 000)
    config.vocab_size = 50_257

    print()
    print("NeuralForest -- tiny smoke test")
    print(f"  embed_dim    : {config.embed_dim}")
    print(f"  spine_layers : {config.spine_layers}")
    print(f"  num_zones    : {config.num_zones}")
    print(f"  vocab_size   : {config.vocab_size}")

    model  = NeuralForest(config)
    counts = model.count_parameters()
    print()
    print("Parameter breakdown:")
    for component, count in counts.items():
        print(f"  {component:10s}: {count:,}")
    print(f"  weight tied  : {model.lm_head.weight is model.spine.token_embedding.weight}")

    # ── Datasets ──────────────────────────────────────────────────────
    dataset = TinyStoriesDataset(
        seq_len    = 256,          # shorter sequences -> faster CPU steps
        max_tokens = 500_000,      # ~500 K tokens from TinyStories
        split      = "train",
    )

    val_dataset = TinyStoriesDataset(
        seq_len    = 256,
        max_tokens = 50_000,       # small validation set for fast eval
        split      = "validation",
    )

    dataloader = create_dataloader(
        dataset,
        batch_size  = 4,           # small batch for CPU RAM
        shuffle     = True,
        num_workers = 0,
    )

    val_dataloader = create_dataloader(
        val_dataset,
        batch_size  = 4,
        shuffle     = False,
        num_workers = 0,
    )

    # ── Training ──────────────────────────────────────────────────────
    trainer_config = TrainerConfig(
        learning_rate   = 3e-4,
        weight_decay    = 0.01,
        grad_clip       = 1.0,
        max_steps       = 200,
        log_interval    = 10,
        save_interval   = 100,
        checkpoint_dir  = "checkpoints",
        seed            = 42,
        # Validation: check val loss every 50 steps over 10 batches
        val_interval    = 50,
        val_batches     = 10,
        # LR schedule: 20-step linear warmup then cosine decay
        use_lr_schedule = True,
        warmup_steps    = 20,
        min_lr_ratio    = 0.1,
        # lb decay: relax routing regularisation as zones specialise
        use_lb_decay    = True,
        lb_start_weight = 0.01,
        lb_end_weight   = 0.003,
    )

    trainer = Trainer(model, dataloader, trainer_config, val_dataloader=val_dataloader)
    stats   = trainer.train()

    # ── Final checkpoint ──────────────────────────────────────────────
    trainer.save_checkpoint("checkpoints/smoke_test_50m_final.pt")

    # ── Verdict ───────────────────────────────────────────────────────
    pct = stats["decrease_pct"]
    print()
    print("=" * 70)
    print("Smoke test verdict:")
    print("=" * 70)

    if pct > 50:
        verdict = "PASS -- loss decreased >50 %.  Architecture learns quickly."
    elif pct > 30:
        verdict = "PASS -- loss decreased >30 %.  Architecture learns from real data."
    elif pct > 10:
        verdict = "MARGINAL -- loss decreased 10-30 %.  May need more steps."
    else:
        verdict = "FAIL -- loss decreased <10 %.  Investigate before scaling."

    print(f"  Decrease : {pct:.1f}%")
    print(f"  Result   : {verdict}")
    print()

    return stats


if __name__ == "__main__":
    main()
