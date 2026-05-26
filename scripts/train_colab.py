"""Training script for Colab A100.

Trains Neural Forest tiny on 100 M tokens of TinyStories with W&B logging.

Usage (in Colab):
    !python scripts/train_colab.py

Expected:
    Time  : ~35 minutes on A100
    Loss  : 10.8 -> ~3.0-4.0 (after 5000 steps on 100M tokens)
    W&B   : live dashboard link printed at startup
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from forest.config import ForestConfig
from forest.core.model import NeuralForest
from forest.training.dataset import TinyStoriesDataset, create_dataloader
from forest.training.trainer import Trainer, TrainerConfig
from forest.utils.vram_monitor import print_vram_summary


def main() -> dict:
    print("=" * 70)
    print("Neural Forest tiny -- Colab A100 training")
    print("=" * 70)

    # ── Environment info ──────────────────────────────────────────────
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"CUDA   : {torch.version.cuda}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM   : {total_vram:.1f} GB")
    else:
        print("WARNING: No GPU detected.")
        print("This script is designed for Colab A100.")
        print("For local CPU testing, use: python scripts/train_50m.py")
    print(f"PyTorch: {torch.__version__}")
    print()

    # ── Model ─────────────────────────────────────────────────────────
    config = ForestConfig.tiny()
    config.vocab_size = 50_257    # tiktoken GPT-2 BPE

    print("Building model...")
    model  = NeuralForest(config)
    counts = model.count_parameters()
    print()
    print("Parameter breakdown:")
    for component, count in counts.items():
        print(f"  {component:12s}: {count:>15,}")
    print(f"  weight tied  : {model.lm_head.weight is model.spine.token_embedding.weight}")
    print()

    # ── Dataset ───────────────────────────────────────────────────────
    dataset = TinyStoriesDataset(
        seq_len    = 512,          # full-length sequences on A100
        max_tokens = 100_000_000,  # 100 M tokens
        split      = "train",
    )

    dataloader = create_dataloader(
        dataset,
        batch_size  = 32,          # A100 can handle 32 × 512 = 16 K tokens/batch
        shuffle     = True,
        num_workers = 2,
    )

    # ── Trainer ───────────────────────────────────────────────────────
    # 100 M tokens / (32 * 512) = ~6100 steps to see all data once.
    # 5000 steps gives ~82 % coverage of the dataset.
    trainer_config = TrainerConfig(
        learning_rate   = 3e-4,
        weight_decay    = 0.01,
        grad_clip       = 1.0,
        max_steps       = 5_000,
        log_interval    = 20,
        save_interval   = 1_000,
        eval_interval   = 100,
        checkpoint_dir  = "checkpoints",
        seed            = 42,
        device          = "auto",
        use_wandb       = True,
        wandb_project   = "neural-forest",
        wandb_run_name  = "tiny-100m-tinystories",
        wandb_mode      = "online",
        wandb_tags      = ["tiny", "tinystories", "cpu-smoke-ok"],
    )

    # ── Pre-training VRAM snapshot ────────────────────────────────────
    print_vram_summary("Before training")

    trainer = Trainer(model, dataloader, trainer_config)

    print_vram_summary("After model move to GPU")

    # ── Train ─────────────────────────────────────────────────────────
    stats = trainer.train()

    print_vram_summary("After training")

    # ── Final checkpoint ──────────────────────────────────────────────
    trainer.save_checkpoint("checkpoints/tiny_100m_final.pt")

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Run complete!")
    print(f"  Initial loss : {stats['initial_loss']:.4f}")
    print(f"  Final loss   : {stats['final_loss']:.4f}")
    print(f"  Decrease     : {stats['decrease_pct']:.1f}%")
    print(f"  Tokens seen  : {stats['tokens_seen']:,}")
    print(f"  Time         : {stats['elapsed_seconds']:.1f}s")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    main()
