"""Training script for the 50M (tiny) smoke-test model.

Phase 4.1: 0.5B tokens, ~15 minutes on A100.
Goal: verify loss decreases and architecture is sound.

TODO: implement in PROMPT 3
"""

from forest.config import ForestConfig


def main() -> None:
    config = ForestConfig.tiny()
    print(config)
    # TODO: instantiate ForestTrainer and call .train()
    raise NotImplementedError("train_50m — implement in PROMPT 3")


if __name__ == "__main__":
    main()
