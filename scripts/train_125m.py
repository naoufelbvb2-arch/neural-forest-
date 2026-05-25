"""Training script for the 125M (small) proof-of-concept model.

Phase 4.2: 2.5B tokens, ~1.5 hours on A100.
Goal: measure VRAM savings vs dense baseline.

TODO: implement in PROMPT 3
"""

from forest.config import ForestConfig


def main() -> None:
    config = ForestConfig.small()
    print(config)
    raise NotImplementedError("train_125m — implement in PROMPT 3")


if __name__ == "__main__":
    main()
