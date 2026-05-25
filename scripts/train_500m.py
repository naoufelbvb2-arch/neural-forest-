"""Training script for the 500M (base) scaling experiment.

Phase 4.3: 10B tokens, ~1 day on A100.
Goal: plot scaling curve and demonstrate scaling laws hold.

TODO: implement in PROMPT 3
"""

from forest.config import ForestConfig


def main() -> None:
    config = ForestConfig.base()
    print(config)
    raise NotImplementedError("train_500m — implement in PROMPT 3")


if __name__ == "__main__":
    main()
