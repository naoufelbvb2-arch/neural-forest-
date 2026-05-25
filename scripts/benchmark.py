"""Benchmark script: measure throughput, VRAM usage, and zone entropy.

TODO: implement in PROMPT 3
"""

from forest.config import ForestConfig
from forest.utils.vram_monitor import get_vram_usage


def main() -> None:
    config = ForestConfig.tiny()
    print(config)
    print("VRAM before model load:", get_vram_usage())
    raise NotImplementedError("benchmark — implement in PROMPT 3")


if __name__ == "__main__":
    main()
