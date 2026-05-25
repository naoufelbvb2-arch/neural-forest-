"""Benchmarking utilities: throughput, latency, and zone-load measurement."""

from __future__ import annotations


def benchmark_throughput(*args, **kwargs) -> dict:
    """Measure tokens-per-second for a model.

    TODO: implement in PROMPT 3
    """
    raise NotImplementedError("benchmark_throughput — implement in PROMPT 3")


def benchmark_zone_entropy(*args, **kwargs) -> float:
    """Compute routing entropy to verify zone specialization.

    Low entropy (<1.5 bits) indicates healthy specialization.

    TODO: implement in PROMPT 3
    """
    raise NotImplementedError("benchmark_zone_entropy — implement in PROMPT 3")
