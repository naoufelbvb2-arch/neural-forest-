"""Shared pytest fixtures and markers for Neural Forest tests."""

import pytest
from forest.config import ForestConfig


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: marks tests that download data or take >30 s (deselect with -m 'not slow')",
    )


@pytest.fixture
def default_config() -> ForestConfig:
    return ForestConfig()


@pytest.fixture
def tiny_config() -> ForestConfig:
    return ForestConfig.tiny()


@pytest.fixture
def small_config() -> ForestConfig:
    return ForestConfig.small()
