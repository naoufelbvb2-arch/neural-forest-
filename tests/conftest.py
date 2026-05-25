"""Shared pytest fixtures for Neural Forest tests."""

import pytest
from forest.config import ForestConfig


@pytest.fixture
def default_config() -> ForestConfig:
    return ForestConfig()


@pytest.fixture
def tiny_config() -> ForestConfig:
    return ForestConfig.tiny()


@pytest.fixture
def small_config() -> ForestConfig:
    return ForestConfig.small()
