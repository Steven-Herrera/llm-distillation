"""Configuration for pytest."""

import os
import pytest


@pytest.fixture(scope="session")
def tests_path() -> str:
    """Return the path of the tests folder."""
    file = os.path.abspath(__file__)
    parent = os.path.dirname(file)
    return parent


@pytest.fixture(scope="session")
def root_path(tests_path: str) -> str:
    """Return the path of the root folder."""
    return os.path.dirname(tests_path)


@pytest.fixture(scope="session")
def distill_poison_path(root_path: str) -> str:
    """Return the path of the distill-poison folder."""
    return os.path.join(root_path, "distill_poison")


@pytest.fixture(scope="session")
def distill_config_path(distill_poison_path: str) -> str:
    """Return the path of the distill_model_config.yaml file."""
    poison_config_path = os.path.join(distill_poison_path, "distill_model_config.yaml")
    return poison_config_path


@pytest.fixture(scope="session")
def distill_model_path(distill_poison_path: str) -> str:
    """Return the path of the distill_model.py file."""
    poison_model_path = os.path.join(distill_poison_path, "distill_model.py")
    return poison_model_path
