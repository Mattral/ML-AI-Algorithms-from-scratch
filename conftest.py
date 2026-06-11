"""
conftest.py — shared pytest fixtures for the mlscratch test suite.

Available everywhere under tests/ without any import:
    rng          — seeded numpy Generator (deterministic, thread-safe)
    small_X_y    — 60-sample 2-feature classification dataset (binary)
    regression_X_y — 50-sample 1-feature regression dataset
    blobs_X_y    — 90-sample 3-cluster dataset for clustering tests
    tiny_grid    — 2×2 GridWorld (no pit) for RL tabular tests
    disc_env     — DiscreteEnv(max_steps=50) for DQN / PPO tests
    cont_env     — ContinuousEnv(max_steps=50) for DDPG / SAC / TD3 tests
"""

from __future__ import annotations

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Root RNG — all fixtures that need randomness derive from this seed so
# the entire test suite is fully deterministic.
# ──────────────────────────────────────────────────────────────────────────────

GLOBAL_SEED = 42


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Session-scoped seeded Generator. Reuse it; don't mutate the seed."""
    return np.random.default_rng(GLOBAL_SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Generic ML datasets
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def small_X_y() -> tuple[np.ndarray, np.ndarray]:
    """
    Binary classification: two well-separated 2-D Gaussian blobs.
    Returns (X, y) with X shape (60, 2) and y in {0, 1}.
    """
    gen = np.random.default_rng(0)
    X0 = gen.normal([0.0, 0.0], 0.5, (30, 2))
    X1 = gen.normal([4.0, 4.0], 0.5, (30, 2))
    X  = np.vstack([X0, X1])
    y  = np.array([0] * 30 + [1] * 30)
    return X, y


@pytest.fixture(scope="session")
def three_class_X_y() -> tuple[np.ndarray, np.ndarray]:
    """
    3-class Gaussian blobs, shape (90, 2).
    """
    gen = np.random.default_rng(1)
    centres = [[0, 0], [6, 0], [3, 5]]
    X = np.vstack([gen.normal(c, 0.5, (30, 2)) for c in centres])
    y = np.repeat([0, 1, 2], 30)
    return X, y


@pytest.fixture(scope="session")
def regression_X_y() -> tuple[np.ndarray, np.ndarray]:
    """
    Simple 1-D linear regression: y = 3x + 1.5 + ε.
    Returns (X, y) with X shape (50, 1).
    """
    gen = np.random.default_rng(2)
    X = np.linspace(-2, 2, 50).reshape(-1, 1)
    y = 3.0 * X.ravel() + 1.5 + gen.normal(0, 0.3, 50)
    return X, y


@pytest.fixture(scope="session")
def blobs_X_y() -> tuple[np.ndarray, np.ndarray]:
    """
    Three well-separated clusters for clustering / unsupervised tests.
    X shape (90, 2); y shape (90,) contains ground-truth cluster labels.
    """
    gen = np.random.default_rng(3)
    centres = [[0, 0], [8, 0], [4, 7]]
    X = np.vstack([gen.normal(c, 0.4, (30, 2)) for c in centres])
    y = np.repeat([0, 1, 2], 30)
    return X, y


@pytest.fixture(scope="session")
def multivariate_X_y() -> tuple[np.ndarray, np.ndarray]:
    """
    4-feature dataset, shape (80, 4), target y = 2x1 - x2 + 0.5x3 + ε.
    """
    gen  = np.random.default_rng(4)
    X    = gen.standard_normal((80, 4))
    y    = 2.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + gen.normal(0, 0.2, 80)
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Reinforcement learning environments
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_grid():
    """2×2 GridWorld with no pit — trivially solvable, fast in tests."""
    from mlscratch.reinforcement.utils import GridWorld
    return GridWorld(size=2, pit=(-1, -1))


@pytest.fixture
def small_grid():
    """Standard 4×4 GridWorld with pit at (1,1)."""
    from mlscratch.reinforcement.utils import GridWorld
    return GridWorld(size=4, pit=(1, 1))


@pytest.fixture
def disc_env():
    """DiscreteEnv — fast continuous env with discrete action wrapper."""
    from mlscratch.reinforcement.utils import DiscreteEnv
    return DiscreteEnv(max_steps=50)


@pytest.fixture
def cont_env():
    """ContinuousEnv — 1-D point-mass continuous control."""
    from mlscratch.reinforcement.utils import ContinuousEnv
    return ContinuousEnv(max_steps=50)


# ──────────────────────────────────────────────────────────────────────────────
# Markers
# ──────────────────────────────────────────────────────────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers so -m 'not slow' works without warnings."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with -m 'not slow')",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks integration-level tests",
    )
