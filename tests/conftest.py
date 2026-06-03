"""Shared fixtures for all test suites."""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    """Seeded RNG for reproducible random data."""
    return np.random.default_rng(42)


@pytest.fixture
def linear_dataset(rng):
    """Simple linear regression dataset: y = 2x1 + 3x2 + noise."""
    n = 200
    X = rng.standard_normal((n, 2))
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + rng.standard_normal(n) * 0.1
    return X, y


@pytest.fixture
def binary_classification_dataset(rng):
    """2-class, 2-feature dataset for classifier tests."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    return X.astype(float), y.astype(float)


@pytest.fixture
def multiclass_dataset(rng):
    """3-class dataset for softmax / multiclass tests."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=400,
        n_features=4,
        n_classes=3,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    return X.astype(float), y
