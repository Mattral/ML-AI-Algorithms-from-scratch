"""
_datasets.py — tiny synthetic-data generators for the example scripts.

Deliberately pure numpy (no scikit-learn, no downloads) so every example
in this folder runs with nothing more than ``pip install mlscratch``.
"""

from __future__ import annotations

import numpy as np


def make_blobs(
    n_samples: int = 300,
    centers: tuple = ((-4.0, -4.0), (4.0, 4.0), (-4.0, 4.0)),
    cluster_std: float = 1.2,
    random_state: int | None = 0,
):
    """Isotropic Gaussian blobs, one per entry in ``centers``."""
    rng = np.random.default_rng(random_state)
    centers_arr = np.asarray(centers, dtype=float)
    n_per_cluster = n_samples // len(centers_arr)

    X_parts, y_parts = [], []
    for k, c in enumerate(centers_arr):
        X_parts.append(rng.normal(loc=c, scale=cluster_std, size=(n_per_cluster, c.shape[0])))
        y_parts.append(np.full(n_per_cluster, k))
    X, y = np.vstack(X_parts), np.concatenate(y_parts)

    order = rng.permutation(len(y))
    return X[order], y[order]


def make_moons(n_samples: int = 300, noise: float = 0.15, random_state: int | None = 0):
    """Two interleaving half-moons — a classic non-linearly-separable benchmark."""
    rng = np.random.default_rng(random_state)
    n_half = n_samples // 2

    theta = np.linspace(0, np.pi, n_half)
    moon_a = np.column_stack([np.cos(theta), np.sin(theta)])
    moon_b = np.column_stack([1.0 - np.cos(theta), 1.0 - np.sin(theta) - 0.5])

    X = np.vstack([moon_a, moon_b]) + rng.normal(0, noise, (2 * n_half, 2))
    y = np.concatenate([np.zeros(n_half), np.ones(n_half)]).astype(int)

    order = rng.permutation(len(y))
    return X[order], y[order]


def make_regression_line(
    n_samples: int = 200, n_features: int = 1, noise: float = 5.0, random_state: int | None = 0
):
    """A linear function of the inputs plus Gaussian noise."""
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-5, 5, size=(n_samples, n_features))
    true_coef = rng.uniform(1, 5, size=n_features)
    y = X @ true_coef + rng.normal(0, noise, n_samples)
    return X, y


def make_sine_regression(n_samples: int = 200, noise: float = 0.3, random_state: int | None = 0):
    """A noisy sine wave — a simple non-linear regression benchmark."""
    rng = np.random.default_rng(random_state)
    X = np.sort(rng.uniform(-3, 3, size=(n_samples, 1)), axis=0)
    y = np.sin(X.ravel() * 1.5) * 3.0 + rng.normal(0, noise, n_samples)
    return X, y
