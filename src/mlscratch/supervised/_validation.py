r"""
mlscratch.supervised._validation
==================================
Internal, private input-validation helpers shared by every estimator in
``mlscratch.supervised`` (trees, ensembles, kernel SVM). Not part of the
public API — do not import this module from outside the package.

Centralising these checks means every estimator raises the *same*
exception type with the *same* message shape for the *same* mistake
(wrong-dimensional `X`, mismatched `X`/`y` length, malformed
`sample_weight`), which is what lets the test suite assert on a single
shared error-message contract (e.g. ``match="samples"``) across five
otherwise-independent algorithm implementations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def validate_x(X: ArrayLike) -> FloatArray:
    """Coerce X to a float64 ndarray and require it to be 2D."""
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    return X_arr


def validate_xy(X: ArrayLike, y: ArrayLike) -> tuple[FloatArray, NDArray]:
    """validate_x(X), plus require y to share X's sample count."""
    X_arr = validate_x(X)
    y_arr = np.asarray(y).flatten()
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}.")
    return X_arr, y_arr


def validate_sample_weight(sample_weight: ArrayLike | None, n_samples: int) -> FloatArray:
    """Return a uniform weight vector if None, else validate a user-supplied one."""
    if sample_weight is None:
        return np.ones(n_samples, dtype=np.float64)
    w = np.asarray(sample_weight, dtype=np.float64).flatten()
    if w.shape[0] != n_samples:
        raise ValueError(f"sample_weight has {w.shape[0]} entries but X has {n_samples} samples.")
    if np.any(w < 0):
        raise ValueError("sample_weight entries must be non-negative.")
    return w
