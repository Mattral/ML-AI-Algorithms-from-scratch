"""
Ridge Regression
================

Ridge regression using the closed-form regularized normal equations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _validate_regression_inputs(
    X: ArrayLike, y: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).flatten()
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}."
        )
    return X_arr, y_arr


class RidgeRegression:
    """Ridge regression with an L2 penalty on coefficients.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength (L2 penalty coefficient).
    add_intercept : bool, default=True
        Whether to fit an intercept term.

    Attributes
    ----------
    coef_ : FloatArray
        Estimated coefficients for each feature.
    intercept_ : float
        Estimated intercept.
    """

    def __init__(self, alpha: float = 1.0, add_intercept: bool = True) -> None:
        self.alpha = float(alpha)
        self.add_intercept = add_intercept
        self.coef_: FloatArray | None = None
        self.intercept_: float | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RidgeRegression":
        """Fit the Ridge regression model using the closed-form solution."""
        X_arr, y_arr = _validate_regression_inputs(X, y)
        if self.add_intercept:
            X_arr = np.column_stack([np.ones(X_arr.shape[0]), X_arr])

        n_features = X_arr.shape[1]
        identity = np.eye(n_features)
        if self.add_intercept:
            identity[0, 0] = 0.0

        coef = np.linalg.solve(
            X_arr.T @ X_arr + self.alpha * identity,
            X_arr.T @ y_arr,
        )

        if self.add_intercept:
            self.intercept_ = float(coef[0])
            self.coef_ = coef[1:].astype(np.float64)
        else:
            self.intercept_ = 0.0
            self.coef_ = coef.astype(np.float64)
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict targets using the fitted Ridge model."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")
        return (X_arr @ self.coef_ + self.intercept_).astype(np.float64)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return R² of the fitted Ridge model."""
        X_arr, y_arr = _validate_regression_inputs(X, y)
        y_pred = self.predict(X_arr)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
