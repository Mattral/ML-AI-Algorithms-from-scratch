"""
Lasso Regression
================

Lasso regression using coordinate descent and an explicit intercept.
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


class LassoRegression:
    """Lasso regression using coordinate descent.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength for the L1 penalty.
    max_iter : int, default=1000
        Maximum number of coordinate descent iterations.
    tol : float, default=1e-4
        Convergence threshold for coefficient updates.

    Attributes
    ----------
    coef_ : FloatArray
        Estimated coefficients for each feature.
    intercept_ : float
        Estimated intercept term.
    loss_history_ : list[float]
        Training loss on each iteration.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coef_: FloatArray | None = None
        self.intercept_: float | None = None
        self.loss_history_: list[float] = []
        self.feature_means_: FloatArray | None = None
        self.y_mean_: float | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LassoRegression":
        """Fit the Lasso regression model to the training data."""
        X_arr, y_arr = _validate_regression_inputs(X, y)
        n_samples, n_features = X_arr.shape
        self.feature_means_ = np.mean(X_arr, axis=0)
        self.y_mean_ = np.mean(y_arr)
        X_centered = X_arr - self.feature_means_
        y_centered = y_arr - self.y_mean_

        self.coef_ = np.zeros(n_features, dtype=np.float64)
        self.intercept_ = 0.0
        self.loss_history_ = []

        X_norm_sq = np.sum(X_centered**2, axis=0) / n_samples
        X_norm_sq = np.where(X_norm_sq == 0.0, 1.0, X_norm_sq)

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()

            for j in range(n_features):
                residual = y_centered - (X_centered @ self.coef_ - X_centered[:, j] * self.coef_[j])
                rho = (X_centered[:, j] @ residual) / n_samples
                if rho < -self.alpha:
                    self.coef_[j] = (rho + self.alpha) / X_norm_sq[j]
                elif rho > self.alpha:
                    self.coef_[j] = (rho - self.alpha) / X_norm_sq[j]
                else:
                    self.coef_[j] = 0.0

            max_coef_change = np.max(np.abs(self.coef_ - coef_old))
            self.intercept_ = self.y_mean_ - float(self.feature_means_ @ self.coef_)
            loss = self._objective(X_arr, y_arr)
            self.loss_history_.append(float(loss))
            if max_coef_change < self.tol:
                break

        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict using the fitted Lasso model."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")
        return (X_arr @ self.coef_ + self.intercept_).astype(np.float64)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return R² of the fitted model on the given data."""
        X_arr, y_arr = _validate_regression_inputs(X, y)
        y_pred = self.predict(X_arr)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def _objective(self, X: FloatArray, y: FloatArray) -> float:
        y_pred = X @ self.coef_ + self.intercept_
        mse = np.mean((y - y_pred) ** 2) / 2.0
        return float(mse + self.alpha * np.sum(np.abs(self.coef_)))
