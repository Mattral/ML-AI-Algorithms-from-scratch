"""
Linear Regression
=================

Ordinary Least Squares and mini-batch gradient descent implementations for
linear regression.

The module uses a clear, sklearn-compatible interface with explicit math.
"""

from __future__ import annotations

from typing import Callable

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


class OrdinaryLeastSquares:
    """Ordinary least squares regression using the normal equations.

    Parameters
    ----------
    add_intercept : bool, default=True
        If True, the model fits an intercept term by prepending a column of ones
        to the design matrix.

    Attributes
    ----------
    coef_ : FloatArray
        Estimated regression coefficients for each feature.
    intercept_ : float
        Estimated bias term.
    residuals_ : FloatArray
        Residual values after fitting.
    """

    def __init__(self, add_intercept: bool = True) -> None:
        self.add_intercept = add_intercept
        self.coef_: FloatArray | None = None
        self.intercept_: float | None = None
        self.residuals_: FloatArray | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "OrdinaryLeastSquares":
        r"""Fit the linear regression model.

        The closed-form least-squares solution is computed via a numerically
        stable least-squares solver:

        .. math::
            \hat{\beta} = \operatorname{argmin}_\beta \|X \beta - y\|_2^2

        Returns
        -------
        self : OrdinaryLeastSquares
        """
        X_arr, y_arr = _validate_regression_inputs(X, y)
        if self.add_intercept:
            X_arr = np.column_stack([np.ones(X_arr.shape[0]), X_arr])

        solution, residuals, *_ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
        if self.add_intercept:
            self.intercept_ = float(solution[0])
            self.coef_ = solution[1:].astype(np.float64)
            y_pred = X_arr[:, 1:] @ self.coef_ + self.intercept_
        else:
            self.intercept_ = 0.0
            self.coef_ = solution.astype(np.float64)
            y_pred = X_arr @ self.coef_

        self.residuals_ = y_arr - y_pred
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict target values for new data."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")
        result = X_arr @ self.coef_ + self.intercept_
        return result.astype(np.float64)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the coefficient of determination R² on the given data."""
        _, y_arr = _validate_regression_inputs(X, y)
        y_pred = self.predict(X)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


class GradientDescentRegressor:
    r"""Linear regression using mini-batch gradient descent.

    The squared error loss is:

    .. math::
        L(W, b) = \frac{1}{n} \sum_{i=1}^n (y_i - X_i W - b)^2

    The gradient with respect to the weights is:

    .. math::
        \frac{\partial L}{\partial W} = -\frac{2}{n} X^\top (y - XW - b)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 1000,
        batch_size: int = 32,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.coef_: FloatArray | None = None
        self.intercept_: float | None = None
        self.loss_history_: list[float] = []

    def fit(self, X: ArrayLike, y: ArrayLike) -> "GradientDescentRegressor":
        """Fit the model using mini-batch gradient descent."""
        X_arr, y_arr = _validate_regression_inputs(X, y)
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X_arr.shape
        self.coef_ = np.zeros(n_features, dtype=np.float64)
        self.intercept_ = 0.0
        self.loss_history_ = []
        learning_rate = self.learning_rate
        prev_loss = float("inf")

        for epoch in range(self.n_epochs):
            indices = rng.permutation(n_samples)
            X_shuffled = X_arr[indices]
            y_shuffled = y_arr[indices]
            coef_before = self.coef_.copy()
            intercept_before = self.intercept_

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                y_pred = X_batch @ self.coef_ + self.intercept_
                errors = y_pred - y_batch
                grad_w = (2.0 / X_batch.shape[0]) * (X_batch.T @ errors)
                grad_b = (2.0 / X_batch.shape[0]) * np.sum(errors)
                self.coef_ -= learning_rate * grad_w
                self.intercept_ -= learning_rate * grad_b

            loss = np.mean((X_arr @ self.coef_ + self.intercept_ - y_arr) ** 2)
            if loss > prev_loss + 1e-12:
                self.coef_ = coef_before
                self.intercept_ = intercept_before
                learning_rate *= 0.5
                loss = prev_loss

            self.loss_history_.append(float(loss))
            prev_loss = loss
            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch} loss={loss:.6f}")
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict target values for new data."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")
        return (X_arr @ self.coef_ + self.intercept_).astype(np.float64)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the coefficient of determination R² on the given data."""
        _, y_arr = _validate_regression_inputs(X, y)
        y_pred = self.predict(X)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
