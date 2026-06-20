r"""
Feature Scalers
================
Stateful, sklearn-style fit/transform preprocessors for rescaling
feature columns prior to model fitting.

StandardScaler
---------------
.. math::
    x' = \frac{x - \mu}{\sigma}

MinMaxScaler
-------------
.. math::
    x' = \frac{x - x_{\min}}{x_{\max}-x_{\min}} \cdot (b-a) + a

RobustScaler
-------------
Centres on the median and scales by the interquartile range, so it is
not skewed by outliers the way StandardScaler's mean/std are.

.. math::
    x' = \frac{x - \mathrm{median}(x)}{Q_3(x) - Q_1(x)}

Normalizer
-----------
Row-wise (per-sample) rescaling to a unit L1, L2, or max norm —
stateless: every row is scaled independently of every other row, so
``fit`` is a no-op kept only for API symmetry with the other
transformers (and so it can be dropped into a uniform preprocessing
pipeline).

Complexity
----------
O(n d) fit and transform for all four transformers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
_EPS = 1e-12


def _validate_x(X: ArrayLike) -> FloatArray:
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    return X_arr


def _check_n_features(X_arr: FloatArray, expected: int) -> None:
    if X_arr.shape[1] != expected:
        raise ValueError(f"X has {X_arr.shape[1]} features but transformer was fit on {expected}.")


class StandardScaler:
    """Standardise features to zero mean and unit variance.

    Parameters
    ----------
    with_mean : bool, default=True
    with_std : bool, default=True

    Attributes
    ----------
    mean_, scale_ : per-feature mean and standard deviation used to
        transform (``scale_`` is floored at 1.0 for constant columns,
        so they pass through as all-zero rather than producing NaNs).
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: FloatArray | None = None
        self.scale_: FloatArray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike) -> StandardScaler:
        X_arr = _validate_x(X)
        self.n_features_in_ = X_arr.shape[1]
        self.mean_ = X_arr.mean(axis=0) if self.with_mean else np.zeros(X_arr.shape[1])
        std = X_arr.std(axis=0) if self.with_std else np.ones(X_arr.shape[1])
        self.scale_ = np.where(std > _EPS, std, 1.0)
        return self

    def transform(self, X: ArrayLike) -> FloatArray:
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform().")
        X_arr = _validate_x(X)
        _check_n_features(X_arr, self.n_features_in_)
        return (X_arr - self.mean_) / self.scale_

    def fit_transform(self, X: ArrayLike) -> FloatArray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: ArrayLike) -> FloatArray:
        if self.mean_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        X_arr = _validate_x(X)
        _check_n_features(X_arr, self.n_features_in_)
        return X_arr * self.scale_ + self.mean_


class MinMaxScaler:
    """Linearly rescale features into ``feature_range`` (default ``[0, 1]``)."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError("feature_range must satisfy min < max.")
        self.feature_range = feature_range
        self.data_min_: FloatArray | None = None
        self.data_max_: FloatArray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike) -> MinMaxScaler:
        X_arr = _validate_x(X)
        self.n_features_in_ = X_arr.shape[1]
        self.data_min_ = X_arr.min(axis=0)
        self.data_max_ = X_arr.max(axis=0)
        return self

    def transform(self, X: ArrayLike) -> FloatArray:
        if self.data_min_ is None:
            raise RuntimeError("Call fit() before transform().")
        X_arr = _validate_x(X)
        _check_n_features(X_arr, self.n_features_in_)
        data_range = np.where(
            self.data_max_ - self.data_min_ > _EPS, self.data_max_ - self.data_min_, 1.0
        )
        lo, hi = self.feature_range
        return (X_arr - self.data_min_) / data_range * (hi - lo) + lo

    def fit_transform(self, X: ArrayLike) -> FloatArray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: ArrayLike) -> FloatArray:
        if self.data_min_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        X_arr = _validate_x(X)
        _check_n_features(X_arr, self.n_features_in_)
        data_range = self.data_max_ - self.data_min_
        lo, hi = self.feature_range
        return (X_arr - lo) / (hi - lo) * data_range + self.data_min_


class RobustScaler:
    """Centre on the median and scale by the interquartile range (IQR),
    so outliers (which inflate mean/std) don't dominate the scaling."""

    def __init__(self, quantile_range: tuple[float, float] = (25.0, 75.0)) -> None:
        q_min, q_max = quantile_range
        if not (0.0 <= q_min < q_max <= 100.0):
            raise ValueError("quantile_range must satisfy 0 <= q_min < q_max <= 100.")
        self.quantile_range = quantile_range
        self.center_: FloatArray | None = None
        self.scale_: FloatArray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike) -> RobustScaler:
        X_arr = _validate_x(X)
        self.n_features_in_ = X_arr.shape[1]
        self.center_ = np.median(X_arr, axis=0)
        q_min, q_max = self.quantile_range
        iqr = np.percentile(X_arr, q_max, axis=0) - np.percentile(X_arr, q_min, axis=0)
        self.scale_ = np.where(iqr > _EPS, iqr, 1.0)
        return self

    def transform(self, X: ArrayLike) -> FloatArray:
        if self.center_ is None:
            raise RuntimeError("Call fit() before transform().")
        X_arr = _validate_x(X)
        _check_n_features(X_arr, self.n_features_in_)
        return (X_arr - self.center_) / self.scale_

    def fit_transform(self, X: ArrayLike) -> FloatArray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: ArrayLike) -> FloatArray:
        if self.center_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        X_arr = _validate_x(X)
        _check_n_features(X_arr, self.n_features_in_)
        return X_arr * self.scale_ + self.center_


class Normalizer:
    """Rescale each *row* (sample) independently to unit norm.

    Parameters
    ----------
    norm : str, default='l2'
        ``'l1'``, ``'l2'``, or ``'max'``.
    """

    def __init__(self, norm: str = "l2") -> None:
        if norm not in ("l1", "l2", "max"):
            raise ValueError("norm must be 'l1', 'l2', or 'max'.")
        self.norm = norm

    def fit(self, X: ArrayLike) -> Normalizer:
        _validate_x(X)  # validate only; this transformer is stateless
        return self

    def transform(self, X: ArrayLike) -> FloatArray:
        X_arr = _validate_x(X)
        if self.norm == "l1":
            norms = np.sum(np.abs(X_arr), axis=1)
        elif self.norm == "l2":
            norms = np.sqrt(np.sum(X_arr**2, axis=1))
        else:
            norms = np.max(np.abs(X_arr), axis=1)
        norms = np.where(norms > _EPS, norms, 1.0)
        return X_arr / norms[:, None]

    def fit_transform(self, X: ArrayLike) -> FloatArray:
        return self.fit(X).transform(X)
