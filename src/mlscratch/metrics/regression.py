r"""
Regression Metrics
===================
Evaluation metrics for regressors, implemented from scratch in pure numpy.

.. math::
    \mathrm{MSE} = \frac1n\sum_i (y_i-\hat y_i)^2, \qquad
    \mathrm{RMSE} = \sqrt{\mathrm{MSE}}, \qquad
    \mathrm{MAE} = \frac1n\sum_i |y_i-\hat y_i|

.. math::
    \mathrm{MAPE} = \frac1n\sum_i \left|\frac{y_i-\hat y_i}{y_i}\right|, \qquad
    R^2 = 1 - \frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}

.. math::
    \mathrm{ExplainedVariance} = 1 - \frac{\mathrm{Var}(y-\hat y)}{\mathrm{Var}(y)}
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_EPS = 1e-12


def _validate(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y_true_arr = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).flatten()
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"y_true has {y_true_arr.shape[0]} samples but y_pred has {y_pred_arr.shape[0]}."
        )
    if y_true_arr.shape[0] == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    return y_true_arr, y_pred_arr


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike, squared: bool = True) -> float:
    """Mean squared error; pass ``squared=False`` for RMSE."""
    y_true_arr, y_pred_arr = _validate(y_true, y_pred)
    mse = float(np.mean((y_true_arr - y_pred_arr) ** 2))
    return mse if squared else float(np.sqrt(mse))


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """:math:`\\sqrt{\\mathrm{MSE}}`."""
    return mean_squared_error(y_true, y_pred, squared=False)


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true_arr, y_pred_arr = _validate(y_true, y_pred)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def mean_absolute_percentage_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute percentage error. Entries with ``|y_true| < eps`` are
    floored at ``eps`` to avoid division by zero, matching common practice."""
    y_true_arr, y_pred_arr = _validate(y_true, y_pred)
    denom = np.where(np.abs(y_true_arr) < _EPS, _EPS, np.abs(y_true_arr))
    return float(np.mean(np.abs((y_true_arr - y_pred_arr) / denom)))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination. Returns 0.0 (rather than NaN/inf) when
    the target has zero variance, a common, well-documented convention."""
    y_true_arr, y_pred_arr = _validate(y_true, y_pred)
    ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
    ss_tot = float(np.sum((y_true_arr - y_true_arr.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > _EPS else 0.0


def explained_variance_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true_arr, y_pred_arr = _validate(y_true, y_pred)
    var_true = float(np.var(y_true_arr))
    var_residual = float(np.var(y_true_arr - y_pred_arr))
    return 1.0 - var_residual / var_true if var_true > _EPS else 0.0
