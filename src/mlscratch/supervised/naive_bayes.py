r"""
Gaussian Naive Bayes Classifier
================================

A probabilistic classifier that assumes each class follows a Gaussian
distribution and that features are conditionally independent given the class.

The model computes class log-likelihoods as:

.. math::
    \log p(\mathbf{x}, y_k)
    = \log \pi_k - \frac{1}{2} \sum_{j=1}^d \left[
        \log(2\pi \sigma_{kj}^2)
        + \frac{(x_j - \mu_{kj})^2}{\sigma_{kj}^2}
    \right]

Complexity
----------
- Training: O(n d)
- Inference: O(n d)
- Space:    O(K d)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _validate_classification_inputs(
    X: ArrayLike,
    y: ArrayLike,
) -> tuple[FloatArray, IntArray]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int).flatten()
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}."
        )
    return X_arr, y_arr


class GaussianNB:
    """Gaussian Naive Bayes classifier.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features added to variances for
        stability in the Gaussian likelihood denominator.
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = float(var_smoothing)
        self.class_count_: IntArray | None = None
        self.class_prior_: FloatArray | None = None
        self.class_mean_: FloatArray | None = None
        self.class_var_: FloatArray | None = None
        self.classes_: IntArray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "GaussianNB":
        X_arr, y_arr = _validate_classification_inputs(X, y)
        self.n_features_in_ = X_arr.shape[1]
        self.classes_, counts = np.unique(y_arr, return_counts=True)
        self.class_count_ = counts.astype(np.int64)
        self.class_prior_ = counts.astype(np.float64) / float(y_arr.size)

        means = []
        variances = []
        for clazz in self.classes_:
            X_class = X_arr[y_arr == clazz]
            means.append(X_class.mean(axis=0))
            variances.append(X_class.var(axis=0) + self.var_smoothing)

        self.class_mean_ = np.vstack(means)
        self.class_var_ = np.vstack(variances)
        return self

    def predict(self, X: ArrayLike) -> IntArray:
        if self.class_prior_ is None or self.class_mean_ is None or self.class_var_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")
        log_likelihood = self._joint_log_likelihood(X_arr)
        argmax = np.argmax(log_likelihood, axis=1)
        return self.classes_[argmax]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        X_arr, y_arr = _validate_classification_inputs(X, y)
        return float(np.mean(self.predict(X_arr) == y_arr))

    def _joint_log_likelihood(self, X: FloatArray) -> FloatArray:
        n_samples, n_features = X.shape
        if self.class_mean_ is None or self.class_var_ is None or self.class_prior_ is None:
            raise RuntimeError("Classifier must be fitted before computing likelihoods.")

        joint = np.empty((n_samples, self.classes_.size), dtype=np.float64)
        for idx, (prior, mean, var) in enumerate(
            zip(self.class_prior_, self.class_mean_, self.class_var_)
        ):
            log_prior = np.log(prior)
            log_det = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            diff = X - mean
            exp_term = -0.5 * np.sum((diff ** 2) / var, axis=1)
            joint[:, idx] = log_prior + log_det + exp_term
        return joint
