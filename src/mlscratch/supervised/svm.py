r"""
Kernel Support Vector Classifier
=================================
A soft-margin SVM solved in its dual form by Platt's Sequential Minimal
Optimization (SMO) algorithm — pure numpy, no external QP solver.

Primal problem (for reference; never solved directly):

.. math::
    \min_{w,b,\xi} \ \tfrac12\|w\|^2 + C\sum_i \xi_i
    \quad \text{s.t.} \quad y_i(w^\top \phi(x_i) + b) \ge 1-\xi_i,\ \xi_i \ge 0

Dual problem (what SMO actually maximises):

.. math::
    \max_{\alpha} \ \sum_i \alpha_i - \tfrac12\sum_{i,j}\alpha_i\alpha_j y_i y_j K(x_i,x_j)
    \quad \text{s.t.} \quad 0 \le \alpha_i \le C,\ \ \sum_i \alpha_i y_i = 0

SMO repeatedly picks a pair :math:`(\alpha_i,\alpha_j)` and solves the
resulting 1-D constrained quadratic exactly in closed form — this is
the simplified heuristic from Platt's original paper / the CS229 SMO
notes, with random second-variable selection rather than the full
working-set heuristic.

Kernels
-------
- ``'linear'``  : :math:`K(x,y) = x^\top y`
- ``'poly'``    : :math:`K(x,y) = (\gamma x^\top y + c_0)^d`
- ``'rbf'``     : :math:`K(x,y) = \exp(-\gamma\|x-y\|^2)`
- ``'sigmoid'`` : :math:`K(x,y) = \tanh(\gamma x^\top y + c_0)`
- a user-supplied callable ``K(X, Y) -> (n_X, n_Y)`` Gram matrix

Multiclass
----------
``SVC`` is natively binary; if more than two classes are present at
``fit`` time it transparently trains one binary SVM per class
(one-vs-rest) and predicts via arg-max of the binary decision
functions.

Complexity
----------
- Training : O(n^2 d) to build the Gram matrix, then O(n) work per SMO
  step for a heuristically-bounded number of sweeps.
- Inference: O(n_SV * d) per sample.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import validate_x, validate_xy

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_EPS = 1e-12
_MAX_SWEEPS = 2000


# ──────────────────────────────────────────────────────────────────────────
# Kernels
# ──────────────────────────────────────────────────────────────────────────


def _linear_kernel(X: FloatArray, Y: FloatArray) -> FloatArray:
    return X @ Y.T


def _poly_kernel(
    X: FloatArray, Y: FloatArray, degree: int, gamma: float, coef0: float
) -> FloatArray:
    return (gamma * (X @ Y.T) + coef0) ** degree


def _rbf_kernel(X: FloatArray, Y: FloatArray, gamma: float) -> FloatArray:
    X_sq = np.sum(X**2, axis=1)[:, None]
    Y_sq = np.sum(Y**2, axis=1)[None, :]
    sq_dists = np.maximum(X_sq + Y_sq - 2.0 * (X @ Y.T), 0.0)
    return np.exp(-gamma * sq_dists)


def _sigmoid_kernel(X: FloatArray, Y: FloatArray, gamma: float, coef0: float) -> FloatArray:
    return np.tanh(gamma * (X @ Y.T) + coef0)


_BUILTIN_KERNELS = {"linear", "poly", "rbf", "sigmoid"}


# ──────────────────────────────────────────────────────────────────────────
# SVC
# ──────────────────────────────────────────────────────────────────────────


class SVC:
    """Kernel Support Vector Classifier trained via Sequential Minimal Optimization.

    Parameters
    ----------
    C : float, default=1.0
        Inverse regularisation strength (penalty on margin violations).
    kernel : str | Callable, default='rbf'
        ``'linear'``, ``'poly'``, ``'rbf'``, ``'sigmoid'``, or a callable
        ``K(X, Y) -> ndarray`` returning a Gram matrix of shape
        ``(len(X), len(Y))``.
    degree : int, default=3
        Degree for the ``'poly'`` kernel.
    gamma : float | str, default='scale'
        Kernel coefficient for ``'rbf'``/``'poly'``/``'sigmoid'``.
        ``'scale'`` uses ``1 / (n_features * X.var())``, ``'auto'`` uses
        ``1 / n_features``.
    coef0 : float, default=0.0
        Independent term for ``'poly'``/``'sigmoid'`` kernels.
    tol : float, default=1e-3
        KKT violation tolerance used to select active variables.
    max_iter : int, default=10
        Number of consecutive full sweeps over the data with *no*
        alpha updates before SMO is declared converged (Platt's
        ``max_passes``). A hard cap of a few thousand total sweeps
        applies regardless, to guarantee termination.
    random_state : int | None, default=None
        Seed for the random selection of the second SMO variable.

    Attributes
    ----------
    support_ : indices of the support vectors within the training data
    support_vectors_ : the support vectors themselves
    dual_coef_ : :math:`\\alpha_i y_i` for each support vector
    intercept_ : the bias term ``b``
    classes_ : sorted unique labels seen during fit
    multiclass_ : whether one-vs-rest decomposition was used
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str | Callable[[FloatArray, FloatArray], FloatArray] = "rbf",
        degree: int = 3,
        gamma: float | str = "scale",
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 10,
        random_state: int | None = None,
    ) -> None:
        if not callable(kernel) and kernel not in _BUILTIN_KERNELS:
            raise ValueError(f"kernel must be one of {_BUILTIN_KERNELS} or a callable.")
        if C <= 0:
            raise ValueError("C must be positive.")
        self.C = float(C)
        self.kernel = kernel
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.random_state = random_state

        self.classes_: NDArray | None = None
        self.n_features_in_: int | None = None
        self.multiclass_: bool = False

        # binary-fit attributes
        self.support_: IntArray | None = None
        self.support_vectors_: FloatArray | None = None
        self.dual_coef_: FloatArray | None = None
        self.intercept_: float | None = None
        self.n_support_: int | None = None
        self.n_iter_: int | None = None

        # multiclass (one-vs-rest) attributes
        self._ovr_estimators_: list[SVC] | None = None

        self._gamma_value: float | None = None
        self._fitted = False

    # -- kernel plumbing -----------------------------------------------------

    def _resolve_gamma(self, X: FloatArray) -> float:
        if isinstance(self.gamma, str):
            if self.gamma == "scale":
                var = float(X.var())
                return 1.0 / (X.shape[1] * var) if var > _EPS else 1.0
            if self.gamma == "auto":
                return 1.0 / X.shape[1]
            raise ValueError("gamma must be 'scale', 'auto', or a positive float.")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")
        return float(self.gamma)

    def _kernel_fn(self, X: FloatArray, Y: FloatArray) -> FloatArray:
        if callable(self.kernel):
            return np.asarray(self.kernel(X, Y), dtype=np.float64)
        if self.kernel == "linear":
            return _linear_kernel(X, Y)
        if self.kernel == "poly":
            return _poly_kernel(X, Y, self.degree, self._gamma_value, self.coef0)
        if self.kernel == "rbf":
            return _rbf_kernel(X, Y, self._gamma_value)
        return _sigmoid_kernel(X, Y, self._gamma_value, self.coef0)

    # -- public API -----------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> SVC:
        """Fit the SVM. Dispatches to one-vs-rest if >2 classes are present."""
        X_arr, y_arr = validate_xy(X, y)
        self.classes_ = np.unique(y_arr)
        self.n_features_in_ = X_arr.shape[1]
        self._gamma_value = self._resolve_gamma(X_arr)

        if self.classes_.size < 2:
            raise ValueError("SVC requires at least 2 classes.")
        if self.classes_.size == 2:
            self.multiclass_ = False
            self._fit_binary(X_arr, y_arr)
        else:
            self.multiclass_ = True
            self._ovr_estimators_ = []
            for cls in self.classes_:
                binary_y = (y_arr == cls).astype(np.int64)  # {0, 1}, 1 = "is this class"
                sub = SVC(
                    C=self.C,
                    kernel=self.kernel,
                    degree=self.degree,
                    gamma=self.gamma,
                    coef0=self.coef0,
                    tol=self.tol,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )
                sub._gamma_value = self._gamma_value
                sub.n_features_in_ = self.n_features_in_
                sub._fit_binary(X_arr, binary_y)
                self._ovr_estimators_.append(sub)
        self._fitted = True
        return self

    def decision_function(self, X: ArrayLike) -> FloatArray:
        """Signed distance to the separating hyperplane (margin).

        For multiclass, returns shape ``(n_samples, n_classes)`` — one
        one-vs-rest margin per class.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before decision_function().")
        X_arr = validate_x(X)
        if self.multiclass_:
            return np.column_stack([est.decision_function(X_arr) for est in self._ovr_estimators_])
        K = self._kernel_fn(self.support_vectors_, X_arr)
        return self.dual_coef_ @ K + self.intercept_

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict class labels."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        scores = self.decision_function(X)
        if self.multiclass_:
            return self.classes_[np.argmax(scores, axis=1)]
        return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy."""
        X_arr, y_arr = validate_xy(X, y)
        return float(np.mean(self.predict(X_arr) == y_arr))

    # -- core binary SMO solver -------------------------------------------------

    def _fit_binary(self, X_arr: FloatArray, y_arr: NDArray) -> None:
        classes = np.unique(y_arr)
        if classes.size != 2:
            raise ValueError("_fit_binary requires exactly 2 classes.")
        self.classes_ = classes
        y_signed = np.where(y_arr == classes[0], -1.0, 1.0)
        n = X_arr.shape[0]
        K = self._kernel_fn(X_arr, X_arr)

        alpha = np.zeros(n, dtype=np.float64)
        b = 0.0
        rng = np.random.default_rng(self.random_state)
        ay = alpha * y_signed  # kept in sync after every update

        def f(i: int) -> float:
            return float(np.dot(ay, K[:, i]) + b)

        passes = 0
        sweeps = 0
        while passes < self.max_iter and sweeps < _MAX_SWEEPS:
            num_changed = 0
            for i in range(n):
                Ei = f(i) - y_signed[i]
                violates = (y_signed[i] * Ei < -self.tol and alpha[i] < self.C) or (
                    y_signed[i] * Ei > self.tol and alpha[i] > 0.0
                )
                if not violates:
                    continue

                j = i
                while j == i:
                    j = int(rng.integers(0, n))
                Ej = f(j) - y_signed[j]

                ai_old, aj_old = alpha[i], alpha[j]
                if y_signed[i] != y_signed[j]:
                    lo = max(0.0, alpha[j] - alpha[i])
                    hi = min(self.C, self.C + alpha[j] - alpha[i])
                else:
                    lo = max(0.0, alpha[i] + alpha[j] - self.C)
                    hi = min(self.C, alpha[i] + alpha[j])
                if lo >= hi:
                    continue

                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                alpha[j] -= y_signed[j] * (Ei - Ej) / eta
                alpha[j] = min(hi, max(lo, alpha[j]))
                if abs(alpha[j] - aj_old) < 1e-7:
                    continue
                alpha[i] += y_signed[i] * y_signed[j] * (aj_old - alpha[j])

                b1 = (
                    b
                    - Ei
                    - y_signed[i] * (alpha[i] - ai_old) * K[i, i]
                    - y_signed[j] * (alpha[j] - aj_old) * K[i, j]
                )
                b2 = (
                    b
                    - Ej
                    - y_signed[i] * (alpha[i] - ai_old) * K[i, j]
                    - y_signed[j] * (alpha[j] - aj_old) * K[j, j]
                )
                if 0.0 < alpha[i] < self.C:
                    b = b1
                elif 0.0 < alpha[j] < self.C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                ay[i] = alpha[i] * y_signed[i]
                ay[j] = alpha[j] * y_signed[j]
                num_changed += 1

            sweeps += 1
            passes = passes + 1 if num_changed == 0 else 0

        sv_mask = alpha > 1e-8
        self.support_ = np.flatnonzero(sv_mask).astype(np.int64)
        self.support_vectors_ = X_arr[sv_mask]
        self.dual_coef_ = ay[sv_mask]
        self.intercept_ = float(b)
        self.n_support_ = int(sv_mask.sum())
        self.n_iter_ = sweeps
        self._fitted = True
