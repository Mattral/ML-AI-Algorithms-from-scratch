"""
Linear Models
=============
Five regularised linear models implemented from scratch.

LinearRegression
-----------------
Ordinary Least Squares via the normal equations and optional gradient descent.
    L(w) = ‖y - Xw‖²

RidgeRegression  (L2)
----------------------
Adds an L2 penalty on weights:
    L(w) = ‖y - Xw‖² + α‖w‖²
Closed-form:  w = (XᵀX + αI)⁻¹ Xᵀy

LassoRegression  (L1)
----------------------
Adds an L1 penalty; solved via coordinate descent:
    L(w) = ‖y - Xw‖² + α Σ|wⱼ|

ElasticNet  (L1 + L2)
----------------------
    L(w) = ‖y - Xw‖² + α·ρ‖w‖₁ + α·(1-ρ)/2·‖w‖²
Coordinate descent, same convergence guarantee as Lasso.

LogisticRegression
-------------------
Binary or multi-class (OvR) with L2 regularisation:
    Binary:   p = σ(Xw + b)
    Multi:    P = softmax(XW + b)
Trained by mini-batch gradient descent.

Only numpy and Python stdlib are used.
"""

from __future__ import annotations
import numpy as np


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def _add_bias(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


# ────────────────────────────────────────────────────────────────
# LinearRegression
# ────────────────────────────────────────────────────────────────

class LinearRegression:
    """
    Ordinary Least Squares Linear Regression.

    Solver ``'exact'``: closed-form normal equations  w = (XᵀX)⁻¹ Xᵀy.
    Solver ``'sgd'``:   mini-batch stochastic gradient descent.

    Parameters
    ----------
    fit_intercept : bool
    solver        : str   'exact' | 'sgd'
    learning_rate : float (sgd only)
    epochs        : int   (sgd only)
    batch_size    : int | None
    random_state  : int | None
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        solver: str = "exact",
        learning_rate: float = 0.01,
        epochs: int = 1000,
        batch_size: int | None = 32,
        random_state: int | None = None,
    ) -> None:
        if solver not in {"exact", "sgd"}:
            raise ValueError("solver must be 'exact' or 'sgd'.")
        self.fit_intercept = fit_intercept
        self.solver        = solver
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.batch_size    = batch_size
        self._rng          = np.random.default_rng(random_state)

        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.losses_: list[float] = []

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        return _add_bias(X) if self.fit_intercept else X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        Xp = self._prepare(X)
        n, p = Xp.shape
        self.losses_ = []

        if self.solver == "exact":
            # w = (XᵀX)⁻¹ Xᵀy  — regularised with tiny ridge for stability
            A = Xp.T @ Xp + 1e-12 * np.eye(p)
            w = np.linalg.solve(A, Xp.T @ y)
        else:
            w = np.zeros(p)
            bs = self.batch_size or n
            for _ in range(self.epochs):
                idx = self._rng.permutation(n)
                ep_loss = 0.0
                for start in range(0, n, bs):
                    mb = idx[start:start + bs]
                    Xb, yb = Xp[mb], y[mb]
                    resid = Xb @ w - yb
                    w -= self.learning_rate * 2 * Xb.T @ resid / len(mb)
                    ep_loss += float(np.mean(resid ** 2))
                self.losses_.append(ep_loss / max(1, n // bs))

        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_      = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_      = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

    @property
    def r2_score(self) -> float | None:
        return None   # computed externally; stub for API completeness


# ────────────────────────────────────────────────────────────────
# RidgeRegression
# ────────────────────────────────────────────────────────────────

class RidgeRegression:
    """
    L2-regularised linear regression (Ridge).

    Parameters
    ----------
    alpha         : float   regularisation strength
    fit_intercept : bool
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
    ) -> None:
        self.alpha         = alpha
        self.fit_intercept = fit_intercept
        self.coef_:      np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        Xp = _add_bias(X) if self.fit_intercept else X
        n, p = Xp.shape
        # Bias column is NOT penalised — zero-out its λ row/col
        reg = self.alpha * np.eye(p)
        if self.fit_intercept:
            reg[0, 0] = 0.0
        w = np.linalg.solve(Xp.T @ Xp + reg, Xp.T @ y)
        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_      = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_      = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


# ────────────────────────────────────────────────────────────────
# LassoRegression
# ────────────────────────────────────────────────────────────────

class LassoRegression:
    """
    L1-regularised linear regression (Lasso) via coordinate descent.

    Parameters
    ----------
    alpha         : float   regularisation strength
    fit_intercept : bool
    max_iter      : int
    tol           : float   coordinate descent convergence tolerance
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        self.alpha         = alpha
        self.fit_intercept = fit_intercept
        self.max_iter      = max_iter
        self.tol           = tol
        self.coef_:      np.ndarray | None = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    @staticmethod
    def _soft_threshold(rho: float, alpha: float) -> float:
        """Coordinate descent closed-form update: S(ρ, α)."""
        if rho > alpha:
            return rho - alpha
        if rho < -alpha:
            return rho + alpha
        return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegression":
        n, p = X.shape
        # Centring (intercept handled by shifting)
        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = float(y.mean())
            Xc = X - X_mean
            yc = y - y_mean
        else:
            Xc, yc, X_mean, y_mean = X, y, np.zeros(p), 0.0

        w = np.zeros(p)

        for it in range(self.max_iter):
            w_old = w.copy()
            for j in range(p):
                # Partial residual (exclude feature j)
                r_j = yc - Xc @ w + Xc[:, j] * w[j]
                rho = float(Xc[:, j] @ r_j) / (Xc[:, j] @ Xc[:, j] + 1e-12)
                w[j] = self._soft_threshold(rho, self.alpha / (2 * n))
            self.n_iter_ = it + 1
            if np.max(np.abs(w - w_old)) < self.tol:
                break

        self.coef_ = w
        if self.fit_intercept:
            self.intercept_ = float(y_mean - X_mean @ w)
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


# ────────────────────────────────────────────────────────────────
# ElasticNet
# ────────────────────────────────────────────────────────────────

class ElasticNet:
    """
    Elastic-Net regression (L1 + L2) via coordinate descent.

    Parameters
    ----------
    alpha : float   total regularisation strength
    l1_ratio : float  ρ ∈ [0,1]; 0 = Ridge, 1 = Lasso
    fit_intercept : bool
    max_iter : int
    tol : float
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        self.alpha         = alpha
        self.l1_ratio      = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter      = max_iter
        self.tol           = tol
        self.coef_:      np.ndarray | None = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNet":
        n, p = X.shape
        if self.fit_intercept:
            X_mean, y_mean = X.mean(axis=0), float(y.mean())
            Xc, yc = X - X_mean, y - y_mean
        else:
            Xc, yc, X_mean, y_mean = X, y, np.zeros(p), 0.0

        w = np.zeros(p)
        alpha_l1 = self.alpha * self.l1_ratio
        alpha_l2 = self.alpha * (1 - self.l1_ratio)

        for it in range(self.max_iter):
            w_old = w.copy()
            for j in range(p):
                r_j  = yc - Xc @ w + Xc[:, j] * w[j]
                rho  = float(Xc[:, j] @ r_j) / n
                denom = float(Xc[:, j] @ Xc[:, j]) / n + alpha_l2
                # Coordinate update: soft-threshold then scale by L2 denominator
                if rho > alpha_l1:
                    w[j] = (rho - alpha_l1) / denom
                elif rho < -alpha_l1:
                    w[j] = (rho + alpha_l1) / denom
                else:
                    w[j] = 0.0
            self.n_iter_ = it + 1
            if np.max(np.abs(w - w_old)) < self.tol:
                break

        self.coef_ = w
        if self.fit_intercept:
            self.intercept_ = float(y_mean - X_mean @ w)
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


# ────────────────────────────────────────────────────────────────
# LogisticRegression
# ────────────────────────────────────────────────────────────────

class LogisticRegression:
    """
    Logistic Regression (binary and multi-class OvR).

    Parameters
    ----------
    C             : float   inverse regularisation strength (larger = less reg)
    fit_intercept : bool
    multi_class   : str     'binary' (auto-detected) or 'ovr'
    learning_rate : float
    epochs        : int
    batch_size    : int | None
    tol           : float   early-stop on loss change
    random_state  : int | None
    """

    def __init__(
        self,
        C: float = 1.0,
        fit_intercept: bool = True,
        multi_class: str = "auto",
        learning_rate: float = 0.1,
        epochs: int = 200,
        batch_size: int | None = 32,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.C             = C
        self.fit_intercept = fit_intercept
        self.multi_class   = multi_class
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.tol           = tol
        self._rng          = np.random.default_rng(random_state)

        self.classes_:   np.ndarray | None = None
        self.coef_:      np.ndarray | None = None   # (n_classes, n_features) or (n_features,)
        self.intercept_: np.ndarray | None = None   # (n_classes,)         or scalar
        self.losses_:    list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        n, p = X.shape

        if K == 2 or self.multi_class == "binary":
            self._fit_binary(X, (y == self.classes_[1]).astype(float), n, p)
        else:
            self._fit_ovr(X, y, n, p, K)
        return self

    def _fit_binary(self, X, y, n, p):
        w = np.zeros(p)
        b = 0.0
        bs = self.batch_size or n
        prev_loss = np.inf
        self.losses_ = []

        for _ in range(self.epochs):
            idx = self._rng.permutation(n)
            ep_loss = 0.0
            n_batches = 0
            for start in range(0, n, bs):
                mb = idx[start:start + bs]
                Xb, yb = X[mb], y[mb]
                p_hat = _sigmoid(Xb @ w + b)
                # Gradient with L2 regularisation (C = 1/lambda)
                eps = 1e-8
                loss = -np.mean(yb * np.log(p_hat + eps) +
                                (1 - yb) * np.log(1 - p_hat + eps))
                err  = p_hat - yb
                dw   = Xb.T @ err / len(mb) + w / (self.C * n)
                db   = err.mean()
                w   -= self.learning_rate * dw
                b   -= self.learning_rate * db
                ep_loss += loss
                n_batches += 1
            ep_loss /= n_batches
            self.losses_.append(ep_loss)
            if abs(prev_loss - ep_loss) < self.tol:
                break
            prev_loss = ep_loss

        self.coef_      = w
        self.intercept_ = np.array([b])

    def _fit_ovr(self, X, y, n, p, K):
        """One-vs-Rest: train K binary classifiers."""
        self.coef_      = np.zeros((K, p))
        self.intercept_ = np.zeros(K)
        for k, cls in enumerate(self.classes_):
            y_bin = (y == cls).astype(float)
            w = np.zeros(p); b = 0.0
            bs = self.batch_size or n
            for _ in range(self.epochs):
                idx = self._rng.permutation(n)
                for start in range(0, n, bs):
                    mb = idx[start:start + bs]
                    Xb, yb = X[mb], y_bin[mb]
                    p_hat = _sigmoid(Xb @ w + b)
                    err   = p_hat - yb
                    w -= self.learning_rate * (Xb.T @ err / len(mb) + w / (self.C * n))
                    b -= self.learning_rate * err.mean()
            self.coef_[k]      = w
            self.intercept_[k] = b

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coef_.ndim == 1:
            return X @ self.coef_ + self.intercept_[0]
        return X @ self.coef_.T + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        if self.coef_.ndim == 1:
            p1 = _sigmoid(scores)
            return np.column_stack([1 - p1, p1])
        return _softmax(scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
