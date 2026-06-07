"""
Gaussian Process Regression (GPR)
===================================
Non-parametric Bayesian regression.  A Gaussian Process defines a
distribution over functions; conditioning on observed data gives a
posterior GP whose mean is used for prediction and whose variance
quantifies uncertainty.

    f ~ GP(0, k(x, x'))
    y = f(x) + ε,  ε ~ N(0, σ_n²)

Posterior predictive:
    μ*   = K(X*, X) [K(X,X) + σ_n² I]⁻¹ y
    Σ*   = K(X*, X*) - K(X*, X) [K(X,X) + σ_n² I]⁻¹ K(X, X*)

Kernels implemented
--------------------
- RBF (Squared Exponential): k(x,x') = σ_f² exp(-||x-x'||²/(2l²))
- Matern52               : k(x,x') = σ_f²(1+√5 r/l + 5r²/(3l²)) exp(-√5 r/l)
- Linear                 : k(x,x') = σ_f² x·x'
- Periodic               : k(x,x') = σ_f² exp(-2 sin²(π|x-x'|/p)/l²)

Only numpy is used.
"""

import numpy as np


# ============================================================
# Kernels
# ============================================================

class RBFKernel:
    """Radial Basis Function (Squared Exponential) kernel."""

    def __init__(self, length_scale: float = 1.0, signal_variance: float = 1.0):
        self.length_scale = length_scale
        self.signal_variance = signal_variance

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1, X2 = np.atleast_2d(X1), np.atleast_2d(X2)
        sq_dist = np.sum(
            (X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2
        )
        return self.signal_variance ** 2 * np.exp(-0.5 * sq_dist / self.length_scale ** 2)


class Matern52Kernel:
    """Matérn 5/2 kernel — rougher than RBF, common for real-world data."""

    def __init__(self, length_scale: float = 1.0, signal_variance: float = 1.0):
        self.length_scale = length_scale
        self.signal_variance = signal_variance

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1, X2 = np.atleast_2d(X1), np.atleast_2d(X2)
        r = np.sqrt(np.sum(
            (X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2
        ))
        sqrt5_r = np.sqrt(5.0) * r / self.length_scale
        return self.signal_variance ** 2 * (
            1.0 + sqrt5_r + sqrt5_r ** 2 / 3.0
        ) * np.exp(-sqrt5_r)


class LinearKernel:
    """Linear (dot-product) kernel."""

    def __init__(self, signal_variance: float = 1.0):
        self.signal_variance = signal_variance

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1, X2 = np.atleast_2d(X1), np.atleast_2d(X2)
        return self.signal_variance ** 2 * (X1 @ X2.T)


class PeriodicKernel:
    """Periodic kernel for modelling repeating patterns."""

    def __init__(
        self,
        length_scale: float = 1.0,
        period: float = 1.0,
        signal_variance: float = 1.0,
    ):
        self.length_scale = length_scale
        self.period = period
        self.signal_variance = signal_variance

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1, X2 = np.atleast_2d(X1), np.atleast_2d(X2)
        # Works for 1-D; use norm for multi-D
        dist = np.sqrt(np.sum(
            (X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2
        ))
        return self.signal_variance ** 2 * np.exp(
            -2.0 * np.sin(np.pi * dist / self.period) ** 2 / self.length_scale ** 2
        )


# ============================================================
# Gaussian Process Regressor
# ============================================================

class GaussianProcessRegressor:
    """
    Gaussian Process Regression.

    Parameters
    ----------
    kernel : callable
        Covariance kernel k(X1, X2).  Defaults to RBFKernel().
    noise_variance : float
        Observation noise σ_n².
    """

    def __init__(self, kernel=None, noise_variance: float = 1e-6):
        self.kernel = kernel if kernel is not None else RBFKernel()
        self.noise_variance = noise_variance
        self.X_train_ = None
        self.alpha_ = None      # (K + σ²I)^{-1} y
        self.L_ = None          # Cholesky factor of (K + σ²I)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cholesky_solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve A x = b via Cholesky decomposition of A."""
        L = np.linalg.cholesky(A)
        # Forward substitution: L z = b
        z = np.linalg.solve(L, b)
        # Back substitution: L^T x = z
        return np.linalg.solve(L.T, z), L

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessRegressor":
        """
        Fit GP to training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples,)
        y : ndarray of shape (n_samples,)
        """
        self.X_train_ = np.atleast_2d(X) if X.ndim == 1 else X
        self.y_train_ = y.copy()

        K = self.kernel(self.X_train_, self.X_train_)
        K_noisy = K + self.noise_variance * np.eye(len(y))

        self.alpha_, self.L_ = self._cholesky_solve(K_noisy, y)
        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ):
        """
        Predictive mean and (optionally) standard deviation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples,)
        return_std : bool

        Returns
        -------
        y_mean : ndarray of shape (n_samples,)
        y_std  : ndarray of shape (n_samples,)  [only if return_std]
        """
        X_ = np.atleast_2d(X) if X.ndim == 1 else X
        K_star = self.kernel(X_, self.X_train_)   # (n_test, n_train)
        y_mean = K_star @ self.alpha_

        if not return_std:
            return y_mean

        # Predictive variance: diag(K** - K*^T (K+σ²I)^{-1} K*)
        v = np.linalg.solve(self.L_, K_star.T)     # (n_train, n_test)
        K_ss = self.kernel(X_, X_)
        var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
        return y_mean, np.sqrt(np.maximum(var, 0.0))

    def sample_posterior(
        self, X: np.ndarray, n_samples: int = 1, random_state=None
    ) -> np.ndarray:
        """
        Draw samples from the posterior distribution.

        Returns
        -------
        samples : ndarray of shape (n_samples, n_test_points)
        """
        rng = np.random.default_rng(random_state)
        X_ = np.atleast_2d(X) if X.ndim == 1 else X
        K_star = self.kernel(X_, self.X_train_)
        K_ss = self.kernel(X_, X_)

        mu = K_star @ self.alpha_
        v = np.linalg.solve(self.L_, K_star.T)
        cov = K_ss - v.T @ v
        # Regularise
        cov += 1e-10 * np.eye(len(mu))
        return rng.multivariate_normal(mu, cov, size=n_samples)
