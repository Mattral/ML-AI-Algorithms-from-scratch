"""
Bayesian Linear Regression
===========================
Treats the model weights as a distribution, not a point estimate.

Model
-----
    y = X w + ε,   ε ~ N(0, β⁻¹)
    w ~ N(0, α⁻¹ I)

With a Gaussian prior over weights and Gaussian noise, the posterior over
weights is also Gaussian (conjugate prior):

    p(w | X, y) = N(w | m_N, S_N)

    S_N = (α I + β X^T X)^{-1}
    m_N = β S_N X^T y

Predictions are also Gaussian:

    p(y* | x*, X, y) = N(y* | m_N^T x*, σ_N²(x*))
    σ_N²(x*) = β⁻¹ + x*^T S_N x*

Parameters α (weight precision) and β (noise precision) can be fixed or
estimated via type-II maximum likelihood (evidence approximation).

Reference: Bishop, PRML, Chapter 3.
Only numpy is used.
"""

import numpy as np


class BayesianLinearRegression:
    """
    Bayesian Linear Regression with conjugate Gaussian prior.

    Parameters
    ----------
    alpha : float
        Prior precision over weights (1 / prior variance).
    beta : float
        Noise precision (1 / noise variance).
    fit_intercept : bool
        If True, prepend a column of ones to X.
    optimize_hyperparams : bool
        If True, estimate alpha and beta via evidence maximisation
        (iterative re-estimation).  Ignored if False.
    max_iter : int
        Maximum iterations for hyperparameter optimisation.
    tol : float
        Convergence tolerance for hyperparameter optimisation.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        fit_intercept: bool = True,
        optimize_hyperparams: bool = False,
        max_iter: int = 300,
        tol: float = 1e-5,
    ):
        self.alpha = alpha
        self.beta = beta
        self.fit_intercept = fit_intercept
        self.optimize_hyperparams = optimize_hyperparams
        self.max_iter = max_iter
        self.tol = tol

        self.m_N_ = None    # posterior mean  (n_features,)
        self.S_N_ = None    # posterior cov   (n_features, n_features)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            return np.column_stack([np.ones(len(X)), X])
        return X

    def _compute_posterior(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple:
        """Return (m_N, S_N) given current alpha and beta."""
        n_features = X.shape[1]
        S_N_inv = self.alpha * np.eye(n_features) + self.beta * X.T @ X
        S_N = np.linalg.inv(S_N_inv)
        m_N = self.beta * S_N @ X.T @ y
        return m_N, S_N

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLinearRegression":
        """
        Compute posterior distribution over weights.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
        """
        X_ = self._add_bias(X)
        n_samples, n_features = X_.shape

        if self.optimize_hyperparams:
            # Evidence approximation (Bishop PRML §3.5.2)
            alpha, beta = self.alpha, self.beta
            for _ in range(self.max_iter):
                m_N, S_N = self._compute_posterior(X_, y)
                # Eigenvalues of β X^T X
                eigvals = np.linalg.eigvalsh(beta * X_.T @ X_)
                gamma = np.sum(eigvals / (alpha + eigvals))

                alpha_new = gamma / (m_N @ m_N)
                residuals = y - X_ @ m_N
                beta_new = (n_samples - gamma) / (residuals @ residuals)

                alpha_new = max(alpha_new, 1e-10)
                beta_new = max(beta_new, 1e-10)

                if abs(alpha_new - alpha) < self.tol and abs(beta_new - beta) < self.tol:
                    alpha, beta = alpha_new, beta_new
                    break
                alpha, beta = alpha_new, beta_new

            self.alpha, self.beta = alpha, beta

        self.m_N_, self.S_N_ = self._compute_posterior(X_, y)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        """
        Predictive mean (and optionally std) for new inputs.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        return_std : bool
            If True, also return the predictive standard deviation.

        Returns
        -------
        y_mean : ndarray of shape (n_samples,)
        y_std  : ndarray of shape (n_samples,)  [only if return_std=True]
        """
        X_ = self._add_bias(X)
        y_mean = X_ @ self.m_N_

        if not return_std:
            return y_mean

        # Predictive variance: β⁻¹ + x^T S_N x
        var = (1.0 / self.beta) + np.einsum("ij,jk,ik->i", X_, self.S_N_, X_)
        return y_mean, np.sqrt(np.maximum(var, 0.0))

    @property
    def coef_(self) -> np.ndarray:
        """Posterior mean weights (excluding bias if fit_intercept=True)."""
        if self.fit_intercept:
            return self.m_N_[1:]
        return self.m_N_

    @property
    def intercept_(self) -> float:
        if self.fit_intercept:
            return float(self.m_N_[0])
        return 0.0
