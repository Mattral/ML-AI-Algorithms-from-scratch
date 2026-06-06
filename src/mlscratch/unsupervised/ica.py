"""
Independent Component Analysis (ICA) — FastICA
================================================
Separates a multivariate signal into additive, statistically independent
components (blind source separation).  This is the FastICA algorithm using
a fixed-point iteration to maximise non-Gaussianity (measured via kurtosis
or negentropy via the logcosh or exp contrast functions).

Steps
-----
1. Whiten X: remove correlations so components have unit variance.
2. For each component, run a fixed-point update on a weight vector w:
       w ← E[X g(w^T X)] − E[g'(w^T X)] w
   then normalise w and orthogonalise against previous components.
3. The independent components are  S = W X_white.

Only numpy is used.
"""

import numpy as np


class FastICA:
    """
    FastICA — Independent Component Analysis.

    Parameters
    ----------
    n_components : int or None
        Number of independent components to extract.
        If None, uses min(n_samples, n_features).
    max_iter : int
        Maximum iterations per component.
    tol : float
        Convergence tolerance (change in w between iterations).
    fun : str
        Contrast function: 'logcosh' (default) or 'exp'.
    random_state : int or None
        Seed for weight initialisation.
    """

    def __init__(
        self,
        n_components: int | None = None,
        max_iter: int = 200,
        tol: float = 1e-4,
        fun: str = "logcosh",
        random_state: int | None = None,
    ):
        if fun not in {"logcosh", "exp"}:
            raise ValueError("fun must be 'logcosh' or 'exp'.")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.fun = fun
        self.random_state = random_state

        self.components_ = None    # (n_components, n_features) unmixing matrix
        self.mixing_ = None        # pseudo-inverse of components_
        self.mean_ = None          # for centering
        self.whitening_ = None     # whitening matrix

    # ------------------------------------------------------------------
    # Contrast functions (g) and their derivatives (g')
    # ------------------------------------------------------------------

    def _g_and_gprime(self, u: np.ndarray):
        """Return g(u) and g'(u) element-wise for the chosen contrast."""
        if self.fun == "logcosh":
            g = np.tanh(u)
            g_prime = 1.0 - g ** 2
        else:  # exp
            exp_u = np.exp(-0.5 * u ** 2)
            g = u * exp_u
            g_prime = (1.0 - u ** 2) * exp_u
        return g, g_prime

    # ------------------------------------------------------------------
    # Whitening
    # ------------------------------------------------------------------

    def _whiten(self, X: np.ndarray):
        """
        Whiten data: zero-mean, unit variance, uncorrelated.

        Returns
        -------
        X_white : ndarray (n_features, n_samples)  — note transposed
        W_white : whitening matrix
        """
        # X is (n_samples, n_features); work in (n_features, n_samples) form
        X_c = X.T                       # (d, n)
        cov = np.cov(X_c, rowvar=True)  # (d, d)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Guard against near-zero eigenvalues
        eigvals = np.maximum(eigvals, 1e-10)
        W_white = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        X_white = W_white @ X_c         # (d, n)
        return X_white, W_white

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "FastICA":
        """
        Fit ICA on X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        n_components = self.n_components or min(n_samples, n_features)

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        X_white, self.whitening_ = self._whiten(X_centered)  # (d, n)

        W = np.zeros((n_components, n_features))  # unmixing in white space

        for p in range(n_components):
            w = rng.standard_normal(n_features)
            w /= np.linalg.norm(w) + 1e-12

            for _ in range(self.max_iter):
                u = w @ X_white                     # (n,)
                g_u, g_prime_u = self._g_and_gprime(u)

                # Fixed-point update
                w_new = (X_white * g_u).mean(axis=1) - g_prime_u.mean() * w

                # Deflation: subtract projections onto previous components
                for j in range(p):
                    w_new -= (w_new @ W[j]) * W[j]

                w_new /= np.linalg.norm(w_new) + 1e-12

                if abs(abs(w_new @ w) - 1.0) < self.tol:
                    w = w_new
                    break
                w = w_new

            W[p] = w

        # Recover components in original (non-whitened) space
        self.components_ = W @ self.whitening_   # (n_components, n_features)
        self.mixing_ = np.linalg.pinv(self.components_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Recover independent components from X.

        Returns
        -------
        S : ndarray of shape (n_samples, n_components)
        """
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and return independent components."""
        self.fit(X)
        return self.transform(X)
