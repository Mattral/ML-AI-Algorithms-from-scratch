"""
Gaussian Mixture Model (GMM) via Expectation-Maximization
==========================================================
Models data as a weighted mixture of K multivariate Gaussian distributions.
Parameters (means, covariances, mixing weights) are found by iterating the
E-step and M-step until the log-likelihood converges.

E-step : compute responsibilities  r[i,k] = P(z=k | x_i)
M-step : update pi_k, mu_k, Sigma_k using the responsibilities

Only numpy is used.
"""

import numpy as np


class GaussianMixtureModel:
    """
    Gaussian Mixture Model fitted by the EM algorithm.

    Parameters
    ----------
    n_components : int
        Number of mixture components (clusters).
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance on the log-likelihood change.
    reg_covar : float
        Small value added to the diagonal of each covariance matrix for
        numerical stability.
    random_state : int or None
        Seed for reproducible centroid initialisation.
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 100,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        # Learned parameters
        self.weights_ = None    # (K,)
        self.means_ = None      # (K, n_features)
        self.covariances_ = None  # (K, n_features, n_features)
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -np.inf

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _multivariate_gaussian(
        self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the multivariate Gaussian PDF for each row of X.

        Returns
        -------
        pdf : ndarray of shape (n_samples,)
        """
        n_features = X.shape[1]
        diff = X - mean                          # (n, d)
        try:
            cov_inv = np.linalg.inv(cov)
            cov_det = np.linalg.det(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
            cov_det = np.linalg.det(cov + np.eye(n_features) * self.reg_covar)

        cov_det = max(cov_det, 1e-300)           # guard against log(0)
        norm = 1.0 / (np.sqrt((2 * np.pi) ** n_features * cov_det))
        exponent = -0.5 * np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
        return norm * np.exp(exponent)

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        Compute responsibilities r[i, k] = P(z=k | x_i).

        Returns
        -------
        r : ndarray of shape (n_samples, n_components)
        """
        n_samples = X.shape[0]
        r = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            r[:, k] = self.weights_[k] * self._multivariate_gaussian(
                X, self.means_[k], self.covariances_[k]
            )

        row_sums = r.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1e-300, row_sums)
        r /= row_sums
        return r

    def _m_step(self, X: np.ndarray, r: np.ndarray) -> None:
        """Update parameters from responsibilities."""
        n_samples, n_features = X.shape
        N_k = r.sum(axis=0)            # effective number per component

        self.weights_ = N_k / n_samples

        for k in range(self.n_components):
            if N_k[k] < 1e-8:
                continue
            self.means_[k] = (r[:, k] @ X) / N_k[k]

            diff = X - self.means_[k]                        # (n, d)
            weighted_diff = r[:, k:k+1] * diff              # (n, d)
            self.covariances_[k] = (
                weighted_diff.T @ diff / N_k[k]
                + np.eye(n_features) * self.reg_covar
            )

    def _log_likelihood(self, X: np.ndarray) -> float:
        """Compute the log-likelihood of the data under current parameters."""
        n_samples = X.shape[0]
        ll = 0.0
        for i in range(n_samples):
            point_ll = sum(
                self.weights_[k]
                * self._multivariate_gaussian(
                    X[i:i+1], self.means_[k], self.covariances_[k]
                )[0]
                for k in range(self.n_components)
            )
            ll += np.log(max(point_ll, 1e-300))
        return ll

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "GaussianMixtureModel":
        """
        Fit GMM parameters to X via EM.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        K = self.n_components

        # Initialise parameters
        self.weights_ = np.full(K, 1.0 / K)
        # Pick K random data points as initial means
        idx = rng.choice(n_samples, K, replace=False)
        self.means_ = X[idx].copy().astype(float)
        # Identity covariances scaled by data variance
        var = np.var(X, axis=0).mean()
        self.covariances_ = np.array(
            [np.eye(n_features) * var for _ in range(K)]
        )

        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            # E-step
            r = self._e_step(X)
            # M-step
            self._m_step(X, r)
            # Check convergence
            ll = self._log_likelihood(X)
            if abs(ll - prev_ll) < self.tol:
                self.converged_ = True
                break
            prev_ll = ll
            self.n_iter_ = iteration + 1

        self.lower_bound_ = prev_ll
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each sample to the most likely component.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        r = self._e_step(X)
        return np.argmax(r, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return responsibility (soft membership) for each component.

        Returns
        -------
        r : ndarray of shape (n_samples, n_components)
        """
        return self._e_step(X)
