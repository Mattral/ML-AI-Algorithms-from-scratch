"""
Principal Component Analysis (PCA)
====================================
Linear dimensionality reduction via eigen-decomposition of the covariance
matrix.  Projects data onto the directions of maximum variance.

Key steps
---------
1. Centre the data (subtract column means).
2. Compute the covariance matrix  C = X^T X / (n - 1).
3. Eigendecompose C to get eigenvalues and eigenvectors.
4. Sort eigenvectors by descending eigenvalue.
5. Project: X_reduced = X_centered @ W,  where W holds the top-k eigenvectors.

Only numpy is used; no scipy or sklearn.
"""

import numpy as np


class PCA:
    """
    Principal Component Analysis.

    Parameters
    ----------
    n_components : int or None
        Number of components to keep.  If None, all components are kept.
    """

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components
        self.components_ = None        # shape (n_components, n_features)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "PCA":
        """
        Compute principal components from X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape

        # 1. Centre
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Covariance matrix (unbiased, divide by n-1)
        cov = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # 3. Eigen-decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 4. Sort descending by eigenvalue
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]   # columns are eigenvectors

        # 5. Keep top-k
        k = self.n_components if self.n_components is not None else n_features
        self.components_ = eigenvectors[:, :k].T          # (k, n_features)
        self.explained_variance_ = eigenvalues[:k]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_var if total_var > 0
            else np.zeros(k)
        )
        return self

    # ------------------------------------------------------------------
    # Transform / inverse_transform
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X onto the principal components.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and immediately transform X."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        Map data from reduced space back to original feature space
        (approximate reconstruction).
        """
        return np.dot(X_reduced, self.components_) + self.mean_
