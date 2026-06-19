"""
Radial Basis Function (RBF) Network
=====================================
A two-layer network where the hidden layer applies radial basis
functions (typically Gaussians) centred on prototype points, and the
output layer is a linear combination of hidden activations.

Architecture
-------------
    φ_j(x) = exp(-‖x - c_j‖² / (2σ_j²))      hidden layer (RBF units)
    ŷ(x)   = Σ_j w_j φ_j(x) + b               output layer (linear)

Training (two-stage, classic RBF approach)
---------------------------------------------
1. **Centre selection**: choose RBF centres c_j via k-means clustering
   on the training inputs (or random subsampling).
2. **Width selection**: σ_j set to the mean distance to the nearest
   neighbouring centre, scaled by a width factor.
3. **Output weights**: solve the linear least-squares problem
       Φ w = y
   in closed form via the pseudo-inverse, where Φ_ij = φ_j(x_i).

This closed-form approach avoids backpropagation for the output layer
entirely — only the (optional) k-means step involves iteration.

Supports both regression and classification (one-hot targets + argmax).

Reference
----------
Broomhead, D. S., & Lowe, D. (1988). Radial basis functions, multi-variable
functional interpolation and adaptive networks. RSRE Memorandum.

Only numpy is used.
"""

from __future__ import annotations

import numpy as np


class RBFNetwork:
    """
    Radial Basis Function Network.

    Parameters
    ----------
    n_centers : int
        Number of RBF units (hidden neurons).
    width_scaling : float
        Multiplier applied to the mean nearest-centre distance to obtain
        each unit's σ.  Larger → smoother, wider basis functions.
    task : str
        ``'regression'`` or ``'classification'``.
    n_classes : int
        Number of classes (classification only).
    ridge : float
        L2 regularisation added to the normal equations
        (Φᵗ Φ + ridge·I) w = Φᵗ y  — improves numerical stability.
    random_state : int or None
    """

    def __init__(
        self,
        n_centers: int = 10,
        width_scaling: float = 1.0,
        task: str = "regression",
        n_classes: int = 2,
        ridge: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        if task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'.")
        self.n_centers     = n_centers
        self.width_scaling = width_scaling
        self.task          = task
        self.n_classes     = n_classes
        self.ridge         = ridge
        self._rng          = np.random.default_rng(random_state)

        self.centers_: np.ndarray | None = None
        self.sigmas_:  np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Centre / width selection
    # ------------------------------------------------------------------

    def _select_centers(self, X: np.ndarray) -> np.ndarray:
        """
        Simple k-means clustering to select RBF centres.

        Falls back to random sampling of distinct points if
        n_centers >= n_samples.
        """
        n_samples = len(X)
        k = min(self.n_centers, n_samples)

        # Initialise centres from random data points
        idx = self._rng.choice(n_samples, k, replace=False)
        centers = X[idx].astype(float).copy()

        for _ in range(50):   # k-means iterations
            # Assign
            dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            labels = np.argmin(dists, axis=1)

            new_centers = centers.copy()
            for j in range(k):
                members = X[labels == j]
                if len(members) > 0:
                    new_centers[j] = members.mean(axis=0)

            if np.allclose(new_centers, centers):
                break
            centers = new_centers

        return centers

    def _compute_widths(self, centers: np.ndarray) -> np.ndarray:
        """σ_j = width_scaling × (mean distance to nearest other centre)."""
        k = len(centers)
        if k == 1:
            return np.array([1.0])

        dists = np.linalg.norm(
            centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2
        )
        np.fill_diagonal(dists, np.inf)
        nearest = dists.min(axis=1)
        sigmas = self.width_scaling * nearest
        sigmas[sigmas == 0] = 1.0   # guard against duplicate centres
        return sigmas

    # ------------------------------------------------------------------
    # Feature mapping
    # ------------------------------------------------------------------

    def _rbf_features(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Φ_ij = exp(-‖x_i - c_j‖² / (2σ_j²)).

        Returns
        -------
        ndarray of shape (n_samples, n_centers)
        """
        dists_sq = np.sum(
            (X[:, np.newaxis, :] - self.centers_[np.newaxis, :, :]) ** 2, axis=2
        )
        return np.exp(-dists_sq / (2.0 * self.sigmas_ ** 2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RBFNetwork":
        """
        Fit the RBF network.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
            Continuous targets (regression) or integer class labels
            (classification).

        Returns
        -------
        self
        """
        self.centers_ = self._select_centers(X)
        self.sigmas_  = self._compute_widths(self.centers_)

        Phi = self._rbf_features(X)                       # (n, n_centers)
        Phi_bias = np.column_stack([Phi, np.ones(len(Phi))])  # add bias column

        if self.task == "classification":
            n = len(y)
            Y = np.zeros((n, self.n_classes))
            Y[np.arange(n), y.astype(int)] = 1.0
        else:
            Y = y.reshape(-1, 1).astype(float)

        # Ridge-regularised normal equations: (ΦᵗΦ + λI) w = Φᵗ Y
        A = Phi_bias.T @ Phi_bias + self.ridge * np.eye(Phi_bias.shape[1])
        B = Phi_bias.T @ Y
        self.weights_ = np.linalg.solve(A, B)              # (n_centers+1, n_outputs)

        return self

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        Phi = self._rbf_features(X)
        Phi_bias = np.column_stack([Phi, np.ones(len(Phi))])
        return Phi_bias @ self.weights_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict targets (regression) or class labels (classification).

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        out = self._predict_raw(X)
        if self.task == "classification":
            return np.argmax(out, axis=1)
        return out.ravel()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return softmax-normalised class scores (classification only).

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification.")
        raw = self._predict_raw(X)
        e   = np.exp(raw - raw.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Return the RBF feature representation Φ(X).

        Returns
        -------
        ndarray of shape (n_samples, n_centers)
        """
        return self._rbf_features(X)
