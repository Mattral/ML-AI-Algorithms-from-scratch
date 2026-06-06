"""
t-Distributed Stochastic Neighbour Embedding (t-SNE)
=====================================================
Non-linear dimensionality reduction that preserves local structure.
Converts high-dimensional Euclidean distances into conditional
probabilities (Gaussian in high-dim, Student-t in low-dim) and minimises
the KL divergence between the two distributions via gradient descent.

Key steps
---------
1. Compute pairwise affinities p_{j|i} in the high-dimensional space using
   a Gaussian kernel; perplexity controls the effective number of neighbours.
2. Symmetrise: p_{ij} = (p_{j|i} + p_{i|j}) / 2n.
3. Initialise low-dimensional embedding Y randomly.
4. Compute q_{ij} in Y using a Student-t kernel (df=1).
5. Gradient descent on KL(P || Q) with momentum.

Reference: van der Maaten & Hinton (2008).
Only numpy is used.
"""

import numpy as np


class TSNE:
    """
    t-SNE dimensionality reduction.

    Parameters
    ----------
    n_components : int
        Dimension of the embedding (almost always 2 or 3).
    perplexity : float
        Effective number of neighbours; typical values 5–50.
    n_iter : int
        Number of gradient-descent iterations.
    learning_rate : float
        Step size for gradient descent.
    momentum : float
        Momentum coefficient for gradient updates.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        n_iter: int = 1000,
        learning_rate: float = 200.0,
        momentum: float = 0.9,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.random_state = random_state
        self.embedding_ = None

    # ------------------------------------------------------------------
    # High-dimensional affinities
    # ------------------------------------------------------------------

    def _pairwise_sq_distances(self, X: np.ndarray) -> np.ndarray:
        """Return matrix of squared Euclidean distances."""
        sum_sq = np.sum(X ** 2, axis=1, keepdims=True)
        D_sq = sum_sq + sum_sq.T - 2.0 * (X @ X.T)
        np.fill_diagonal(D_sq, 0.0)
        return np.maximum(D_sq, 0.0)

    def _conditional_probabilities(
        self, D_sq: np.ndarray, sigma: float, i: int
    ) -> np.ndarray:
        """Compute p_{j|i} for a given bandwidth sigma."""
        d = D_sq[i].copy()
        d[i] = np.inf                  # exclude self
        exp_d = np.exp(-d / (2.0 * sigma ** 2))
        denom = exp_d.sum()
        return exp_d / (denom + 1e-12)

    def _binary_search_sigma(
        self, D_sq: np.ndarray, i: int, target_perp: float,
        tol: float = 1e-5, max_iter: int = 50
    ) -> float:
        """Find sigma_i such that perplexity(p_{.|i}) == target_perp."""
        sigma_low, sigma_high = 1e-10, 1e5
        sigma = 1.0

        for _ in range(max_iter):
            p = self._conditional_probabilities(D_sq, sigma, i)
            # Shannon entropy
            p_safe = np.maximum(p, 1e-12)
            H = -np.sum(p_safe * np.log2(p_safe))
            perp = 2.0 ** H

            if abs(perp - target_perp) < tol:
                break
            if perp < target_perp:
                sigma_low = sigma
                sigma = (sigma + sigma_high) / 2.0
            else:
                sigma_high = sigma
                sigma = (sigma + sigma_low) / 2.0

        return sigma

    def _compute_P(self, X: np.ndarray) -> np.ndarray:
        """Compute symmetric joint probabilities P."""
        n = len(X)
        D_sq = self._pairwise_sq_distances(X)
        P = np.zeros((n, n))

        for i in range(n):
            sigma = self._binary_search_sigma(D_sq, i, self.perplexity)
            P[i] = self._conditional_probabilities(D_sq, sigma, i)

        # Symmetrise and normalise
        P = (P + P.T) / (2.0 * n)
        P = np.maximum(P, 1e-12)
        return P

    # ------------------------------------------------------------------
    # Low-dimensional affinities
    # ------------------------------------------------------------------

    def _compute_Q(self, Y: np.ndarray) -> tuple:
        """
        Compute Student-t affinities in the embedding.

        Returns
        -------
        Q : normalised affinities
        num : unnormalised numerator (needed for gradient)
        """
        D_sq = self._pairwise_sq_distances(Y)
        num = 1.0 / (1.0 + D_sq)
        np.fill_diagonal(num, 0.0)
        denom = num.sum()
        Q = num / (denom + 1e-12)
        Q = np.maximum(Q, 1e-12)
        return Q, num

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit t-SNE and return 2-D (or n_components-D) embedding.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        Y : ndarray of shape (n_samples, n_components)
        """
        rng = np.random.default_rng(self.random_state)
        n = len(X)

        # Step 1: compute high-dim affinities
        P = self._compute_P(X)
        # Early exaggeration (first 250 iters)
        P_exag = P * 4.0

        # Step 2: random initialisation of embedding
        Y = rng.standard_normal((n, self.n_components)) * 1e-4
        velocity = np.zeros_like(Y)

        for t in range(self.n_iter):
            p_use = P_exag if t < 250 else P
            Q, num = self._compute_Q(Y)

            # Gradient of KL divergence
            PQ_diff = p_use - Q           # (n, n)
            grad = np.zeros_like(Y)
            for i in range(n):
                # dC/dY_i = 4 * sum_j (p_ij - q_ij) * (y_i - y_j) * (1 + ||y_i-y_j||^2)^-1
                diff = Y[i] - Y           # (n, n_components)
                grad[i] = 4.0 * (PQ_diff[i] * num[i] @ diff.reshape(n, -1)).sum(axis=0) \
                    if self.n_components == 1 \
                    else 4.0 * np.dot(PQ_diff[i] * num[i], diff)

            # Momentum update
            velocity = self.momentum * velocity - self.learning_rate * grad
            Y = Y + velocity

            # Centre embedding
            Y -= Y.mean(axis=0)

        self.embedding_ = Y
        return Y

    def fit(self, X: np.ndarray) -> "TSNE":
        """Fit t-SNE (embedding stored in self.embedding_)."""
        self.fit_transform(X)
        return self
