"""
Restricted Boltzmann Machine (RBM)
====================================
A bipartite, undirected probabilistic graphical model with visible units
v and hidden units h, trained to model the joint distribution P(v, h)
via an energy function (Smolensky, 1986; Hinton, 2002).

Energy function
-----------------
    E(v, h) = -aᵗv - bᵗh - vᵗWh

Conditional distributions (Gibbs sampling)
---------------------------------------------
    P(h_j=1 | v) = σ(b_j + Σ_i v_i W_ij)
    P(v_i=1 | h) = σ(a_i + Σ_j h_j W_ij)

Training — Contrastive Divergence (CD-k)
-------------------------------------------
1. v⁰ = data
2. h⁰ ~ P(h | v⁰)
3. Repeat k times:  v¹ ~ P(v | h⁰),  h¹ ~ P(h | v¹)
4. Update:
       ΔW ∝ ⟨v⁰h⁰ᵗ⟩ - ⟨v^k h^kᵗ⟩
       Δa ∝ v⁰ - v^k
       Δb ∝ h⁰ - h^k

Free energy
-------------
    F(v) = -aᵗv - Σ_j log(1 + exp(b_j + (Wᵗv)_j))

Used as a building block for Deep Belief Networks and for unsupervised
feature learning / dimensionality reduction.

Reference
----------
Hinton, G. E. (2002). Training products of experts by minimizing
contrastive divergence. Neural Computation, 14(8), 1771-1800.

Only numpy is used.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class RestrictedBoltzmannMachine:
    """
    Restricted Boltzmann Machine trained with Contrastive Divergence.

    Parameters
    ----------
    n_visible : int
    n_hidden : int
    learning_rate : float
    cd_k : int
        Number of Gibbs sampling steps for CD-k (default 1, i.e. CD-1).
    epochs : int
    batch_size : int or None
    random_state : int or None
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        learning_rate: float = 0.05,
        cd_k: int = 1,
        epochs: int = 50,
        batch_size: int | None = 32,
        random_state: int | None = None,
    ) -> None:
        self.n_visible     = n_visible
        self.n_hidden      = n_hidden
        self.learning_rate = learning_rate
        self.cd_k          = cd_k
        self.epochs        = epochs
        self.batch_size    = batch_size
        self._rng          = np.random.default_rng(random_state)

        scale = 0.01
        self.W = self._rng.normal(0, scale, (n_visible, n_hidden))
        self.a = np.zeros(n_visible)    # visible bias
        self.b = np.zeros(n_hidden)     # hidden bias

        self.reconstruction_errors_: list[float] = []

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_hidden(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (P(h=1|v), sampled h)."""
        p_h = _sigmoid(v @ self.W + self.b)
        h   = (self._rng.random(p_h.shape) < p_h).astype(float)
        return p_h, h

    def _sample_visible(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (P(v=1|h), sampled v)."""
        p_v = _sigmoid(h @ self.W.T + self.a)
        v   = (self._rng.random(p_v.shape) < p_v).astype(float)
        return p_v, v

    # ------------------------------------------------------------------
    # Training — Contrastive Divergence (CD-k)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "RestrictedBoltzmannMachine":
        """
        Train the RBM on binary (or [0,1]-valued) data X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_visible)

        Returns
        -------
        self
        """
        n = len(X)
        bs = self.batch_size or n
        self.reconstruction_errors_ = []

        for _ in range(self.epochs):
            idx = self._rng.permutation(n)
            epoch_err = 0.0
            n_batches = 0

            for start in range(0, n, bs):
                mb = idx[start:start + bs]
                v0 = X[mb]
                B  = len(v0)

                # Positive phase
                p_h0, h0 = self._sample_hidden(v0)

                # Gibbs chain for CD-k
                v_k, h_k = v0, h0
                for _ in range(self.cd_k):
                    p_vk, v_k = self._sample_visible(h_k)
                    p_hk, h_k = self._sample_hidden(v_k)

                # Gradient (use probabilities, not samples, for hidden — standard trick)
                pos_assoc = v0.T @ p_h0
                neg_assoc = v_k.T @ p_hk

                self.W += self.learning_rate * (pos_assoc - neg_assoc) / B
                self.a += self.learning_rate * (v0 - v_k).mean(axis=0)
                self.b += self.learning_rate * (p_h0 - p_hk).mean(axis=0)

                epoch_err += float(np.mean((v0 - v_k) ** 2))
                n_batches += 1

            self.reconstruction_errors_.append(epoch_err / n_batches)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute hidden-unit activation probabilities P(h=1|v).

        Returns
        -------
        ndarray of shape (n_samples, n_hidden)
        """
        return _sigmoid(X @ self.W + self.b)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        One Gibbs step v → h → v̂ (probabilities, not samples).

        Returns
        -------
        ndarray of shape (n_samples, n_visible)
        """
        p_h = self.transform(X)
        return _sigmoid(p_h @ self.W.T + self.a)

    def free_energy(self, X: np.ndarray) -> np.ndarray:
        """
        Free energy F(v) = -aᵗv - Σ_j log(1 + exp(b_j + (Wᵗv)_j)).
        Lower free energy ⇒ more "typical" under the model.

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        vbias_term = X @ self.a
        wx_b       = X @ self.W + self.b
        hidden_term = np.sum(np.log1p(np.exp(wx_b)), axis=1)
        return -vbias_term - hidden_term

    def sample(
        self,
        n_samples: int,
        n_gibbs_steps: int = 1000,
        v_init: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Generate samples via Gibbs sampling from a random (or given) start.

        Parameters
        ----------
        n_samples : int
        n_gibbs_steps : int
            Number of full Gibbs sweeps per chain.
        v_init : ndarray of shape (n_samples, n_visible) or None
            Initial visible state.  If None, random binary init.

        Returns
        -------
        ndarray of shape (n_samples, n_visible)
        """
        if v_init is None:
            v = (self._rng.random((n_samples, self.n_visible)) < 0.5).astype(float)
        else:
            v = v_init.copy()

        for _ in range(n_gibbs_steps):
            _, h = self._sample_hidden(v)
            _, v = self._sample_visible(h)

        return v
