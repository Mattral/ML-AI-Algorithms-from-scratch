"""
Hopfield Network
=================
A recurrent, fully-connected network that stores patterns as attractors
of an energy landscape (Hopfield, 1982).  Used as a content-addressable
associative memory.

Storage (Hebbian learning rule)
---------------------------------
For patterns p^(1), ..., p^(M) ∈ {-1, +1}^N:

    W_ij = (1/N) Σ_μ p_i^(μ) p_j^(μ),   W_ii = 0

Energy function
-----------------
    E(s) = -½ Σ_ij W_ij s_i s_j

Recall dynamics
-----------------
Asynchronous update (one neuron at a time, random order) or synchronous
update (all neurons at once):

    s_i ← sign(Σ_j W_ij s_j)

The network converges to a local minimum of E — ideally the nearest
stored pattern, enabling error correction / pattern completion.

Capacity
---------
Approximately 0.138 N patterns can be stored reliably (Amit et al., 1985).

Reference
----------
Hopfield, J. J. (1982). Neural networks and physical systems with emergent
collective computational abilities. PNAS, 79(8), 2554-2558.

Only numpy is used.
"""

from __future__ import annotations

import numpy as np


class HopfieldNetwork:
    """
    Discrete Hopfield Network with bipolar {-1, +1} states.

    Parameters
    ----------
    n_units : int
        Number of neurons (= dimensionality of stored patterns).
    random_state : int or None
        Seed for asynchronous update ordering.
    """

    def __init__(self, n_units: int, random_state: int | None = None) -> None:
        self.n_units = n_units
        self.weights = np.zeros((n_units, n_units))
        self._rng    = np.random.default_rng(random_state)
        self.n_patterns_stored_ = 0

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def fit(self, patterns: np.ndarray) -> "HopfieldNetwork":
        """
        Store patterns via the Hebbian outer-product rule.

        Parameters
        ----------
        patterns : ndarray of shape (n_patterns, n_units)
            Each row is a bipolar pattern with values in {-1, +1}.

        Returns
        -------
        self
        """
        patterns = np.atleast_2d(patterns).astype(float)
        if patterns.shape[1] != self.n_units:
            raise ValueError(
                f"Pattern dimension {patterns.shape[1]} != n_units {self.n_units}."
            )

        self.weights = (patterns.T @ patterns) / self.n_units
        np.fill_diagonal(self.weights, 0.0)
        self.n_patterns_stored_ = len(patterns)
        return self

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def energy(self, state: np.ndarray) -> float:
        """
        Compute E(s) = -½ sᵗ W s.

        Parameters
        ----------
        state : ndarray of shape (n_units,)

        Returns
        -------
        float
        """
        return float(-0.5 * state @ self.weights @ state)

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(
        self,
        state: np.ndarray,
        mode: str = "async",
        max_iter: int = 100,
    ) -> np.ndarray:
        """
        Run network dynamics from an initial state to convergence.

        Parameters
        ----------
        state : ndarray of shape (n_units,)
            Initial state, values in {-1, +1} (or any sign-able reals).
        mode : str
            ``'async'`` — update one randomly-chosen neuron at a time
                          (classic Hopfield dynamics, guaranteed convergence
                          for symmetric W with zero diagonal).
            ``'sync'``  — update all neurons simultaneously each step.
        max_iter : int
            Maximum number of update sweeps.

        Returns
        -------
        ndarray of shape (n_units,) — converged state
        """
        if mode not in {"async", "sync"}:
            raise ValueError("mode must be 'async' or 'sync'.")

        s = np.sign(state.astype(float))
        s[s == 0] = 1.0   # break ties

        if mode == "sync":
            for _ in range(max_iter):
                s_new = np.sign(self.weights @ s)
                s_new[s_new == 0] = 1.0
                if np.array_equal(s_new, s):
                    break
                s = s_new
            return s

        # Asynchronous updates
        for _ in range(max_iter):
            order   = self._rng.permutation(self.n_units)
            changed = False
            for i in order:
                activation = self.weights[i] @ s
                new_val = 1.0 if activation >= 0 else -1.0
                if new_val != s[i]:
                    s[i] = new_val
                    changed = True
            if not changed:
                break

        return s

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def is_stable(self, pattern: np.ndarray) -> bool:
        """
        Check whether ``pattern`` is a fixed point of the dynamics
        (i.e. recall(pattern) == pattern).

        Returns
        -------
        bool
        """
        recalled = self.recall(pattern.copy(), mode="sync", max_iter=1)
        return bool(np.array_equal(recalled, np.sign(pattern)))

    def hamming_distance(self, a: np.ndarray, b: np.ndarray) -> int:
        """Number of differing bipolar units between two states."""
        return int(np.sum(np.sign(a) != np.sign(b)))

    def overlap(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Normalised overlap (similarity) between two bipolar states,
        in [-1, 1].  +1 = identical, -1 = exact inverse, 0 = orthogonal.
        """
        return float(np.dot(np.sign(a), np.sign(b)) / self.n_units)
