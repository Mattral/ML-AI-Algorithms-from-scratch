"""
Hidden Markov Model (HMM)
==========================
Discrete-observation HMM with the three fundamental algorithms:

1. Forward algorithm    — compute P(observations | model)
2. Viterbi algorithm    — find the most likely hidden state sequence
3. Baum-Welch algorithm — EM to estimate transition, emission, and initial
                          state probabilities from observation sequences

Notation
--------
N  : number of hidden states
M  : number of distinct observation symbols
T  : length of an observation sequence

Parameters
----------
A  : transition matrix        (N, N),  A[i,j] = P(s_t=j | s_{t-1}=i)
B  : emission matrix          (N, M),  B[i,k] = P(o_t=k | s_t=i)
pi : initial state distribution (N,),  pi[i] = P(s_1=i)

All computations are performed in log-space where possible.
Only numpy is used.
"""

import numpy as np


class HiddenMarkovModel:
    """
    Discrete Hidden Markov Model.

    Parameters
    ----------
    n_states : int
        Number of hidden states N.
    n_observations : int
        Size of the observation alphabet M.
    random_state : int or None
        Seed for parameter initialisation.
    """

    def __init__(
        self,
        n_states: int,
        n_observations: int,
        random_state: int | None = None,
    ):
        self.n_states = n_states
        self.n_observations = n_observations
        self.random_state = random_state

        self.A = None    # (N, N)
        self.B = None    # (N, M)
        self.pi = None   # (N,)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_params(self) -> None:
        """Random row-stochastic initialisation."""
        rng = np.random.default_rng(self.random_state)
        N, M = self.n_states, self.n_observations

        A_raw = rng.random((N, N)) + 0.1
        self.A = A_raw / A_raw.sum(axis=1, keepdims=True)

        B_raw = rng.random((N, M)) + 0.1
        self.B = B_raw / B_raw.sum(axis=1, keepdims=True)

        pi_raw = rng.random(N) + 0.1
        self.pi = pi_raw / pi_raw.sum()

    # ------------------------------------------------------------------
    # Forward algorithm  — O(N² T)
    # ------------------------------------------------------------------

    def forward(self, obs: np.ndarray) -> tuple:
        """
        Compute forward variable α_t(i) = P(o_1…o_t, s_t=i | model)
        scaled for numerical stability.

        Returns
        -------
        alpha : ndarray (T, N)
        scales : ndarray (T,)   — scaling coefficients
        log_likelihood : float
        """
        T = len(obs)
        N = self.n_states
        alpha = np.zeros((T, N))
        scales = np.zeros(T)

        alpha[0] = self.pi * self.B[:, obs[0]]
        scales[0] = alpha[0].sum()
        alpha[0] /= scales[0] + 1e-300

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]
            scales[t] = alpha[t].sum()
            alpha[t] /= scales[t] + 1e-300

        log_likelihood = np.sum(np.log(scales + 1e-300))
        return alpha, scales, log_likelihood

    # ------------------------------------------------------------------
    # Backward algorithm
    # ------------------------------------------------------------------

    def backward(self, obs: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """
        Compute backward variable β_t(i), scaled by the same coefficients.

        Returns
        -------
        beta : ndarray (T, N)
        """
        T = len(obs)
        N = self.n_states
        beta = np.zeros((T, N))
        beta[T - 1] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t] = (self.A * self.B[:, obs[t + 1]][np.newaxis, :]) @ beta[t + 1]
            beta[t] /= scales[t + 1] + 1e-300

        return beta

    # ------------------------------------------------------------------
    # Viterbi algorithm
    # ------------------------------------------------------------------

    def viterbi(self, obs: np.ndarray) -> np.ndarray:
        """
        Find the most likely state sequence using the Viterbi algorithm.

        Parameters
        ----------
        obs : ndarray of shape (T,) with integer observations in [0, M)

        Returns
        -------
        states : ndarray of shape (T,)
        """
        T = len(obs)
        N = self.n_states

        log_A = np.log(self.A + 1e-300)
        log_B = np.log(self.B + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)

        delta[0] = log_pi + log_B[:, obs[0]]

        for t in range(1, T):
            trans = delta[t - 1][:, np.newaxis] + log_A  # (N, N)
            psi[t] = np.argmax(trans, axis=0)
            delta[t] = np.max(trans, axis=0) + log_B[:, obs[t]]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    # ------------------------------------------------------------------
    # Baum-Welch (EM)
    # ------------------------------------------------------------------

    def fit(
        self,
        sequences: list,
        n_iter: int = 100,
        tol: float = 1e-4,
    ) -> "HiddenMarkovModel":
        """
        Estimate HMM parameters from observation sequences using Baum-Welch.

        Parameters
        ----------
        sequences : list of 1-D integer arrays
            Each element is one observation sequence.
        n_iter : int
            Maximum EM iterations.
        tol : float
            Convergence tolerance on total log-likelihood change.

        Returns
        -------
        self
        """
        self._init_params()
        N = self.n_states
        M = self.n_observations
        prev_ll = -np.inf

        for _ in range(n_iter):
            # Accumulators
            A_num = np.zeros((N, N))
            B_num = np.zeros((N, M))
            pi_num = np.zeros(N)
            total_ll = 0.0

            for obs in sequences:
                T = len(obs)
                alpha, scales, ll = self.forward(obs)
                beta = self.backward(obs, scales)
                total_ll += ll

                # gamma_t(i) = P(s_t=i | obs, model)
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

                # xi_t(i,j) = P(s_t=i, s_{t+1}=j | obs, model)
                for t in range(T - 1):
                    xi = (
                        alpha[t][:, np.newaxis]
                        * self.A
                        * self.B[:, obs[t + 1]][np.newaxis, :]
                        * beta[t + 1][np.newaxis, :]
                    )
                    xi_sum = xi.sum()
                    xi /= xi_sum + 1e-300
                    A_num += xi

                for t in range(T):
                    B_num[:, obs[t]] += gamma[t]

                pi_num += gamma[0]

            # M-step: normalise
            self.A = A_num / (A_num.sum(axis=1, keepdims=True) + 1e-300)
            self.B = B_num / (B_num.sum(axis=1, keepdims=True) + 1e-300)
            self.pi = pi_num / (pi_num.sum() + 1e-300)

            if abs(total_ll - prev_ll) < tol:
                break
            prev_ll = total_ll

        return self

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def log_likelihood(self, obs: np.ndarray) -> float:
        """Return log P(obs | model)."""
        _, _, ll = self.forward(obs)
        return ll

    def sample(self, length: int, random_state=None) -> tuple:
        """
        Generate a synthetic observation sequence of the given length.

        Returns
        -------
        states : ndarray (length,)
        observations : ndarray (length,)
        """
        rng = np.random.default_rng(random_state)
        states = np.zeros(length, dtype=int)
        observations = np.zeros(length, dtype=int)

        states[0] = rng.choice(self.n_states, p=self.pi)
        observations[0] = rng.choice(self.n_observations, p=self.B[states[0]])

        for t in range(1, length):
            states[t] = rng.choice(self.n_states, p=self.A[states[t - 1]])
            observations[t] = rng.choice(self.n_observations, p=self.B[states[t]])

        return states, observations
