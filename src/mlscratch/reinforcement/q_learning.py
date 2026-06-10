"""
Q-Learning
===========
Tabular and linear-function-approximation variants of the classic
off-policy TD control algorithm (Watkins & Dayan, 1992).

Tabular Q-Learning
------------------
Maintains a Q-table Q[s, a] and updates via:

    Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]

Supports ε-greedy exploration with optional linear or exponential decay.

Linear Q-Learning (Linear Function Approximation)
--------------------------------------------------
Represents Q(s,a) = φ(s,a)^T w where φ is a hand-crafted feature
vector and w are learned weights — useful for larger state spaces.

Both classes follow the same fit() / predict_action() API and expose
episode-level training via train_episode().

Only numpy and Python stdlib are used.
"""

from __future__ import annotations
import numpy as np


# ============================================================
# Tabular Q-Learning
# ============================================================

class QLearning:
    """
    Tabular Q-Learning agent.

    Parameters
    ----------
    n_states : int
    n_actions : int
    alpha : float          learning rate
    gamma : float          discount factor
    epsilon : float        initial exploration probability
    epsilon_min : float    minimum exploration probability
    epsilon_decay : float  multiplicative decay per episode
    random_state : int | None
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        random_state: int | None = None,
    ):
        self.n_states     = n_states
        self.n_actions    = n_actions
        self.alpha        = alpha
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng         = np.random.default_rng(random_state)

        # Q-table initialised to zeros
        self.Q: np.ndarray = np.zeros((n_states, n_actions))

        # Episode-level tracking
        self.episode_rewards_: list[float] = []
        self.epsilons_: list[float] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: int, greedy: bool = False) -> int:
        """ε-greedy action selection."""
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        return int(np.argmax(self.Q[state]))

    # ------------------------------------------------------------------
    # Single update step
    # ------------------------------------------------------------------

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """
        Apply one Q-learning update.

        Returns
        -------
        td_error : float
        """
        target = reward if done else reward + self.gamma * np.max(self.Q[next_state])
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return float(td_error)

    # ------------------------------------------------------------------
    # Episode training
    # ------------------------------------------------------------------

    def train_episode(self, env) -> float:
        """
        Run one full episode and return total reward.

        Parameters
        ----------
        env : object with .reset() → int and .step(a) → (int, float, bool)
        """
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = self.select_action(state)
            next_state, reward, done = env.step(action)
            self.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_rewards_.append(total_reward)
        self.epsilons_.append(self.epsilon)
        return total_reward

    def train(self, env, n_episodes: int) -> "QLearning":
        """Train for n_episodes episodes."""
        for _ in range(n_episodes):
            self.train_episode(env)
        return self

    # ------------------------------------------------------------------
    # Value / policy helpers
    # ------------------------------------------------------------------

    def value_function(self) -> np.ndarray:
        """V(s) = max_a Q(s,a)  for all states."""
        return self.Q.max(axis=1)

    def policy(self) -> np.ndarray:
        """Greedy policy: π(s) = argmax_a Q(s,a)."""
        return self.Q.argmax(axis=1)


# ============================================================
# Double Q-Learning
# ============================================================

class DoubleQLearning:
    """
    Double Q-Learning (van Hasselt, 2010).

    Maintains two independent Q-tables Q_A and Q_B.
    On each step, one is selected at random for the update, using the
    other to evaluate the greedy action — removing maximisation bias.

    Same API as QLearning.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        random_state: int | None = None,
    ):
        self.n_states     = n_states
        self.n_actions    = n_actions
        self.alpha        = alpha
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng         = np.random.default_rng(random_state)

        self.Q_A: np.ndarray = np.zeros((n_states, n_actions))
        self.Q_B: np.ndarray = np.zeros((n_states, n_actions))

        self.episode_rewards_: list[float] = []
        self.epsilons_: list[float] = []

    @property
    def Q(self) -> np.ndarray:
        """Combined Q estimate (average of both tables)."""
        return (self.Q_A + self.Q_B) / 2.0

    def select_action(self, state: int, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        return int(np.argmax(self.Q[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        if self._rng.random() < 0.5:
            # Update A, evaluate with B
            a_star = int(np.argmax(self.Q_A[next_state]))
            target = reward if done else reward + self.gamma * self.Q_B[next_state, a_star]
            td_error = target - self.Q_A[state, action]
            self.Q_A[state, action] += self.alpha * td_error
        else:
            # Update B, evaluate with A
            a_star = int(np.argmax(self.Q_B[next_state]))
            target = reward if done else reward + self.gamma * self.Q_A[next_state, a_star]
            td_error = target - self.Q_B[state, action]
            self.Q_B[state, action] += self.alpha * td_error
        return float(td_error)

    def train_episode(self, env) -> float:
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = self.select_action(state)
            next_state, reward, done = env.step(action)
            self.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_rewards_.append(total_reward)
        self.epsilons_.append(self.epsilon)
        return total_reward

    def train(self, env, n_episodes: int) -> "DoubleQLearning":
        for _ in range(n_episodes):
            self.train_episode(env)
        return self

    def value_function(self) -> np.ndarray:
        return self.Q.max(axis=1)

    def policy(self) -> np.ndarray:
        return self.Q.argmax(axis=1)


# ============================================================
# Linear Function Approximation Q-Learning
# ============================================================

class LinearQLearning:
    """
    Q-Learning with linear function approximation.

    Q(s, a) ≈ φ(s, a)^T w

    Feature construction: one-hot state × one-hot action tiling.
    Works with integer state/action spaces.

    Parameters
    ----------
    n_states : int
    n_actions : int
    alpha, gamma, epsilon, epsilon_min, epsilon_decay : see QLearning
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.01,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        random_state: int | None = None,
    ):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng      = np.random.default_rng(random_state)

        self.n_features = n_states * n_actions
        self.w = np.zeros(self.n_features)

        self.episode_rewards_: list[float] = []

    def _features(self, state: int, action: int) -> np.ndarray:
        """One-hot feature vector for (state, action) pair."""
        phi = np.zeros(self.n_features)
        phi[state * self.n_actions + action] = 1.0
        return phi

    def _q(self, state: int, action: int) -> float:
        return float(self.w @ self._features(state, action))

    def select_action(self, state: int, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        q_vals = [self._q(state, a) for a in range(self.n_actions)]
        return int(np.argmax(q_vals))

    def update(self, state, action, reward, next_state, done) -> float:
        q_next = max(self._q(next_state, a) for a in range(self.n_actions))
        target = reward if done else reward + self.gamma * q_next
        td_error = target - self._q(state, action)
        self.w += self.alpha * td_error * self._features(state, action)
        return float(td_error)

    def train_episode(self, env) -> float:
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = self.select_action(state)
            next_state, reward, done = env.step(action)
            self.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_rewards_.append(total_reward)
        return total_reward

    def train(self, env, n_episodes: int) -> "LinearQLearning":
        for _ in range(n_episodes):
            self.train_episode(env)
        return self

    @property
    def Q(self) -> np.ndarray:
        """Recover Q-table from weight vector."""
        Q = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                Q[s, a] = self._q(s, a)
        return Q
