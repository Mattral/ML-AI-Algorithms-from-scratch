"""
mlscratch.reinforcement.utils
==============================
Shared infrastructure used by all RL algorithms:

  - GridWorld          : discrete 4-action environment (tabular RL)
  - ContinuousEnv      : continuous state/action environment for deep RL
  - ReplayBuffer       : experience replay with uniform random sampling
  - PrioritizedReplayBuffer : sum-tree prioritised experience replay (DQN+)
  - MLP                : multi-layer perceptron (forward + backward)
  - OrnsteinUhlenbeckNoise : temporally-correlated exploration noise (DDPG)
  - GaussianNoise      : i.i.d. Gaussian exploration noise

Only numpy and Python stdlib are used throughout.
"""

from __future__ import annotations
import numpy as np
from collections import deque


# ============================================================
# Environments
# ============================================================

class GridWorld:
    """
    Simple deterministic grid-world for tabular RL.

    Layout  (default 4×4):
        S . . .
        . # . .
        . . . .
        . . . G

    S = start (0,0), G = goal (3,3), # = pit (1,1).
    Actions: 0=up, 1=down, 2=left, 3=right.
    Rewards: +10 (goal), -10 (pit), -0.1 (step).
    Episode ends on goal or pit.
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    N_ACTIONS = 4

    def __init__(self, size: int = 4, pit: tuple = (1, 1)):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.pit  = pit
        self._state: tuple | None = None

    # ------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        return self.size * self.size

    def _encode(self, pos: tuple) -> int:
        return pos[0] * self.size + pos[1]

    def reset(self) -> int:
        self._state = (0, 0)
        return self._encode(self._state)

    def step(self, action: int) -> tuple[int, float, bool]:
        r, c = self._state
        dr, dc = self.ACTIONS[action]
        nr = max(0, min(self.size - 1, r + dr))
        nc = max(0, min(self.size - 1, c + dc))
        self._state = (nr, nc)
        s = self._encode(self._state)

        if self._state == self.goal:
            return s, 10.0, True
        if self._state == self.pit:
            return s, -10.0, True
        return s, -0.1, False

    def render(self) -> str:
        rows = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                pos = (r, c)
                if pos == self._state:
                    row.append("A")
                elif pos == self.goal:
                    row.append("G")
                elif pos == self.pit:
                    row.append("#")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)


class ContinuousEnv:
    """
    Lightweight continuous control environment — 1-D point mass.

    State  : [position, velocity]  ∈ ℝ²
    Action : force ∈ [-1, 1]       (clipped)
    Goal   : drive position to 0 with velocity 0.

    Reward : -(position² + 0.1 velocity² + 0.001 force²)
    Episode terminates after `max_steps` steps.

    This is a minimal stand-in for MuJoCo-style envs; used for testing
    deep RL algorithms without external dependencies.
    """

    STATE_DIM  = 2
    ACTION_DIM = 1
    ACTION_LOW  = -1.0
    ACTION_HIGH =  1.0

    def __init__(self, max_steps: int = 200):
        self.max_steps = max_steps
        self._state: np.ndarray | None = None
        self._t: int = 0

    def reset(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        self._state = rng.uniform([-0.5, -0.2], [0.5, 0.2])
        self._t = 0
        return self._state.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        a = float(np.clip(np.asarray(action).ravel()[0], self.ACTION_LOW, self.ACTION_HIGH))
        pos, vel = self._state
        # Simple Euler integration: m=1, drag=0.1
        acc  = a - 0.1 * vel
        vel  = vel + 0.05 * acc
        pos  = pos + 0.05 * vel
        pos  = float(np.clip(pos, -2.0, 2.0))
        vel  = float(np.clip(vel, -2.0, 2.0))
        self._state = np.array([pos, vel])
        reward = -(pos**2 + 0.1 * vel**2 + 0.001 * a**2)
        self._t += 1
        done = self._t >= self.max_steps
        return self._state.copy(), reward, done

    @property
    def state_dim(self) -> int:
        return self.STATE_DIM

    @property
    def action_dim(self) -> int:
        return self.ACTION_DIM


class DiscreteEnv:
    """
    Discrete-action wrapper around ContinuousEnv for DQN testing.
    Actions: {−1.0, −0.5, 0.0, +0.5, +1.0}
    """

    DISCRETE_ACTIONS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    N_ACTIONS = 5
    STATE_DIM = 2

    def __init__(self, max_steps: int = 200):
        self._env = ContinuousEnv(max_steps)

    def reset(self, rng=None) -> np.ndarray:
        return self._env.reset(rng)

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool]:
        a = self.DISCRETE_ACTIONS[action_idx]
        return self._env.step(np.array([a]))

    @property
    def state_dim(self) -> int:
        return self.STATE_DIM


# ============================================================
# Replay Buffers
# ============================================================

class ReplayBuffer:
    """
    Circular experience replay buffer for off-policy algorithms.

    Stores (state, action, reward, next_state, done) tuples.
    Sampling is uniform random (no priorities).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int, rng: np.random.Generator | None = None
               ) -> tuple:
        rng = rng or np.random.default_rng()
        indices = rng.choice(len(self._buf), size=batch_size, replace=False)
        batch = [self._buf[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.stack(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


class _SumTree:
    """Binary sum tree for O(log n) priority sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)
        self.data: list = [None] * capacity
        self._ptr = 0
        self._size = 0

    def _propagate(self, idx: int, delta: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def update(self, idx: int, priority: float) -> None:
        leaf = idx + self.capacity - 1
        delta = priority - self.tree[leaf]
        self.tree[leaf] = priority
        self._propagate(leaf, delta)

    def add(self, priority: float, data) -> None:
        self.data[self._ptr] = data
        self.update(self._ptr, priority)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _retrieve(self, idx: int, s: float) -> int:
        left  = 2 * idx + 1
        right = 2 * idx + 2
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def get(self, s: float) -> tuple[int, float, object]:
        leaf = self._retrieve(0, s)
        data_idx = leaf - self.capacity + 1
        return data_idx, self.tree[leaf], self.data[data_idx]

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def __len__(self) -> int:
        return self._size


class PrioritizedReplayBuffer:
    """
    Proportional Prioritised Experience Replay (Schaul et al., 2015).

    Parameters
    ----------
    capacity : int
    alpha : float   priority exponent  (0 = uniform, 1 = full priority)
    beta  : float   IS-weight exponent (0 = no correction, 1 = full)
    beta_increment : float  anneal beta toward 1 each sample call
    eps   : float   small constant added to |TD-error| for stability
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-4,
        eps: float = 1e-5,
    ):
        self.alpha = alpha
        self.beta  = beta
        self.beta_increment = beta_increment
        self.eps   = eps
        self._tree = _SumTree(capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        # New transitions get max current priority (greedy)
        if self._tree._size == 0:
            max_p = 1.0
        else:
            leaf_start = self._tree.capacity - 1
            leaf_end   = leaf_start + self._tree._size
            max_p = float(self._tree.tree[leaf_start:leaf_end].max())
            if max_p == 0:
                max_p = 1.0
        self._tree.add(max_p, (
            np.array(state,  dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int, rng: np.random.Generator | None = None
               ) -> tuple:
        rng = rng or np.random.default_rng()
        n = len(self._tree)
        segment = self._tree.total / batch_size

        idxs, priorities, transitions = [], [], []
        for i in range(batch_size):
            s = rng.uniform(segment * i, segment * (i + 1))
            idx, p, data = self._tree.get(s)
            idxs.append(idx)
            priorities.append(p)
            transitions.append(data)

        # IS weights
        probs = np.array(priorities) / self._tree.total
        weights = (n * probs) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            np.stack(states),
            np.stack(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
            np.array(weights, dtype=np.float32),
            np.array(idxs),
        )

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for idx, p in zip(idxs, priorities):
            self._tree.update(int(idx), float(p))

    def __len__(self) -> int:
        return len(self._tree)


# ============================================================
# Neural Network (pure numpy MLP with backprop)
# ============================================================

def _relu(x):      return np.maximum(0.0, x)
def _relu_d(x):    return (x > 0).astype(float)
def _tanh(x):      return np.tanh(x)
def _tanh_d(x):    return 1.0 - np.tanh(x) ** 2
def _sigmoid(x):   return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
def _linear(x):    return x
def _linear_d(x):  return np.ones_like(x)


class MLP:
    """
    Multi-layer perceptron with configurable architecture.

    Supports:
    - Arbitrary depth / width
    - ReLU hidden activations, configurable output activation
    - Mini-batch gradient descent with Adam optimiser
    - Soft / hard target-network parameter copy

    Parameters
    ----------
    layer_sizes : list[int]  e.g. [state_dim, 256, 256, action_dim]
    output_activation : str  'linear' | 'tanh' | 'sigmoid'
    lr : float
    """

    def __init__(
        self,
        layer_sizes: list[int],
        output_activation: str = "linear",
        lr: float = 1e-3,
        random_state: int | None = None,
    ):
        rng = np.random.default_rng(random_state)
        self.layer_sizes = layer_sizes
        self.lr = lr

        # Weight initialisation (He for ReLU)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            scale = np.sqrt(2.0 / fan_in)
            self.W.append(rng.normal(0, scale, (fan_in, layer_sizes[i + 1])))
            self.b.append(np.zeros(layer_sizes[i + 1]))

        # Adam moments
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self._t = 0  # Adam time step

        out_acts = {
            "linear":  (_linear,  _linear_d),
            "tanh":    (_tanh,    _tanh_d),
            "sigmoid": (_sigmoid, None),
        }
        self._out_act, self._out_act_d = out_acts[output_activation]
        self._hidden_act, self._hidden_act_d = _relu, _relu_d

        # Cache for backprop
        self._cache: dict = {}

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        x : (batch, in_dim)  or (in_dim,) for single sample
        Returns output of shape (batch, out_dim) or (out_dim,).
        """
        scalar = x.ndim == 1
        if scalar:
            x = x[np.newaxis, :]

        a = x
        if training:
            self._cache = {"a": [a]}
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            if i < len(self.W) - 1:
                a = self._hidden_act(z)
                if training:
                    self._cache.setdefault("z", []).append(z)
            else:
                a = self._out_act(z)
                if training:
                    self._cache.setdefault("z", []).append(z)
            if training:
                self._cache["a"].append(a)

        return a[0] if scalar else a

    def backward(self, d_out: np.ndarray) -> None:
        """
        Compute gradients and apply Adam update.
        d_out : (batch, out_dim)  — gradient of loss w.r.t. network output.
        """
        if d_out.ndim == 1:
            d_out = d_out[np.newaxis, :]
        n = d_out.shape[0]
        self._t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        # Output layer delta
        z_out = self._cache["z"][-1]
        if self._out_act_d is not None:
            delta = d_out * self._out_act_d(z_out)
        else:
            delta = d_out  # linear pass-through for sigmoid (handled externally)

        for i in reversed(range(len(self.W))):
            a_prev = self._cache["a"][i]
            gW = a_prev.T @ delta / n
            gb = delta.mean(axis=0)

            # Adam
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * gW
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * gW ** 2
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * gb
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * gb ** 2

            mW_hat = self.mW[i] / (1 - beta1 ** self._t)
            vW_hat = self.vW[i] / (1 - beta2 ** self._t)
            mb_hat = self.mb[i] / (1 - beta1 ** self._t)
            vb_hat = self.vb[i] / (1 - beta2 ** self._t)

            self.W[i] -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.b[i] -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

            if i > 0:
                delta = (delta @ self.W[i].T) * self._hidden_act_d(
                    self._cache["z"][i - 1]
                )

    def soft_update(self, target: "MLP", tau: float) -> None:
        """θ_target ← τ θ_online + (1-τ) θ_target"""
        for w_s, w_t in zip(self.W, target.W):
            w_t[:] = tau * w_s + (1 - tau) * w_t
        for b_s, b_t in zip(self.b, target.b):
            b_t[:] = tau * b_s + (1 - tau) * b_t

    def hard_update(self, target: "MLP") -> None:
        """θ_target ← θ_online"""
        for w_s, w_t in zip(self.W, target.W):
            w_t[:] = w_s.copy()
        for b_s, b_t in zip(self.b, target.b):
            b_t[:] = b_s.copy()

    def copy_weights_from(self, source: "MLP") -> None:
        """Copy weights from another MLP of identical architecture."""
        for i in range(len(self.W)):
            self.W[i] = source.W[i].copy()
            self.b[i] = source.b[i].copy()


# ============================================================
# Exploration Noise
# ============================================================

class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration.

    dx_t = θ(μ - x_t)dt + σ dW_t

    Parameters
    ----------
    size : int
    mu : float       long-run mean
    theta : float    mean reversion rate
    sigma : float    noise scale
    dt : float       time step
    """

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2,
        random_state: int | None = None,
    ):
        self.mu    = np.full(size, mu)
        self.theta = theta
        self.sigma = sigma
        self.dt    = dt
        self._rng  = np.random.default_rng(random_state)
        self.reset()

    def reset(self) -> None:
        self.x = self.mu.copy()

    def sample(self) -> np.ndarray:
        dx = (
            self.theta * (self.mu - self.x) * self.dt
            + self.sigma * np.sqrt(self.dt) * self._rng.standard_normal(self.mu.shape)
        )
        self.x = self.x + dx
        return self.x.copy()


class GaussianNoise:
    """
    i.i.d. Gaussian exploration noise with optional decay.

    Parameters
    ----------
    size : int
    sigma : float      initial std
    sigma_min : float  minimum std after decay
    decay : float      multiplicative decay per call to sample()
    """

    def __init__(
        self,
        size: int,
        sigma: float = 0.1,
        sigma_min: float = 0.01,
        decay: float = 1.0,
        random_state: int | None = None,
    ):
        self.size      = size
        self.sigma     = sigma
        self.sigma_min = sigma_min
        self.decay     = decay
        self._rng      = np.random.default_rng(random_state)

    def sample(self) -> np.ndarray:
        noise = self._rng.normal(0, self.sigma, self.size)
        self.sigma = max(self.sigma_min, self.sigma * self.decay)
        return noise
