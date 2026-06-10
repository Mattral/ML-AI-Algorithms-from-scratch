"""
Deep Q-Network (DQN)
=====================
Neural-network function approximator for Q-learning, with three
production-grade enhancements:

  1. Experience Replay         — breaks temporal correlations (Mnih et al., 2013)
  2. Target Network            — stabilises training targets  (Mnih et al., 2015)
  3. Double DQN                — removes maximisation bias    (van Hasselt et al., 2016)

Optional:
  4. Dueling Network           — separate V(s) and A(s,a) streams
                                 (Wang et al., 2016)
  5. Prioritised Replay        — focuses on high-TD-error transitions
                                 (Schaul et al., 2015)

Update rule (Double DQN):
    a* = argmax_a  Q_online(s', a)
    y  = r + γ (1-done) Q_target(s', a*)
    L  = (y - Q_online(s, a))²

Only numpy and Python stdlib are used.
"""

from __future__ import annotations
import numpy as np
from copy import deepcopy

from .utils import ReplayBuffer, PrioritizedReplayBuffer, MLP


# ============================================================
# Dueling MLP
# ============================================================

class DuelingMLP:
    """
    Dueling network: two heads sharing a common feature trunk.

    Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)

    Parameters
    ----------
    state_dim   : int
    n_actions   : int
    hidden_sizes: list[int]   size of shared hidden layers
    lr          : float
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_sizes: list[int] | None = None,
        lr: float = 1e-3,
        random_state: int | None = None,
    ):
        hidden_sizes = hidden_sizes or [128, 128]
        self.n_actions = n_actions
        rng = np.random.default_rng(random_state)

        # Shared trunk
        trunk_sizes = [state_dim] + hidden_sizes
        self._trunk = MLP(trunk_sizes, output_activation="linear",
                          lr=lr, random_state=random_state)

        # Value head: hidden[-1] → 1
        self._value_head = MLP([hidden_sizes[-1], 64, 1],
                                output_activation="linear", lr=lr,
                                random_state=random_state)
        # Advantage head: hidden[-1] → n_actions
        self._adv_head   = MLP([hidden_sizes[-1], 64, n_actions],
                                output_activation="linear", lr=lr,
                                random_state=random_state)

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        scalar = x.ndim == 1
        if scalar:
            x = x[np.newaxis, :]
        h   = self._trunk.forward(x, training=training)
        V   = self._value_head.forward(h, training=training)          # (B,1)
        A   = self._adv_head.forward(h, training=training)            # (B,A)
        Q   = V + A - A.mean(axis=1, keepdims=True)
        return Q[0] if scalar else Q

    def soft_update(self, target: "DuelingMLP", tau: float) -> None:
        self._trunk.soft_update(target._trunk, tau)
        self._value_head.soft_update(target._value_head, tau)
        self._adv_head.soft_update(target._adv_head, tau)

    def hard_update(self, target: "DuelingMLP") -> None:
        self._trunk.hard_update(target._trunk)
        self._value_head.hard_update(target._value_head)
        self._adv_head.hard_update(target._adv_head)

    def copy_weights_from(self, source: "DuelingMLP") -> None:
        self._trunk.copy_weights_from(source._trunk)
        self._value_head.copy_weights_from(source._value_head)
        self._adv_head.copy_weights_from(source._adv_head)

    def backward(self, d_out: np.ndarray) -> None:
        # Simplified backward: treat Q output as direct loss gradient
        # into the advantage head (standard approach for DQN)
        d_A = d_out - d_out.mean(axis=1, keepdims=True)
        d_V = d_out.mean(axis=1, keepdims=True) * np.ones((d_out.shape[0], 1))
        self._adv_head.backward(d_A)
        self._value_head.backward(d_V)
        # Trunk gradient = sum of both heads (simplified)
        self._trunk.backward(d_out.mean(axis=1, keepdims=True) *
                             np.ones((d_out.shape[0],
                                      self._trunk.layer_sizes[-1])))


# ============================================================
# DQN Agent
# ============================================================

class DQN:
    """
    Deep Q-Network agent.

    Parameters
    ----------
    state_dim       : int
    n_actions       : int
    hidden_sizes    : list[int]
    lr              : float          learning rate
    gamma           : float          discount factor
    epsilon         : float          initial exploration ε
    epsilon_min     : float          minimum ε
    epsilon_decay   : float          multiplicative decay per step
    batch_size      : int
    buffer_capacity : int
    target_update   : int            hard target update every N steps
    tau             : float | None   soft update coeff; None → hard update
    double_dqn      : bool           use Double DQN
    dueling         : bool           use Dueling Network
    prioritized     : bool           use Prioritised Replay
    random_state    : int | None
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_sizes: list[int] | None = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        target_update: int = 100,
        tau: float | None = None,
        double_dqn: bool = True,
        dueling: bool = False,
        prioritized: bool = False,
        random_state: int | None = None,
    ):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.tau           = tau
        self.double_dqn    = double_dqn
        self.prioritized   = prioritized
        self._rng          = np.random.default_rng(random_state)
        self._step         = 0

        hidden = hidden_sizes or [128, 128]

        # Online and target networks
        if dueling:
            self.online_net = DuelingMLP(state_dim, n_actions, hidden, lr, random_state)
            self.target_net = DuelingMLP(state_dim, n_actions, hidden, lr, random_state)
        else:
            self.online_net = MLP([state_dim] + hidden + [n_actions],
                                   output_activation="linear", lr=lr,
                                   random_state=random_state)
            self.target_net = MLP([state_dim] + hidden + [n_actions],
                                   output_activation="linear", lr=lr,
                                   random_state=random_state)

        # Sync target = online at init
        self.online_net.hard_update(self.target_net)

        # Replay buffer
        if prioritized:
            self.buffer = PrioritizedReplayBuffer(buffer_capacity)
        else:
            self.buffer = ReplayBuffer(buffer_capacity)

        # Logging
        self.losses_: list[float] = []
        self.episode_rewards_: list[float] = []
        self.epsilons_: list[float] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy action with linear annealing."""
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        q = self.online_net.forward(state)
        return int(np.argmax(q))

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def _learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        if self.prioritized:
            states, actions, rewards, next_states, dones, weights, idxs = \
                self.buffer.sample(self.batch_size, self._rng)
        else:
            states, actions, rewards, next_states, dones = \
                self.buffer.sample(self.batch_size, self._rng)
            weights = np.ones(self.batch_size)

        actions = actions.ravel().astype(int)

        # Compute targets
        with_no_grad = True   # conceptually; we don't call backward on target_net
        q_next_target = self.target_net.forward(next_states)  # (B, A)

        if self.double_dqn:
            q_next_online = self.online_net.forward(next_states)
            a_star = np.argmax(q_next_online, axis=1)           # online selects
            q_next_val = q_next_target[np.arange(self.batch_size), a_star]
        else:
            q_next_val = q_next_target.max(axis=1)

        targets = rewards + self.gamma * (1.0 - dones) * q_next_val  # (B,)

        # Compute predictions and loss gradient
        q_pred_all = self.online_net.forward(states, training=True)   # (B, A)
        q_pred = q_pred_all[np.arange(self.batch_size), actions]      # (B,)

        td_errors = targets - q_pred                                   # (B,)
        loss = float(np.mean(weights * td_errors ** 2))

        # Gradient: dL/dQ_pred = -2 * w * td_error (averaged in backward)
        d_out = np.zeros_like(q_pred_all)
        d_out[np.arange(self.batch_size), actions] = (
            -2.0 * weights * td_errors / self.batch_size
        )
        self.online_net.backward(d_out)

        # Update priorities
        if self.prioritized:
            self.buffer.update_priorities(idxs, td_errors)

        return loss

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float | None:
        """Store transition, learn, update target, decay ε."""
        self.buffer.push(state, np.array([action]), reward, next_state, done)
        self._step += 1

        loss = self._learn()
        if loss is not None:
            self.losses_.append(loss)

        # Target update
        if self.tau is not None:
            self.online_net.soft_update(self.target_net, self.tau)
        elif self._step % self.target_update == 0:
            self.online_net.hard_update(self.target_net)

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss

    # ------------------------------------------------------------------
    # Episode training
    # ------------------------------------------------------------------

    def train_episode(self, env) -> float:
        """Run one episode and return total reward."""
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = self.select_action(state)
            next_state, reward, done = env.step(action)
            self.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        self.episode_rewards_.append(total_reward)
        self.epsilons_.append(self.epsilon)
        return total_reward

    def train(self, env, n_episodes: int) -> "DQN":
        for _ in range(n_episodes):
            self.train_episode(env)
        return self
