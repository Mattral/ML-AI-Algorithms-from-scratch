"""
Deep Deterministic Policy Gradient (DDPG) and TD3
===================================================
DDPG (Lillicrap et al., 2015) extends DQN to continuous action spaces:
  - Deterministic actor  π_θ(s) → a   (output activation: tanh → scaled)
  - Critic Q_φ(s, a)     approximates action-value function
  - Target networks (soft update) for both actor and critic
  - Ornstein-Uhlenbeck or Gaussian noise for exploration

TD3 — Twin Delayed Deep Deterministic Policy Gradient (Fujimoto et al., 2018)
-----------------------------------------------------------------------
Three key improvements over DDPG:
  1. Twin critics          — two independent Q-networks; use min for targets
  2. Delayed policy update — actor updated every `policy_delay` critic steps
  3. Target policy noise   — smoothed noisy targets prevent over-fitting to peaks

Update equations
----------------
Critic targets (TD3):
    ã = π_θ'(s') + clip(N(0,σ̃), -c, c)          # smoothed target action
    y = r + γ(1-d) min(Q_1'(s',ã), Q_2'(s',ã))

Actor loss (DDPG / TD3):
    L_π = -E[Q_1(s, π_θ(s))]

Only numpy and Python stdlib are used.
"""

from __future__ import annotations
import numpy as np
from .utils import ReplayBuffer, MLP, OrnsteinUhlenbeckNoise, GaussianNoise


# ============================================================
# DDPG
# ============================================================

class DDPG:
    """
    Deep Deterministic Policy Gradient agent.

    Parameters
    ----------
    state_dim       : int
    action_dim      : int
    action_low      : float    lower bound of action space
    action_high     : float    upper bound of action space
    hidden_sizes    : list[int]
    actor_lr        : float
    critic_lr       : float
    gamma           : float
    tau             : float    soft update coefficient
    buffer_capacity : int
    batch_size      : int
    noise_type      : str      'ou' | 'gaussian'
    noise_sigma     : float    exploration noise scale
    warmup_steps    : int      random actions before learning starts
    random_state    : int | None
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: float = -1.0,
        action_high: float = 1.0,
        hidden_sizes: list[int] | None = None,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        noise_type: str = "ou",
        noise_sigma: float = 0.1,
        warmup_steps: int = 1000,
        random_state: int | None = None,
    ):
        self.action_dim  = action_dim
        self.action_low  = action_low
        self.action_high = action_high
        self.gamma       = gamma
        self.tau         = tau
        self.batch_size  = batch_size
        self.warmup_steps = warmup_steps
        self._rng        = np.random.default_rng(random_state)
        self._step       = 0

        hidden = hidden_sizes or [256, 256]
        act_scale = (action_high - action_low) / 2.0
        act_bias  = (action_high + action_low) / 2.0
        self._act_scale = act_scale
        self._act_bias  = act_bias

        # Actor: s → a  (tanh output scaled to [low, high])
        self.actor        = MLP([state_dim] + hidden + [action_dim],
                                 output_activation="tanh", lr=actor_lr,
                                 random_state=random_state)
        self.actor_target = MLP([state_dim] + hidden + [action_dim],
                                 output_activation="tanh", lr=actor_lr,
                                 random_state=random_state)
        self.actor.hard_update(self.actor_target)

        # Critic: (s, a) → Q
        self.critic        = MLP([state_dim + action_dim] + hidden + [1],
                                  output_activation="linear", lr=critic_lr,
                                  random_state=random_state)
        self.critic_target = MLP([state_dim + action_dim] + hidden + [1],
                                  output_activation="linear", lr=critic_lr,
                                  random_state=random_state)
        self.critic.hard_update(self.critic_target)

        # Replay
        self.buffer = ReplayBuffer(buffer_capacity)

        # Exploration noise
        if noise_type == "ou":
            self.noise = OrnsteinUhlenbeckNoise(
                action_dim, sigma=noise_sigma, random_state=random_state
            )
        else:
            self.noise = GaussianNoise(action_dim, sigma=noise_sigma,
                                       random_state=random_state)

        # Logging
        self.actor_losses_: list[float] = []
        self.critic_losses_: list[float] = []
        self.episode_rewards_: list[float] = []

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def _scale_action(self, a_tanh: np.ndarray) -> np.ndarray:
        return a_tanh * self._act_scale + self._act_bias

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        a = self.actor.forward(state)                      # tanh in [-1,1]
        if add_noise:
            a = a + self.noise.sample()
            a = np.clip(a, -1.0, 1.0)
        return self._scale_action(a)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _learn(self) -> tuple[float, float] | tuple[None, None]:
        if len(self.buffer) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size, self._rng)

        # ── Critic update ──────────────────────────────────────────────
        # Target action from actor_target
        a_next = self.actor_target.forward(next_states)            # (B, A_dim)
        a_next = np.clip(a_next, -1.0, 1.0)
        sa_next = np.concatenate([next_states,
                                   a_next * self._act_scale + self._act_bias], axis=1)

        q_next = self.critic_target.forward(sa_next).ravel()       # (B,)
        y = rewards + self.gamma * (1.0 - dones) * q_next          # (B,)

        # Normalise stored actions back to [-1,1] for concat
        a_norm = (actions - self._act_bias) / self._act_scale
        sa = np.concatenate([states, actions], axis=1)
        q_pred = self.critic.forward(sa, training=True).ravel()    # (B,)

        td_errors = y - q_pred
        critic_loss = float(np.mean(td_errors ** 2))

        d_critic = -2.0 * td_errors[:, np.newaxis] / self.batch_size
        self.critic.backward(d_critic)

        # ── Actor update ───────────────────────────────────────────────
        a_pred = self.actor.forward(states, training=True)         # (B, A_dim)
        a_scaled = a_pred * self._act_scale + self._act_bias
        sa_pred = np.concatenate([states, a_scaled], axis=1)

        q_actor = self.critic.forward(sa_pred, training=True).ravel()
        actor_loss = float(-np.mean(q_actor))

        # dL/da = -dQ/da  (chain through critic → actor)
        d_q_wrt_sa = np.ones((self.batch_size, 1)) / self.batch_size
        # Gradient w.r.t. action part only
        d_a = d_q_wrt_sa * (-1.0) * self._act_scale               # (B, A_dim)
        self.actor.backward(d_a)

        # ── Soft target updates ────────────────────────────────────────
        self.actor.soft_update(self.actor_target, self.tau)
        self.critic.soft_update(self.critic_target, self.tau)

        return actor_loss, critic_loss

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> tuple[float | None, float | None]:
        self.buffer.push(state, action, reward, next_state, done)
        self._step += 1

        if self._step < self.warmup_steps:
            return None, None

        actor_loss, critic_loss = self._learn()
        if actor_loss is not None:
            self.actor_losses_.append(actor_loss)
            self.critic_losses_.append(critic_loss)
        return actor_loss, critic_loss

    def train_episode(self, env) -> float:
        state = env.reset(self._rng)
        self.noise.reset() if hasattr(self.noise, 'reset') else None
        total_reward = 0.0
        done = False

        while not done:
            if self._step < self.warmup_steps:
                action = self._rng.uniform(self.action_low, self.action_high,
                                           self.action_dim)
            else:
                action = self.select_action(state)
            next_state, reward, done = env.step(action)
            self.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        self.episode_rewards_.append(total_reward)
        return total_reward

    def train(self, env, n_episodes: int) -> "DDPG":
        for _ in range(n_episodes):
            self.train_episode(env)
        return self


# ============================================================
# TD3
# ============================================================

class TD3(DDPG):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3).

    Inherits from DDPG and adds:
    - Second critic (critic2 + critic2_target)
    - Policy delay: actor updated every `policy_delay` critic steps
    - Target policy smoothing: Gaussian noise clipped to ±noise_clip

    Parameters (additional to DDPG)
    --------------------------------
    policy_delay     : int    critic updates per actor update (default 2)
    target_noise     : float  std of smoothing noise on target actions
    noise_clip       : float  clipping bound for smoothing noise
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: float = -1.0,
        action_high: float = 1.0,
        hidden_sizes: list[int] | None = None,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        noise_type: str = "gaussian",
        noise_sigma: float = 0.1,
        warmup_steps: int = 1000,
        policy_delay: int = 2,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        random_state: int | None = None,
    ):
        super().__init__(
            state_dim, action_dim, action_low, action_high, hidden_sizes,
            actor_lr, critic_lr, gamma, tau, buffer_capacity, batch_size,
            noise_type, noise_sigma, warmup_steps, random_state,
        )
        self.policy_delay  = policy_delay
        self.target_noise  = target_noise
        self.noise_clip    = noise_clip
        self._critic_steps = 0

        hidden = hidden_sizes or [256, 256]
        # Second critic pair
        self.critic2        = MLP([state_dim + action_dim] + hidden + [1],
                                   output_activation="linear", lr=critic_lr,
                                   random_state=random_state)
        self.critic2_target = MLP([state_dim + action_dim] + hidden + [1],
                                   output_activation="linear", lr=critic_lr,
                                   random_state=random_state)
        self.critic2.hard_update(self.critic2_target)

    def _learn(self) -> tuple[float, float] | tuple[None, None]:
        if len(self.buffer) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size, self._rng)

        self._critic_steps += 1

        # ── Target action with smoothing noise ────────────────────────
        a_next = self.actor_target.forward(next_states)
        noise  = np.clip(
            self._rng.normal(0, self.target_noise, a_next.shape),
            -self.noise_clip, self.noise_clip
        )
        a_next = np.clip(a_next + noise, -1.0, 1.0)
        a_next_scaled = a_next * self._act_scale + self._act_bias
        sa_next = np.concatenate([next_states, a_next_scaled], axis=1)

        # ── Twin critics targets (take min) ───────────────────────────
        q1_next = self.critic_target.forward(sa_next).ravel()
        q2_next = self.critic2_target.forward(sa_next).ravel()
        q_next  = np.minimum(q1_next, q2_next)
        y       = rewards + self.gamma * (1.0 - dones) * q_next

        # ── Update both critics ───────────────────────────────────────
        sa = np.concatenate([states, actions], axis=1)

        q1_pred = self.critic.forward(sa, training=True).ravel()
        td1 = y - q1_pred
        critic1_loss = float(np.mean(td1 ** 2))
        self.critic.backward(-2.0 * td1[:, np.newaxis] / self.batch_size)

        q2_pred = self.critic2.forward(sa, training=True).ravel()
        td2 = y - q2_pred
        critic2_loss = float(np.mean(td2 ** 2))
        self.critic2.backward(-2.0 * td2[:, np.newaxis] / self.batch_size)

        critic_loss = (critic1_loss + critic2_loss) / 2.0
        actor_loss = None

        # ── Delayed actor update ──────────────────────────────────────
        if self._critic_steps % self.policy_delay == 0:
            a_pred   = self.actor.forward(states, training=True)
            a_scaled = a_pred * self._act_scale + self._act_bias
            sa_pred  = np.concatenate([states, a_scaled], axis=1)
            q_actor  = self.critic.forward(sa_pred, training=True).ravel()
            actor_loss = float(-np.mean(q_actor))
            d_a = -np.ones((self.batch_size, self.action_dim)) * self._act_scale \
                  / self.batch_size
            self.actor.backward(d_a)
            self.actor.soft_update(self.actor_target, self.tau)

        self.critic.soft_update(self.critic_target, self.tau)
        self.critic2.soft_update(self.critic2_target, self.tau)

        return actor_loss, critic_loss
