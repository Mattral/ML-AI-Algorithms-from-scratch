"""
Soft Actor-Critic (SAC)
========================
Maximum-entropy off-policy actor-critic algorithm for continuous control
(Haarnoja et al., 2018, 2019).

Objective
---------
Jointly optimise expected return *and* policy entropy:

    J(π) = E[ Σ_t γ^t (r_t + α H[π(·|s_t)]) ]

where α is a temperature parameter that trades off exploration vs. exploitation.

Key components
--------------
  Actor   π_φ(a|s)          — stochastic Gaussian policy with reparameterisation
  Critic  Q_θ1, Q_θ2        — twin soft Q-functions (minimum used for targets)
  Target  Q̄_θ1, Q̄_θ2       — exponential moving average of critic weights
  Temperature  α (or log α) — entropy coefficient, optionally auto-tuned via:
                              J(α) = E[-α (log π(a|s) + H̄)]

Reparameterisation trick (squashed Gaussian):
    ã = tanh(μ + σ ε),  ε ~ N(0,I)
    log π(ã|s) = log N(ε; 0,I) - sum log(1 - tanh²(μ + σε) + δ)

Update equations
----------------
Q targets:
    ã' ~ π(·|s'),   y = r + γ(1-d)[min Q̄_i(s',ã') - α log π(ã'|s')]
Critic loss:
    L_Q = E[(Q_i(s,a) - y)²]  for i ∈ {1,2}
Actor loss:
    L_π = E[α log π(ã|s) - min Q_i(s, ã)]
Alpha loss (auto-tune):
    L_α = E[-α (log π(a|s) + H̄)]

Only numpy and Python stdlib are used.
"""

from __future__ import annotations
import numpy as np
from .utils import ReplayBuffer, MLP


class SAC:
    """
    Soft Actor-Critic for continuous control.

    Parameters
    ----------
    state_dim       : int
    action_dim      : int
    action_low      : float
    action_high     : float
    hidden_sizes    : list[int]
    actor_lr        : float
    critic_lr       : float
    alpha_lr        : float      learning rate for entropy temperature
    gamma           : float      discount factor
    tau             : float      soft target update coefficient
    alpha           : float      initial entropy temperature (ignored if auto_alpha)
    auto_alpha      : bool       automatically tune entropy temperature
    target_entropy  : float | None
                                 target entropy (default: -action_dim)
    buffer_capacity : int
    batch_size      : int
    warmup_steps    : int
    log_std_min     : float
    log_std_max     : float
    random_state    : int | None
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: float = -1.0,
        action_high: float = 1.0,
        hidden_sizes: list[int] | None = None,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: float | None = None,
        buffer_capacity: int = 100_000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        random_state: int | None = None,
    ):
        self.action_dim    = action_dim
        self.action_low    = action_low
        self.action_high   = action_high
        self.gamma         = gamma
        self.tau           = tau
        self.log_alpha     = np.log(alpha)        # optimise log α
        self.alpha         = alpha
        self.auto_alpha    = auto_alpha
        self.target_entropy = (target_entropy if target_entropy is not None
                               else -float(action_dim))
        self.alpha_lr      = alpha_lr
        self.batch_size    = batch_size
        self.warmup_steps  = warmup_steps
        self.log_std_min   = log_std_min
        self.log_std_max   = log_std_max
        self._rng          = np.random.default_rng(random_state)
        self._step         = 0

        # Action scaling
        self._act_scale = (action_high - action_low) / 2.0
        self._act_bias  = (action_high + action_low) / 2.0

        hidden = hidden_sizes or [256, 256]

        # Actor: s → [mean, log_std]  (size 2*action_dim)
        self.actor = MLP([state_dim] + hidden + [action_dim * 2],
                          output_activation="linear", lr=actor_lr,
                          random_state=random_state)

        # Twin critics: (s, a) → Q
        self.critic1        = MLP([state_dim + action_dim] + hidden + [1],
                                   output_activation="linear", lr=critic_lr,
                                   random_state=random_state)
        self.critic1_target = MLP([state_dim + action_dim] + hidden + [1],
                                   output_activation="linear", lr=critic_lr,
                                   random_state=random_state)
        self.critic1.hard_update(self.critic1_target)

        self.critic2        = MLP([state_dim + action_dim] + hidden + [1],
                                   output_activation="linear", lr=critic_lr,
                                   random_state=random_state)
        self.critic2_target = MLP([state_dim + action_dim] + hidden + [1],
                                   output_activation="linear", lr=critic_lr,
                                   random_state=random_state)
        self.critic2.hard_update(self.critic2_target)

        # Alpha optimiser state (single-parameter Adam)
        self._alpha_m = 0.0
        self._alpha_v = 0.0
        self._alpha_t = 0

        # Replay
        self.buffer = ReplayBuffer(buffer_capacity)

        # Logging
        self.actor_losses_: list[float]   = []
        self.critic_losses_: list[float]  = []
        self.alpha_losses_: list[float]   = []
        self.alphas_: list[float]         = []
        self.episode_rewards_: list[float] = []

    # ------------------------------------------------------------------
    # Squashed Gaussian policy
    # ------------------------------------------------------------------

    def _actor_output(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, log_std) from actor network output."""
        out = self.actor.forward(states)           # (B, 2*A) or (2*A,)
        if out.ndim == 1:
            mean    = out[:self.action_dim]
            log_std = out[self.action_dim:]
        else:
            mean    = out[:, :self.action_dim]
            log_std = out[:, self.action_dim:]
        log_std = np.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def _sample_action(
        self,
        states: np.ndarray,
        training: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample action via reparameterisation + squashing.

        Returns
        -------
        action   : scaled to [action_low, action_high]
        log_prob : log π(a|s) accounting for tanh squashing
        """
        mean, log_std = self._actor_output(states)
        std = np.exp(log_std)

        # Reparameterisation
        eps  = self._rng.standard_normal(mean.shape)
        pre_tanh = mean + std * eps           # pre-squash

        # Squash
        a_tanh = np.tanh(pre_tanh)

        # Log prob (Gaussian) - log det(Jacobian of tanh)
        log_prob_gauss = -0.5 * (
            ((pre_tanh - mean) / (std + 1e-8)) ** 2
            + 2 * log_std + np.log(2 * np.pi)
        )
        # Squashing correction: log(1 - tanh²(u) + δ)
        log_prob_correction = np.log(1.0 - a_tanh ** 2 + 1e-6)

        if log_prob_gauss.ndim == 1:
            log_prob = float((log_prob_gauss - log_prob_correction).sum())
        else:
            log_prob = (log_prob_gauss - log_prob_correction).sum(axis=1)

        action = a_tanh * self._act_scale + self._act_bias
        return action, log_prob

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action for environment interaction.

        Parameters
        ----------
        state       : (state_dim,)
        deterministic : if True, use mean action (no sampling)
        """
        mean, _ = self._actor_output(state)
        if deterministic:
            return np.tanh(mean) * self._act_scale + self._act_bias
        action, _ = self._sample_action(state)
        return action

    # ------------------------------------------------------------------
    # Alpha (entropy temperature) update
    # ------------------------------------------------------------------

    def _update_alpha(self, log_probs: np.ndarray) -> float:
        """
        Gradient step on J(α) = E[-α (log π + H̄)]
        using Adam on log α (ensures α > 0).
        """
        self._alpha_t += 1
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

        # Gradient dJ/d(log α) = -E[log π + H̄]
        grad_log_alpha = float(-np.mean(log_probs + self.target_entropy))

        self._alpha_m = beta1 * self._alpha_m + (1 - beta1) * grad_log_alpha
        self._alpha_v = beta2 * self._alpha_v + (1 - beta2) * grad_log_alpha ** 2
        m_hat = self._alpha_m / (1 - beta1 ** self._alpha_t)
        v_hat = self._alpha_v / (1 - beta2 ** self._alpha_t)

        self.log_alpha -= self.alpha_lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        self.alpha = float(np.exp(self.log_alpha))

        alpha_loss = float(np.mean(-self.alpha * (log_probs + self.target_entropy)))
        return alpha_loss

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def _learn(self) -> tuple[float, float, float]:
        if len(self.buffer) < self.batch_size:
            return None, None, None

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size, self._rng)

        # ── Critic targets ────────────────────────────────────────────
        a_next, log_prob_next = self._sample_action(next_states)
        sa_next = np.concatenate([next_states, a_next], axis=1)

        q1_next = self.critic1_target.forward(sa_next).ravel()
        q2_next = self.critic2_target.forward(sa_next).ravel()
        q_next  = np.minimum(q1_next, q2_next)

        # Soft Bellman target
        y = rewards + self.gamma * (1.0 - dones) * (q_next - self.alpha * log_prob_next)

        # ── Critic update ─────────────────────────────────────────────
        sa = np.concatenate([states, actions], axis=1)

        q1_pred = self.critic1.forward(sa, training=True).ravel()
        td1 = y - q1_pred
        c1_loss = float(np.mean(td1 ** 2))
        self.critic1.backward(-2.0 * td1[:, np.newaxis] / self.batch_size)

        q2_pred = self.critic2.forward(sa, training=True).ravel()
        td2 = y - q2_pred
        c2_loss = float(np.mean(td2 ** 2))
        self.critic2.backward(-2.0 * td2[:, np.newaxis] / self.batch_size)

        critic_loss = (c1_loss + c2_loss) / 2.0

        # ── Actor update ──────────────────────────────────────────────
        a_new, log_prob_new = self._sample_action(states)
        sa_new = np.concatenate([states, a_new], axis=1)

        q1_new = self.critic1.forward(sa_new).ravel()
        q2_new = self.critic2.forward(sa_new).ravel()
        q_min  = np.minimum(q1_new, q2_new)

        actor_loss = float(np.mean(self.alpha * log_prob_new - q_min))

        # Actor gradient: dL/d(actor_params) via chain rule through log_prob and q
        # dL/d(a) = α d(log π)/d(a) - d(Q)/d(a) (simplified: uniform gradient direction)
        d_a = (self.alpha * np.ones((self.batch_size, self.action_dim))
               - np.ones((self.batch_size, self.action_dim))) / self.batch_size

        # Map through tanh squashing to get gradient for actor output
        mean, log_std = self._actor_output(states)
        std = np.exp(log_std)
        a_tanh_part = (a_new - self._act_bias) / self._act_scale
        sech2 = 1.0 - a_tanh_part ** 2  # d(tanh)/d(pre_tanh)
        d_pretanh = d_a * self._act_scale * sech2
        d_mean    = d_pretanh
        d_log_std = d_pretanh * (a_new - self._act_bias - mean * self._act_scale) / \
                    (self._act_scale + 1e-8)

        d_actor_out = np.concatenate([d_mean, d_log_std], axis=1)
        # Force forward with training=True to set cache
        self.actor.forward(states, training=True)
        self.actor.backward(d_actor_out)

        # ── Alpha update ──────────────────────────────────────────────
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = self._update_alpha(log_prob_new)

        # ── Soft target updates ───────────────────────────────────────
        self.critic1.soft_update(self.critic1_target, self.tau)
        self.critic2.soft_update(self.critic2_target, self.tau)

        return actor_loss, critic_loss, alpha_loss

    # ------------------------------------------------------------------
    # Environment interaction
    # ------------------------------------------------------------------

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> tuple[float | None, float | None, float | None]:
        self.buffer.push(state, action, reward, next_state, done)
        self._step += 1

        if self._step < self.warmup_steps:
            return None, None, None

        al, cl, alpha_l = self._learn()
        if al is not None:
            self.actor_losses_.append(al)
            self.critic_losses_.append(cl)
            self.alpha_losses_.append(alpha_l)
            self.alphas_.append(self.alpha)

        return al, cl, alpha_l

    def train_episode(self, env) -> float:
        rng_arg = self._rng if hasattr(env, '_env') else None
        state = env.reset(rng_arg) if rng_arg is not None else env.reset()
        total_reward = 0.0
        done = False

        while not done:
            if self._step < self.warmup_steps:
                action = self._rng.uniform(
                    self.action_low, self.action_high, self.action_dim
                )
            else:
                action = self.select_action(state)
            next_state, reward, done = env.step(action)
            self.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        self.episode_rewards_.append(total_reward)
        return total_reward

    def train(self, env, n_episodes: int) -> "SAC":
        for _ in range(n_episodes):
            self.train_episode(env)
        return self
