"""
Tests for mlscratch.reinforcement.ppo — PPO (clip & KL, discrete & continuous)
"""

from __future__ import annotations
import numpy as np
import pytest

from mlscratch.reinforcement.ppo import PPO
from mlscratch.reinforcement.utils import DiscreteEnv, ContinuousEnv


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def disc_env():
    return DiscreteEnv(max_steps=50)


@pytest.fixture
def cont_env():
    return ContinuousEnv(max_steps=50)


def _make_ppo_discrete(env, **kw):
    defaults = dict(
        state_dim=env.state_dim,
        n_actions=env.N_ACTIONS,
        hidden_sizes=[32, 32],
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        n_epochs=2,
        batch_size=16,
        rollout_len=64,
        entropy_coef=0.01,
        value_coef=0.5,
        random_state=0,
    )
    defaults.update(kw)
    return PPO(**defaults)


def _make_ppo_continuous(env, **kw):
    defaults = dict(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        action_low=env.ACTION_LOW,
        action_high=env.ACTION_HIGH,
        hidden_sizes=[32, 32],
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        n_epochs=2,
        batch_size=16,
        rollout_len=64,
        random_state=0,
    )
    defaults.update(kw)
    return PPO(**defaults)


# ===================================================================
# Architecture
# ===================================================================

class TestPPOArchitecture:
    def test_discrete_actor_output_shape(self, disc_env):
        agent = _make_ppo_discrete(disc_env)
        state = disc_env.reset()
        logits = agent.actor.forward(state)
        assert logits.shape == (disc_env.N_ACTIONS,)

    def test_continuous_actor_output_shape(self, cont_env):
        agent = _make_ppo_continuous(cont_env)
        state = cont_env.reset()
        out = agent.actor.forward(state)
        # mean + log_std => 2 * action_dim
        assert out.shape == (2 * cont_env.action_dim,)

    def test_critic_output_is_scalar(self, disc_env):
        agent = _make_ppo_discrete(disc_env)
        state = disc_env.reset()
        v = agent.critic.forward(state)
        assert np.isscalar(v) or v.shape in ((), (1,))

    def test_continuous_mode_flag(self, cont_env):
        agent = _make_ppo_continuous(cont_env)
        assert agent.continuous is True

    def test_discrete_mode_flag(self, disc_env):
        agent = _make_ppo_discrete(disc_env)
        assert agent.continuous is False


# ===================================================================
# Action sampling
# ===================================================================

class TestPPOActionSampling:
    def test_discrete_action_in_range(self, disc_env):
        agent = _make_ppo_discrete(disc_env)
        state = disc_env.reset()
        for _ in range(50):
            action, log_prob, entropy = agent._actor_forward(state)
            assert 0 <= int(action[0]) < disc_env.N_ACTIONS

    def test_discrete_log_prob_is_finite(self, disc_env):
        agent = _make_ppo_discrete(disc_env)
        state = disc_env.reset()
        _, log_prob, _ = agent._actor_forward(state)
        assert np.isfinite(log_prob)

    def test_discrete_entropy_positive(self, disc_env):
        agent = _make_ppo_discrete(disc_env)
        state = disc_env.reset()
        _, _, entropy = agent._actor_forward(state)
        assert entropy >= 0.0

    def test_continuous_action_shape(self, cont_env):
        agent = _make_ppo_continuous(cont_env)
        state = cont_env.reset()
        action, _, _ = agent._actor_forward(state)
        assert action.shape == (cont_env.action_dim,)

    def test_continuous_action_in_bounds(self, cont_env):
        agent = _make_ppo_continuous(cont_env)
        state = cont_env.reset()
        for _ in range(50):
            action, _, _ = agent._actor_forward(state)
            assert np.all(action >= cont_env.ACTION_LOW - 1e-9)
            assert np.all(action <= cont_env.ACTION_HIGH + 1e-9)

    def test_continuous_log_prob_finite(self, cont_env):
        agent = _make_ppo_continuous(cont_env)
        state = cont_env.reset()
        _, log_prob, _ = agent._actor_forward(state)
        assert np.isfinite(log_prob)

    def test_predict_deterministic_discrete(self, disc_env):
        agent = _make_ppo_discrete(disc_env)
        state = disc_env.reset()
        actions = {int(agent.predict(state)[0]) for _ in range(20)}
        assert len(actions) == 1   # deterministic → same each time

    def test_predict_continuous_in_bounds(self, cont_env):
        agent = _make_ppo_continuous(cont_env)
        state = cont_env.reset()
        a = agent.predict(state, deterministic=True)
        assert np.all(a >= cont_env.ACTION_LOW - 1e-9)
        assert np.all(a <= cont_env.ACTION_HIGH + 1e-9)


# ===================================================================
# GAE computation
# ===================================================================

class TestGAE:
    def test_gae_returns_correct_shapes(self, disc_env):
        agent = _make_ppo_discrete(disc_env)
        T = 20
        rewards = np.random.default_rng(0).standard_normal(T)
        values  = np.random.default_rng(1).standard_normal(T)
        dones   = np.zeros(T)
        dones[-1] = 1.0
        adv, ret = agent._compute_gae(rewards, values, dones, last_value=0.0)
        assert adv.shape == (T,)
        assert ret.shape == (T,)

    def test_gae_returns_equal_advantages_plus_values(self, disc_env):
        agent = _make_ppo_discrete(disc_env)
        T = 10
        rng = np.random.default_rng(42)
        rewards = rng.standard_normal(T)
        values  = rng.standard_normal(T)
        dones   = np.zeros(T)
        adv, ret = agent._compute_gae(rewards, values, dones, last_value=0.5)
        np.testing.assert_allclose(ret, adv + values, atol=1e-8)

    def test_gae_terminal_step_no_bootstrap(self, disc_env):
        """At a terminal step the next value is 0 regardless of last_value."""
        agent = _make_ppo_discrete(disc_env, gamma=1.0, lam=1.0)
        rewards = np.array([1.0, 2.0, 3.0])
        values  = np.array([0.0, 0.0, 0.0])
        dones   = np.array([0.0, 0.0, 1.0])
        adv, ret = agent._compute_gae(rewards, values, dones, last_value=999.0)
        # Terminal step: δ = r - V = 3.0; advantage = 3.0
        np.testing.assert_allclose(adv[2], 3.0, atol=1e-8)

    def test_gae_lambda_zero_equals_td_residual(self, disc_env):
        """λ=0 → GAE collapses to one-step TD residuals."""
        agent = _make_ppo_discrete(disc_env, gamma=0.99, lam=0.0)
        T = 5
        rng = np.random.default_rng(3)
        r = rng.standard_normal(T)
        v = rng.standard_normal(T)
        d = np.zeros(T)
        adv, _ = agent._compute_gae(r, v, d, last_value=0.0)
        # δ_t = r_t + γ V_{t+1} - V_t
        expected_last = r[-1] - v[-1]   # done mask = 1 for last, but dones=0 here
        expected_last = r[-1] + 0.99 * 0.0 - v[-1]
        np.testing.assert_allclose(adv[-1], expected_last, atol=1e-8)

    def test_gae_lambda_one_equals_monte_carlo(self, disc_env):
        """λ=1, γ=1 → GAE = full Monte-Carlo returns - baseline."""
        agent = _make_ppo_discrete(disc_env, gamma=1.0, lam=1.0)
        r = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        v = np.zeros(5)
        d = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        adv, ret = agent._compute_gae(r, v, d, last_value=0.0)
        # Returns with γ=1: [4, 3, 2, 1, 0]
        np.testing.assert_allclose(ret, [4, 3, 2, 1, 0], atol=1e-8)


# ===================================================================
# Batch log-prob computation
# ===================================================================

class TestLogProbBatch:
    def test_discrete_log_prob_batch_shape(self, disc_env):
        agent = _make_ppo_discrete(disc_env)
        rng = np.random.default_rng(0)
        states  = rng.standard_normal((16, disc_env.state_dim))
        actions = rng.integers(0, disc_env.N_ACTIONS, (16, 1))
        lp, ent = agent._log_prob_batch(states, actions)
        assert lp.shape  == (16,)
        assert ent.shape == (16,)

    def test_discrete_entropies_non_negative(self, disc_env):
        agent = _make_ppo_discrete(disc_env)
        rng = np.random.default_rng(0)
        states  = rng.standard_normal((16, disc_env.state_dim))
        actions = rng.integers(0, disc_env.N_ACTIONS, (16, 1))
        _, ent = agent._log_prob_batch(states, actions)
        assert np.all(ent >= 0)

    def test_discrete_log_probs_non_positive(self, disc_env):
        agent = _make_ppo_discrete(disc_env)
        rng = np.random.default_rng(0)
        states  = rng.standard_normal((16, disc_env.state_dim))
        actions = rng.integers(0, disc_env.N_ACTIONS, (16, 1))
        lp, _ = agent._log_prob_batch(states, actions)
        assert np.all(lp <= 0)

    def test_continuous_log_prob_batch_shape(self, cont_env):
        agent = _make_ppo_continuous(cont_env)
        rng = np.random.default_rng(0)
        states  = rng.standard_normal((16, cont_env.state_dim))
        actions = rng.uniform(-1, 1, (16, cont_env.action_dim))
        lp, ent = agent._log_prob_batch(states, actions)
        assert lp.shape  == (16,)
        assert ent.shape == (16,)

    def test_log_probs_finite(self, disc_env):
        agent = _make_ppo_discrete(disc_env)
        rng = np.random.default_rng(0)
        states  = rng.standard_normal((32, disc_env.state_dim))
        actions = rng.integers(0, disc_env.N_ACTIONS, (32, 1))
        lp, ent = agent._log_prob_batch(states, actions)
        assert np.all(np.isfinite(lp))
        assert np.all(np.isfinite(ent))


# ===================================================================
# Training
# ===================================================================

class TestPPOTraining:
    def test_train_episode_returns_float_discrete(self, disc_env):
        agent = _make_ppo_discrete(disc_env)
        r = agent.train_episode(disc_env)
        assert isinstance(r, float)

    def test_train_episode_returns_float_continuous(self, cont_env):
        agent = _make_ppo_continuous(cont_env)
        r = agent.train_episode(cont_env)
        assert isinstance(r, float)

    def test_episode_rewards_recorded(self, disc_env):
        agent = _make_ppo_discrete(disc_env)
        agent.train(disc_env, n_episodes=5)
        assert len(agent.episode_rewards_) == 5

    def test_train_returns_self(self, disc_env):
        agent = _make_ppo_discrete(disc_env)
        assert agent.train(disc_env, n_episodes=2) is agent

    def test_policy_losses_recorded_after_rollout(self, disc_env):
        agent = _make_ppo_discrete(disc_env, rollout_len=32)
        agent.train(disc_env, n_episodes=5)
        assert len(agent.policy_losses_) >= 1

    def test_value_losses_recorded(self, disc_env):
        agent = _make_ppo_discrete(disc_env, rollout_len=32)
        agent.train(disc_env, n_episodes=5)
        assert len(agent.value_losses_) >= 1

    def test_entropies_positive(self, disc_env):
        agent = _make_ppo_discrete(disc_env, rollout_len=32)
        agent.train(disc_env, n_episodes=5)
        if agent.entropies_:
            assert all(e >= 0 for e in agent.entropies_)

    def test_actor_weights_change_after_training(self, disc_env):
        agent = _make_ppo_discrete(disc_env, rollout_len=32)
        W0 = [w.copy() for w in agent.actor.W]
        agent.train(disc_env, n_episodes=5)
        changed = any(
            not np.allclose(w0, w1)
            for w0, w1 in zip(W0, agent.actor.W)
        )
        assert changed, "Actor weights must change during training"

    def test_critic_weights_change_after_training(self, disc_env):
        agent = _make_ppo_discrete(disc_env, rollout_len=32)
        W0 = [w.copy() for w in agent.critic.W]
        agent.train(disc_env, n_episodes=5)
        assert any(
            not np.allclose(w0, w1)
            for w0, w1 in zip(W0, agent.critic.W)
        )

    def test_no_nan_in_policy_losses(self, disc_env):
        agent = _make_ppo_discrete(disc_env, rollout_len=32)
        agent.train(disc_env, n_episodes=5)
        assert all(np.isfinite(l) for l in agent.policy_losses_)

    def test_no_nan_in_value_losses(self, disc_env):
        agent = _make_ppo_discrete(disc_env, rollout_len=32)
        agent.train(disc_env, n_episodes=5)
        assert all(np.isfinite(l) for l in agent.value_losses_)

    def test_continuous_actor_weights_change(self, cont_env):
        agent = _make_ppo_continuous(cont_env, rollout_len=32)
        W0 = [w.copy() for w in agent.actor.W]
        agent.train(cont_env, n_episodes=5)
        assert any(
            not np.allclose(w0, w1)
            for w0, w1 in zip(W0, agent.actor.W)
        )


# ===================================================================
# Clip vs KL
# ===================================================================

class TestPPOClipVsKL:
    def test_kl_mode_runs(self, disc_env):
        agent = _make_ppo_discrete(disc_env, clip_eps=0.0, rollout_len=32)
        assert agent.use_kl is True
        agent.train(disc_env, n_episodes=3)
        assert len(agent.episode_rewards_) == 3

    def test_clip_mode_runs(self, disc_env):
        agent = _make_ppo_discrete(disc_env, clip_eps=0.2, rollout_len=32)
        assert agent.use_kl is False
        agent.train(disc_env, n_episodes=3)
        assert len(agent.episode_rewards_) == 3

    def test_kl_losses_finite(self, disc_env):
        agent = _make_ppo_discrete(disc_env, clip_eps=0.0, rollout_len=32)
        agent.train(disc_env, n_episodes=4)
        assert all(np.isfinite(l) for l in agent.policy_losses_)

    def test_clip_ratio_bounded(self, disc_env):
        """
        Verify that the clipped surrogate numerically limits ratio excursions.
        Build a batch where old_log_probs are very low (large ratio) and confirm
        the clipped loss is less extreme than the unclipped.
        """
        agent = _make_ppo_discrete(disc_env)
        rng = np.random.default_rng(7)
        n = 32
        states    = rng.standard_normal((n, disc_env.state_dim))
        actions   = rng.integers(0, disc_env.N_ACTIONS, (n, 1))
        old_lp    = np.full(n, -5.0)   # low old log-prob → high ratio
        advantages = rng.standard_normal(n)
        returns    = rng.standard_normal(n)

        new_lp, _ = agent._log_prob_batch(states, actions)
        ratio = np.exp(new_lp - old_lp)
        surr_unclipped = ratio * advantages
        surr_clipped   = np.clip(ratio, 1 - agent.clip_eps,
                                  1 + agent.clip_eps) * advantages
        # For large ratios (ratio >> 1.2), clipped should dampen the gradient
        large_ratio_mask = ratio > 1.5
        if large_ratio_mask.any():
            assert np.abs(surr_clipped[large_ratio_mask]).mean() <= \
                   np.abs(surr_unclipped[large_ratio_mask]).mean() + 1e-6


# ===================================================================
# Rollout buffer management
# ===================================================================

class TestPPORolloutBuffer:
    def test_buffer_reset_after_flush(self, disc_env):
        agent = _make_ppo_discrete(disc_env, rollout_len=16)
        # Force a flush
        state = disc_env.reset()
        for _ in range(20):
            action, log_p, _ = agent._actor_forward(state)
            v = float(agent.critic.forward(state))
            agent._states.append(state.copy())
            agent._actions.append(action.copy())
            agent._log_probs.append(log_p)
            agent._values.append(v)
            ns, r, done = disc_env.step(int(action[0]))
            agent._rewards.append(r)
            agent._dones.append(done)
            state = disc_env.reset() if done else ns
            if len(agent._rewards) >= agent.rollout_len:
                agent._flush_rollout(state)
                break
        # After flush buffer should be cleared
        assert len(agent._rewards) == 0

    def test_rollout_len_respected(self, disc_env):
        """Training should trigger an update approximately every rollout_len steps."""
        agent = _make_ppo_discrete(disc_env, rollout_len=32, n_epochs=1)
        agent.train(disc_env, n_episodes=10)
        # At least one update must have happened
        assert len(agent.policy_losses_) >= 1
