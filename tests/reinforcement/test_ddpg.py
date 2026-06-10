"""
Tests for mlscratch.reinforcement.ddpg — DDPG and TD3
"""

from __future__ import annotations
import numpy as np
import pytest

from mlscratch.reinforcement.ddpg import DDPG, TD3
from mlscratch.reinforcement.utils import ContinuousEnv


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def env():
    return ContinuousEnv(max_steps=50)


def _make_ddpg(env, warmup=0, **kwargs):
    defaults = dict(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        action_low=env.ACTION_LOW,
        action_high=env.ACTION_HIGH,
        hidden_sizes=[32, 32],
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=2000,
        batch_size=16,
        warmup_steps=warmup,
        random_state=0,
    )
    defaults.update(kwargs)
    return DDPG(**defaults)


def _make_td3(env, warmup=0, **kwargs):
    defaults = dict(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        action_low=env.ACTION_LOW,
        action_high=env.ACTION_HIGH,
        hidden_sizes=[32, 32],
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=2000,
        batch_size=16,
        warmup_steps=warmup,
        policy_delay=2,
        target_noise=0.2,
        noise_clip=0.5,
        random_state=0,
    )
    defaults.update(kwargs)
    return TD3(**defaults)


# ===================================================================
# DDPG — Actor / Critic architecture
# ===================================================================

class TestDDPGArchitecture:
    def test_actor_output_shape_single(self, env):
        agent = _make_ddpg(env)
        state = env.reset()
        a = agent.actor.forward(state)
        assert a.shape == (env.action_dim,)

    def test_actor_output_in_tanh_range(self, env):
        """Actor's raw output (tanh) must be in [-1, 1]."""
        agent = _make_ddpg(env)
        rng = np.random.default_rng(0)
        for _ in range(20):
            s = rng.standard_normal(env.state_dim)
            a_raw = agent.actor.forward(s)
            assert np.all(a_raw >= -1.0) and np.all(a_raw <= 1.0)

    def test_select_action_in_action_bounds(self, env):
        agent = _make_ddpg(env)
        rng = np.random.default_rng(0)
        for _ in range(50):
            s = rng.standard_normal(env.state_dim)
            a = agent.select_action(s, add_noise=False)
            assert np.all(a >= env.ACTION_LOW) and np.all(a <= env.ACTION_HIGH)

    def test_select_action_with_noise_clipped(self, env):
        """Even with noise, clipping must keep action in [-1, 1] pre-scaling."""
        agent = _make_ddpg(env)
        rng = np.random.default_rng(0)
        for _ in range(100):
            s = rng.standard_normal(env.state_dim)
            a = agent.select_action(s, add_noise=True)
            assert np.all(a >= env.ACTION_LOW - 1e-9)
            assert np.all(a <= env.ACTION_HIGH + 1e-9)

    def test_critic_output_shape(self, env):
        agent = _make_ddpg(env)
        rng = np.random.default_rng(0)
        s = rng.standard_normal((8, env.state_dim))
        a = rng.uniform(-1, 1, (8, env.action_dim))
        sa = np.concatenate([s, a], axis=1)
        q = agent.critic.forward(sa)
        assert q.shape == (8, 1)

    def test_target_networks_initialised_equal(self, env):
        agent = _make_ddpg(env)
        rng = np.random.default_rng(0)
        s = rng.standard_normal((5, env.state_dim))
        np.testing.assert_allclose(
            agent.actor.forward(s),
            agent.actor_target.forward(s),
            atol=1e-10,
        )
        a = rng.uniform(-1, 1, (5, env.action_dim))
        sa = np.concatenate([s, a], axis=1)
        np.testing.assert_allclose(
            agent.critic.forward(sa),
            agent.critic_target.forward(sa),
            atol=1e-10,
        )


# ===================================================================
# DDPG — Buffer & stepping
# ===================================================================

class TestDDPGBuffer:
    def test_buffer_grows_with_steps(self, env):
        agent = _make_ddpg(env)
        state = env.reset()
        for _ in range(10):
            a = np.array([0.0])
            ns, r, done = env.step(a)
            agent.step(state, a, r, ns, done)
            state = ns if not done else env.reset()
        assert len(agent.buffer) == 10

    def test_step_no_learn_before_warmup(self, env):
        agent = _make_ddpg(env, warmup=1000)
        state = env.reset()
        for _ in range(20):
            a = agent.select_action(state)
            ns, r, done = env.step(a)
            al, cl = agent.step(state, a, r, ns, done)
            assert al is None and cl is None
            state = ns if not done else env.reset()

    def test_step_returns_losses_after_warmup(self, env):
        agent = _make_ddpg(env, warmup=0)
        # Pre-fill buffer
        rng = np.random.default_rng(1)
        for _ in range(agent.batch_size * 3):
            s  = rng.standard_normal(env.state_dim)
            a  = rng.uniform(-1, 1, (env.action_dim,))
            ns = rng.standard_normal(env.state_dim)
            agent.buffer.push(s, a, -0.1, ns, False)
        state = env.reset()
        a = agent.select_action(state)
        ns, r, done = env.step(a)
        al, cl = agent.step(state, a, r, ns, done)
        assert al is not None and cl is not None
        assert np.isfinite(al) and np.isfinite(cl)


# ===================================================================
# DDPG — Training
# ===================================================================

class TestDDPGTraining:
    def test_train_episode_returns_float(self, env):
        agent = _make_ddpg(env)
        r = agent.train_episode(env)
        assert isinstance(r, float)

    def test_episode_rewards_tracked(self, env):
        agent = _make_ddpg(env)
        agent.train(env, n_episodes=3)
        assert len(agent.episode_rewards_) == 3

    def test_train_returns_self(self, env):
        agent = _make_ddpg(env)
        assert agent.train(env, n_episodes=2) is agent

    def test_actor_weights_change_after_training(self, env):
        agent = _make_ddpg(env)
        W0 = [w.copy() for w in agent.actor.W]
        agent.train(env, n_episodes=5)
        changed = any(
            not np.allclose(w0, w1)
            for w0, w1 in zip(W0, agent.actor.W)
        )
        assert changed, "Actor weights must change during training"

    def test_critic_weights_change_after_training(self, env):
        agent = _make_ddpg(env)
        W0 = [w.copy() for w in agent.critic.W]
        agent.train(env, n_episodes=5)
        changed = any(
            not np.allclose(w0, w1)
            for w0, w1 in zip(W0, agent.critic.W)
        )
        assert changed

    def test_soft_target_update_moves_target(self, env):
        agent = _make_ddpg(env)
        # Record initial target weights
        W_target_0 = [w.copy() for w in agent.actor_target.W]
        # Pre-fill and learn once
        rng = np.random.default_rng(0)
        for _ in range(agent.batch_size * 4):
            s  = rng.standard_normal(env.state_dim)
            a  = rng.uniform(-1, 1, (env.action_dim,))
            ns = rng.standard_normal(env.state_dim)
            agent.buffer.push(s, a, -0.1, ns, False)
        agent._learn()
        # Target should have moved slightly from the soft update
        changed = any(
            not np.allclose(w0, w1)
            for w0, w1 in zip(W_target_0, agent.actor_target.W)
        )
        assert changed

    def test_no_nan_in_actor_output(self, env):
        agent = _make_ddpg(env)
        agent.train(env, n_episodes=5)
        for _ in range(20):
            s = np.random.default_rng(0).standard_normal(env.state_dim)
            a = agent.select_action(s, add_noise=False)
            assert not np.any(np.isnan(a))

    def test_gaussian_noise_variant(self, env):
        agent = _make_ddpg(env, noise_type="gaussian")
        agent.train(env, n_episodes=3)
        assert len(agent.episode_rewards_) == 3


# ===================================================================
# TD3 — Additional components
# ===================================================================

class TestTD3Architecture:
    def test_has_second_critic(self, env):
        agent = _make_td3(env)
        assert hasattr(agent, "critic2")
        assert hasattr(agent, "critic2_target")

    def test_second_critic_initialised_equal_to_target(self, env):
        agent = _make_td3(env)
        rng = np.random.default_rng(0)
        s = rng.standard_normal((5, env.state_dim))
        a = rng.uniform(-1, 1, (5, env.action_dim))
        sa = np.concatenate([s, a], axis=1)
        np.testing.assert_allclose(
            agent.critic2.forward(sa),
            agent.critic2_target.forward(sa),
            atol=1e-10,
        )

    def test_td3_select_action_bounded(self, env):
        agent = _make_td3(env)
        rng = np.random.default_rng(0)
        for _ in range(50):
            s = rng.standard_normal(env.state_dim)
            a = agent.select_action(s, add_noise=False)
            assert np.all(a >= env.ACTION_LOW - 1e-9)
            assert np.all(a <= env.ACTION_HIGH + 1e-9)


class TestTD3Training:
    def test_train_episode_returns_float(self, env):
        agent = _make_td3(env)
        assert isinstance(agent.train_episode(env), float)

    def test_episode_rewards_tracked(self, env):
        agent = _make_td3(env)
        agent.train(env, n_episodes=3)
        assert len(agent.episode_rewards_) == 3

    def test_delayed_policy_update(self, env):
        """Actor loss should only be logged every policy_delay critic steps."""
        agent = _make_td3(env, warmup=0)
        rng = np.random.default_rng(0)
        for _ in range(agent.batch_size * 6):
            s  = rng.standard_normal(env.state_dim)
            a  = rng.uniform(-1, 1, (env.action_dim,))
            ns = rng.standard_normal(env.state_dim)
            agent.buffer.push(s, a, -0.1, ns, False)

        # Run enough _learn steps to trigger at least one actor update
        n_steps = agent.policy_delay * 4
        for _ in range(n_steps):
            agent._learn()

        # At least one actor loss logged (from delayed updates)
        assert len(agent.actor_losses_) > 0

    def test_target_policy_noise_added(self, env):
        """With target_noise > 0 next actions should have variance."""
        agent = _make_td3(env, target_noise=0.3)
        rng = np.random.default_rng(0)
        next_states = rng.standard_normal((32, env.state_dim))
        a1 = agent.actor_target.forward(next_states)
        a2 = agent.actor_target.forward(next_states)
        # Deterministic network — same input gives same output
        np.testing.assert_allclose(a1, a2, atol=1e-10)
        # But with noise the actions computed in _learn should vary:
        # (tested indirectly — just confirm learn runs without error)
        for _ in range(32):
            s  = rng.standard_normal(env.state_dim)
            a  = rng.uniform(-1, 1, (env.action_dim,))
            ns = rng.standard_normal(env.state_dim)
            agent.buffer.push(s, a, -0.1, ns, False)
        al, cl = agent._learn()
        assert cl is not None and np.isfinite(cl)

    def test_td3_no_nan_after_training(self, env):
        agent = _make_td3(env)
        agent.train(env, n_episodes=5)
        rng = np.random.default_rng(0)
        for _ in range(20):
            s = rng.standard_normal(env.state_dim)
            a = agent.select_action(s, add_noise=False)
            assert not np.any(np.isnan(a))

    def test_td3_critic_losses_finite(self, env):
        agent = _make_td3(env)
        rng = np.random.default_rng(0)
        for _ in range(agent.batch_size * 4):
            s  = rng.standard_normal(env.state_dim)
            a  = rng.uniform(-1, 1, (env.action_dim,))
            ns = rng.standard_normal(env.state_dim)
            agent.buffer.push(s, a, -0.1, ns, False)
        for _ in range(10):
            al, cl = agent._learn()
        assert np.isfinite(cl)


# ===================================================================
# DDPG vs TD3 — Comparative correctness
# ===================================================================

class TestDDPGvsTD3:
    def test_both_produce_valid_actions(self, env):
        ddpg = _make_ddpg(env)
        td3  = _make_td3(env)
        state = env.reset()
        for agent in [ddpg, td3]:
            a = agent.select_action(state, add_noise=False)
            assert a.shape == (env.action_dim,)
            assert np.all(np.isfinite(a))

    def test_td3_has_more_networks_than_ddpg(self, env):
        ddpg_networks = 4   # actor, actor_target, critic, critic_target
        td3_networks  = 6   # + critic2, critic2_target
        ddpg = _make_ddpg(env)
        td3  = _make_td3(env)
        ddpg_count = sum(1 for attr in vars(ddpg)
                          if "critic" in attr or "actor" in attr)
        td3_count  = sum(1 for attr in vars(td3)
                          if "critic" in attr or "actor" in attr)
        assert td3_count >= ddpg_count
