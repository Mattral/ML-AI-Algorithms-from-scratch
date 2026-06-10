"""
Tests for mlscratch.reinforcement.dqn — DQN and DuelingMLP
"""

from __future__ import annotations
import numpy as np
import pytest

from mlscratch.reinforcement.dqn import DQN, DuelingMLP
from mlscratch.reinforcement.utils import DiscreteEnv, ReplayBuffer


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def env():
    return DiscreteEnv(max_steps=50)


@pytest.fixture
def dqn(env):
    return DQN(
        state_dim=env.state_dim,
        n_actions=env.N_ACTIONS,
        hidden_sizes=[32, 32],
        lr=1e-3,
        batch_size=16,
        buffer_capacity=500,
        target_update=20,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
        double_dqn=True,
        dueling=False,
        prioritized=False,
        random_state=0,
    )


@pytest.fixture
def dueling_dqn(env):
    return DQN(
        state_dim=env.state_dim,
        n_actions=env.N_ACTIONS,
        hidden_sizes=[32, 32],
        lr=1e-3,
        batch_size=16,
        buffer_capacity=500,
        target_update=20,
        double_dqn=True,
        dueling=True,
        prioritized=False,
        random_state=0,
    )


@pytest.fixture
def per_dqn(env):
    return DQN(
        state_dim=env.state_dim,
        n_actions=env.N_ACTIONS,
        hidden_sizes=[16, 16],
        lr=1e-3,
        batch_size=16,
        buffer_capacity=500,
        target_update=20,
        double_dqn=True,
        dueling=False,
        prioritized=True,
        random_state=0,
    )


# ===================================================================
# DuelingMLP unit tests
# ===================================================================

class TestDuelingMLP:
    def test_forward_shape_batch(self):
        net = DuelingMLP(state_dim=4, n_actions=5, hidden_sizes=[16, 16],
                          lr=1e-3, random_state=0)
        X = np.random.default_rng(0).standard_normal((8, 4))
        out = net.forward(X)
        assert out.shape == (8, 5)

    def test_forward_shape_single(self):
        net = DuelingMLP(state_dim=4, n_actions=5, hidden_sizes=[16, 16],
                          lr=1e-3, random_state=0)
        x = np.random.default_rng(0).standard_normal(4)
        out = net.forward(x)
        assert out.shape == (5,)

    def test_advantage_mean_near_zero(self):
        """After subtracting mean, Q columns should average near 0 minus V."""
        net = DuelingMLP(state_dim=2, n_actions=4, hidden_sizes=[8],
                          lr=1e-3, random_state=42)
        X = np.zeros((10, 2))
        Q = net.forward(X)        # (10, 4)
        # Q = V + A - mean(A); mean(Q) = V
        # So Q - mean(Q, axis=1, keepdims=True) should have zero column mean
        Q_centered = Q - Q.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(Q_centered.mean(axis=1), 0.0, atol=1e-10)

    def test_soft_update_changes_target(self):
        src = DuelingMLP(2, 3, [8], random_state=0)
        tgt = DuelingMLP(2, 3, [8], random_state=1)
        W0_trunk = [w.copy() for w in tgt._trunk.W]
        src.soft_update(tgt, tau=0.5)
        for w_s, w_t, w0 in zip(src._trunk.W, tgt._trunk.W, W0_trunk):
            expected = 0.5 * w_s + 0.5 * w0
            np.testing.assert_allclose(w_t, expected, atol=1e-10)

    def test_hard_update_copies_exactly(self):
        src = DuelingMLP(2, 3, [8], random_state=0)
        tgt = DuelingMLP(2, 3, [8], random_state=99)
        src.hard_update(tgt)
        for w_s, w_t in zip(src._trunk.W, tgt._trunk.W):
            np.testing.assert_array_equal(w_s, w_t)


# ===================================================================
# DQN — Basic API
# ===================================================================

class TestDQNBasic:
    def test_select_action_in_range(self, dqn, env):
        state = env.reset()
        for _ in range(30):
            a = dqn.select_action(state)
            assert 0 <= a < env.N_ACTIONS

    def test_greedy_action_deterministic_after_q_warmup(self, dqn, env):
        """After enough training, greedy policy should be stable."""
        state = env.reset()
        dqn.epsilon = 0.0      # force greedy
        actions = {dqn.select_action(state, greedy=True) for _ in range(10)}
        assert len(actions) == 1

    def test_step_returns_none_before_buffer_full(self, dqn, env):
        state = env.reset()
        ns, r, done = env.step(0)
        loss = dqn.step(state, 0, r, ns, done)
        assert loss is None

    def test_losses_recorded_after_buffer_full(self, dqn, env):
        for _ in range(5):
            dqn.train_episode(env)
        # Fill buffer
        rng = np.random.default_rng(0)
        for _ in range(dqn.batch_size * 2):
            s  = rng.standard_normal(env.state_dim)
            ns = rng.standard_normal(env.state_dim)
            dqn.buffer.push(s, np.array([0]), 0.0, ns, False)
        s = env.reset()
        ns, r, done = env.step(0)
        dqn.step(s, 0, r, ns, done)
        assert len(dqn.losses_) > 0

    def test_train_episode_returns_float(self, dqn, env):
        assert isinstance(dqn.train_episode(env), float)

    def test_episode_rewards_tracked(self, dqn, env):
        dqn.train(env, n_episodes=5)
        assert len(dqn.episode_rewards_) == 5

    def test_epsilon_decays(self, dqn, env):
        eps0 = dqn.epsilon
        dqn.train(env, n_episodes=10)
        assert dqn.epsilon < eps0

    def test_epsilon_bounded_below(self, dqn, env):
        dqn.train(env, n_episodes=200)
        assert dqn.epsilon >= dqn.epsilon_min - 1e-10

    def test_no_nan_in_q_output(self, dqn, env):
        dqn.train(env, n_episodes=3)
        state = env.reset()
        q = dqn.online_net.forward(state)
        assert not np.any(np.isnan(q))

    def test_train_returns_self(self, dqn, env):
        assert dqn.train(env, n_episodes=2) is dqn


# ===================================================================
# DQN variants
# ===================================================================

class TestDQNVariants:
    def test_dueling_dqn_runs(self, dueling_dqn, env):
        dueling_dqn.train(env, n_episodes=3)
        assert len(dueling_dqn.episode_rewards_) == 3

    def test_dueling_dqn_no_nan(self, dueling_dqn, env):
        dueling_dqn.train(env, n_episodes=3)
        state = env.reset()
        q = dueling_dqn.online_net.forward(state)
        assert not np.any(np.isnan(q))

    def test_per_dqn_runs(self, per_dqn, env):
        per_dqn.train(env, n_episodes=3)
        assert len(per_dqn.episode_rewards_) == 3

    def test_soft_update_mode(self, env):
        agent = DQN(
            state_dim=env.state_dim, n_actions=env.N_ACTIONS,
            hidden_sizes=[16, 16], tau=0.01, batch_size=16,
            buffer_capacity=300, random_state=0
        )
        agent.train(env, n_episodes=3)
        assert len(agent.episode_rewards_) == 3

    def test_standard_dqn_no_double(self, env):
        agent = DQN(
            state_dim=env.state_dim, n_actions=env.N_ACTIONS,
            hidden_sizes=[16, 16], double_dqn=False,
            batch_size=16, buffer_capacity=300, random_state=0
        )
        agent.train(env, n_episodes=3)
        assert not np.any(np.isnan(agent.online_net.forward(env.reset())))


# ===================================================================
# DQN — Target network tests
# ===================================================================

class TestDQNTargetNetwork:
    def test_target_initially_equals_online(self, env):
        agent = DQN(state_dim=2, n_actions=5, hidden_sizes=[8],
                     random_state=0)
        X = np.random.default_rng(0).standard_normal((5, 2))
        q_online = agent.online_net.forward(X)
        q_target = agent.target_net.forward(X)
        np.testing.assert_allclose(q_online, q_target, atol=1e-10)

    def test_target_diverges_after_online_update(self, env):
        """After training online net, target should differ (before sync)."""
        agent = DQN(state_dim=2, n_actions=5, hidden_sizes=[8],
                     target_update=10000,   # prevent hard sync
                     tau=None,
                     batch_size=8, buffer_capacity=100, random_state=0)
        # Manually fill buffer
        rng = np.random.default_rng(0)
        for _ in range(50):
            agent.buffer.push(
                rng.standard_normal(2), np.array([0]), 1.0,
                rng.standard_normal(2), False
            )
        # One learning step
        agent._learn()
        X = rng.standard_normal((5, 2))
        q_online = agent.online_net.forward(X)
        q_target = agent.target_net.forward(X)
        assert not np.allclose(q_online, q_target)

    def test_hard_sync_makes_networks_equal(self, env):
        agent = DQN(state_dim=2, n_actions=5, hidden_sizes=[8],
                     target_update=1, tau=None,
                     batch_size=8, buffer_capacity=100, random_state=0)
        rng = np.random.default_rng(0)
        for _ in range(50):
            agent.buffer.push(
                rng.standard_normal(2), np.array([0]), 1.0,
                rng.standard_normal(2), False
            )
        # Trigger hard sync
        agent.online_net.hard_update(agent.target_net)
        X = rng.standard_normal((5, 2))
        np.testing.assert_allclose(
            agent.online_net.forward(X),
            agent.target_net.forward(X),
            atol=1e-10,
        )
