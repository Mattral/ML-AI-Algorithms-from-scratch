"""
Tests for mlscratch.reinforcement.utils

Covers: GridWorld, ContinuousEnv, DiscreteEnv,
        ReplayBuffer, PrioritizedReplayBuffer,
        MLP, OrnsteinUhlenbeckNoise, GaussianNoise.
"""

from __future__ import annotations
import numpy as np
import pytest

from mlscratch.reinforcement.utils import (
    GridWorld, ContinuousEnv, DiscreteEnv,
    ReplayBuffer, PrioritizedReplayBuffer,
    MLP, OrnsteinUhlenbeckNoise, GaussianNoise,
)


# ===================================================================
# GridWorld
# ===================================================================

class TestGridWorld:
    def test_reset_returns_integer_state(self):
        env = GridWorld()
        s = env.reset()
        assert isinstance(s, (int, np.integer))

    def test_reset_returns_start_state(self):
        env = GridWorld(size=4)
        assert env.reset() == 0     # (0,0) → 0*4+0 = 0

    def test_step_returns_triple(self):
        env = GridWorld()
        env.reset()
        result = env.step(1)        # down
        assert len(result) == 3

    def test_step_reward_is_float(self):
        env = GridWorld()
        env.reset()
        _, r, _ = env.step(0)
        assert isinstance(r, float)

    def test_step_done_is_bool(self):
        env = GridWorld()
        env.reset()
        _, _, done = env.step(0)
        assert isinstance(done, bool)

    def test_goal_gives_positive_reward_and_terminates(self):
        env = GridWorld(size=2, pit=(-1, -1))     # no pit
        # S=(0,0), G=(1,1) — navigate: down then right
        env.reset()
        env.step(1)   # down  → (1,0)
        _, r, done = env.step(3)   # right → (1,1) = goal
        assert r == pytest.approx(10.0)
        assert done is True

    def test_pit_gives_negative_reward_and_terminates(self):
        env = GridWorld(size=4, pit=(1, 1))
        env.reset()
        env.step(1)   # down  → (1,0)
        _, r, done = env.step(3)   # right → (1,1) = pit
        assert r == pytest.approx(-10.0)
        assert done is True

    def test_step_reward_during_navigation(self):
        env = GridWorld()
        env.reset()
        _, r, done = env.step(3)   # right (not goal/pit)
        assert r == pytest.approx(-0.1)
        assert done is False

    def test_n_states(self):
        assert GridWorld(size=4).n_states == 16
        assert GridWorld(size=5).n_states == 25

    def test_action_clips_to_boundary(self):
        env = GridWorld(size=4)
        env.reset()                 # at (0,0)
        s, _, _ = env.step(0)      # up → stays at (0,0)
        assert s == 0

    def test_render_returns_string(self):
        env = GridWorld()
        env.reset()
        assert isinstance(env.render(), str)


# ===================================================================
# ContinuousEnv
# ===================================================================

class TestContinuousEnv:
    def test_reset_shape(self):
        env = ContinuousEnv()
        s = env.reset()
        assert s.shape == (2,)

    def test_reset_reproducible(self):
        rng = np.random.default_rng(0)
        env = ContinuousEnv()
        s1 = env.reset(rng)
        rng = np.random.default_rng(0)
        s2 = env.reset(rng)
        np.testing.assert_array_equal(s1, s2)

    def test_step_returns_triple(self):
        env = ContinuousEnv()
        env.reset()
        result = env.step(np.array([0.0]))
        assert len(result) == 3

    def test_step_next_state_shape(self):
        env = ContinuousEnv()
        env.reset()
        ns, _, _ = env.step(np.array([0.5]))
        assert ns.shape == (2,)

    def test_step_reward_is_non_positive(self):
        env = ContinuousEnv()
        env.reset()
        _, r, _ = env.step(np.array([0.0]))
        assert r <= 0.0

    def test_done_after_max_steps(self):
        env = ContinuousEnv(max_steps=3)
        env.reset()
        for _ in range(2):
            _, _, done = env.step(np.array([0.0]))
            assert done is False
        _, _, done = env.step(np.array([0.0]))
        assert done is True

    def test_action_clipping(self):
        env = ContinuousEnv()
        s0 = env.reset()
        s_clipped, r1, _ = env.step(np.array([999.0]))
        env.reset()
        for _ in range(1):
            env._state = s0.copy()
        s_exact, r2, _ = env.step(np.array([1.0]))
        np.testing.assert_allclose(s_clipped, s_exact, atol=1e-10)

    def test_state_dim_property(self):
        assert ContinuousEnv().state_dim == 2

    def test_action_dim_property(self):
        assert ContinuousEnv().action_dim == 1


# ===================================================================
# DiscreteEnv
# ===================================================================

class TestDiscreteEnv:
    def test_reset_shape(self):
        s = DiscreteEnv().reset()
        assert s.shape == (2,)

    def test_step_valid_action(self):
        env = DiscreteEnv()
        env.reset()
        ns, r, done = env.step(2)
        assert ns.shape == (2,)
        assert isinstance(r, float)
        assert isinstance(done, bool)

    def test_n_actions(self):
        assert DiscreteEnv.N_ACTIONS == 5

    def test_all_actions_valid(self):
        env = DiscreteEnv()
        env.reset()
        for a in range(DiscreteEnv.N_ACTIONS):
            env.reset()
            env.step(a)     # no error


# ===================================================================
# ReplayBuffer
# ===================================================================

class TestReplayBuffer:
    @pytest.fixture
    def buf(self):
        return ReplayBuffer(capacity=100)

    def _fill(self, buf, n=10):
        rng = np.random.default_rng(0)
        for i in range(n):
            buf.push(
                rng.standard_normal(4),
                rng.standard_normal(2),
                float(rng.standard_normal()),
                rng.standard_normal(4),
                bool(i == n - 1),
            )

    def test_len_zero_initially(self, buf):
        assert len(buf) == 0

    def test_len_after_push(self, buf):
        self._fill(buf, 5)
        assert len(buf) == 5

    def test_capacity_capped(self):
        buf = ReplayBuffer(capacity=5)
        self._fill(buf, 10)
        assert len(buf) == 5

    def test_sample_shapes(self, buf):
        self._fill(buf, 20)
        s, a, r, ns, d = buf.sample(8, np.random.default_rng(0))
        assert s.shape  == (8, 4)
        assert a.shape  == (8, 2)
        assert r.shape  == (8,)
        assert ns.shape == (8, 4)
        assert d.shape  == (8,)

    def test_sample_reproducible(self, buf):
        self._fill(buf, 30)
        s1, _, _, _, _ = buf.sample(5, np.random.default_rng(7))
        s2, _, _, _, _ = buf.sample(5, np.random.default_rng(7))
        np.testing.assert_array_equal(s1, s2)

    def test_done_values_binary(self, buf):
        self._fill(buf, 20)
        _, _, _, _, d = buf.sample(10, np.random.default_rng(0))
        assert np.all((d == 0.0) | (d == 1.0))

    def test_rewards_are_float(self, buf):
        self._fill(buf, 10)
        _, _, r, _, _ = buf.sample(5, np.random.default_rng(0))
        assert r.dtype == np.float32


# ===================================================================
# PrioritizedReplayBuffer
# ===================================================================

class TestPrioritizedReplayBuffer:
    @pytest.fixture
    def pbuf(self):
        return PrioritizedReplayBuffer(capacity=200)

    def _fill(self, buf, n=50):
        rng = np.random.default_rng(0)
        for i in range(n):
            buf.push(
                rng.standard_normal(3),
                rng.standard_normal(1),
                float(rng.standard_normal()),
                rng.standard_normal(3),
                False,
            )

    def test_len_after_push(self, pbuf):
        self._fill(pbuf, 30)
        assert len(pbuf) == 30

    def test_sample_returns_7_tuple(self, pbuf):
        self._fill(pbuf, 60)
        result = pbuf.sample(16, np.random.default_rng(0))
        assert len(result) == 7

    def test_is_weights_in_unit_interval(self, pbuf):
        self._fill(pbuf, 60)
        *_, weights, _ = pbuf.sample(16, np.random.default_rng(0))
        assert np.all(weights >= 0) and np.all(weights <= 1.0 + 1e-6)

    def test_is_weights_max_is_one(self, pbuf):
        self._fill(pbuf, 60)
        *_, weights, _ = pbuf.sample(16, np.random.default_rng(0))
        assert abs(weights.max() - 1.0) < 1e-6

    def test_update_priorities_no_error(self, pbuf):
        self._fill(pbuf, 60)
        *_, idxs = pbuf.sample(16, np.random.default_rng(0))
        pbuf.update_priorities(idxs, np.ones(16) * 0.5)

    def test_beta_anneals_toward_one(self, pbuf):
        self._fill(pbuf, 60)
        beta_start = pbuf.beta
        for _ in range(100):
            pbuf.sample(16, np.random.default_rng(0))
        assert pbuf.beta > beta_start

    def test_beta_never_exceeds_one(self, pbuf):
        self._fill(pbuf, 60)
        for _ in range(10000):
            pbuf.sample(8, np.random.default_rng(0))
        assert pbuf.beta <= 1.0 + 1e-9


# ===================================================================
# MLP
# ===================================================================

class TestMLP:
    @pytest.fixture
    def net(self):
        return MLP([4, 16, 8, 2], output_activation="linear",
                   lr=1e-3, random_state=0)

    def test_forward_shape_batch(self, net):
        X = np.random.default_rng(0).standard_normal((10, 4))
        out = net.forward(X)
        assert out.shape == (10, 2)

    def test_forward_shape_scalar(self, net):
        x = np.random.default_rng(0).standard_normal(4)
        out = net.forward(x)
        assert out.shape == (2,)

    def test_tanh_output_bounded(self):
        net = MLP([3, 8, 2], output_activation="tanh", random_state=0)
        X = np.random.default_rng(0).standard_normal((50, 3)) * 100
        out = net.forward(X)
        assert np.all(out >= -1.0) and np.all(out <= 1.0)

    def test_sigmoid_output_bounded(self):
        net = MLP([3, 8, 1], output_activation="sigmoid", random_state=0)
        X = np.random.default_rng(0).standard_normal((20, 3)) * 100
        out = net.forward(X)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_backward_changes_weights(self, net):
        X = np.random.default_rng(0).standard_normal((8, 4))
        W_before = [w.copy() for w in net.W]
        net.forward(X, training=True)
        net.backward(np.ones((8, 2)) * 0.01)
        for w_b, w_a in zip(W_before, net.W):
            assert not np.allclose(w_b, w_a), "Weights should change after backward"

    def test_soft_update_interpolates(self):
        src = MLP([4, 8, 2], random_state=0)
        tgt = MLP([4, 8, 2], random_state=1)
        W0 = [w.copy() for w in tgt.W]
        src.soft_update(tgt, tau=0.5)
        for w_s, w_t, w0 in zip(src.W, tgt.W, W0):
            expected = 0.5 * w_s + 0.5 * w0
            np.testing.assert_allclose(w_t, expected, atol=1e-10)

    def test_hard_update_copies_exactly(self):
        src = MLP([4, 8, 2], random_state=0)
        tgt = MLP([4, 8, 2], random_state=99)
        src.hard_update(tgt)
        for w_s, w_t in zip(src.W, tgt.W):
            np.testing.assert_array_equal(w_s, w_t)

    def test_mlp_learns_simple_function(self):
        """MLP should reduce loss on y = 2x over 200 gradient steps."""
        rng = np.random.default_rng(0)
        net = MLP([1, 16, 1], output_activation="linear", lr=0.01, random_state=0)
        X = rng.standard_normal((32, 1))
        y = 2.0 * X

        losses = []
        for _ in range(200):
            pred = net.forward(X, training=True)
            loss = float(np.mean((pred - y) ** 2))
            losses.append(loss)
            net.backward(2.0 * (pred - y) / len(X))

        assert losses[-1] < losses[0], "Loss should decrease during training"
        assert losses[-1] < 0.5

    def test_no_nan_after_training(self, net):
        rng = np.random.default_rng(0)
        X   = rng.standard_normal((16, 4))
        for _ in range(50):
            out = net.forward(X, training=True)
            net.backward(np.ones_like(out) * 0.001)
        out = net.forward(X)
        assert not np.any(np.isnan(out))


# ===================================================================
# OrnsteinUhlenbeckNoise
# ===================================================================

class TestOUNoise:
    def test_sample_shape(self):
        noise = OrnsteinUhlenbeckNoise(size=3, random_state=0)
        s = noise.sample()
        assert s.shape == (3,)

    def test_reset_returns_to_mean(self):
        noise = OrnsteinUhlenbeckNoise(size=2, mu=0.0, random_state=0)
        for _ in range(100):
            noise.sample()
        noise.reset()
        np.testing.assert_array_equal(noise.x, np.zeros(2))

    def test_samples_are_correlated(self):
        """Consecutive OU samples should be correlated (not independent)."""
        noise = OrnsteinUhlenbeckNoise(size=1, theta=0.01, sigma=0.1,
                                        random_state=42)
        samples = [noise.sample()[0] for _ in range(200)]
        # First-order autocorrelation should be substantially positive
        corr = np.corrcoef(samples[:-1], samples[1:])[0, 1]
        assert corr > 0.5

    def test_reproducible(self):
        n1 = OrnsteinUhlenbeckNoise(size=3, random_state=5)
        n2 = OrnsteinUhlenbeckNoise(size=3, random_state=5)
        s1 = np.stack([n1.sample() for _ in range(10)])
        s2 = np.stack([n2.sample() for _ in range(10)])
        np.testing.assert_array_equal(s1, s2)


# ===================================================================
# GaussianNoise
# ===================================================================

class TestGaussianNoise:
    def test_sample_shape(self):
        noise = GaussianNoise(size=4, random_state=0)
        assert noise.sample().shape == (4,)

    def test_sigma_decays(self):
        noise = GaussianNoise(size=1, sigma=1.0, sigma_min=0.01,
                               decay=0.9, random_state=0)
        for _ in range(50):
            noise.sample()
        assert noise.sigma < 1.0

    def test_sigma_never_below_min(self):
        noise = GaussianNoise(size=1, sigma=1.0, sigma_min=0.1,
                               decay=0.5, random_state=0)
        for _ in range(1000):
            noise.sample()
        assert noise.sigma >= 0.1 - 1e-10

    def test_no_decay_when_decay_is_one(self):
        noise = GaussianNoise(size=2, sigma=0.5, decay=1.0, random_state=0)
        for _ in range(100):
            noise.sample()
        assert abs(noise.sigma - 0.5) < 1e-10
