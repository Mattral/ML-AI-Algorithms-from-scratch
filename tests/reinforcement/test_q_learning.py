"""
Tests for mlscratch.reinforcement.q_learning
Covers: QLearning, DoubleQLearning, LinearQLearning
"""

from __future__ import annotations
import numpy as np
import pytest

from mlscratch.reinforcement.q_learning import (
    QLearning, DoubleQLearning, LinearQLearning
)
from mlscratch.reinforcement.utils import GridWorld


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def small_grid():
    return GridWorld(size=4, pit=(1, 1))


@pytest.fixture
def tiny_grid():
    """2×2 grid — no pit — trivially solvable."""
    return GridWorld(size=2, pit=(-1, -1))


# ===================================================================
# QLearning — Basic API
# ===================================================================

class TestQLearningBasic:
    def test_q_table_shape(self, small_grid):
        agent = QLearning(small_grid.n_states, small_grid.N_ACTIONS)
        assert agent.Q.shape == (16, 4)

    def test_q_table_initialised_to_zero(self, small_grid):
        agent = QLearning(small_grid.n_states, small_grid.N_ACTIONS)
        np.testing.assert_array_equal(agent.Q, 0.0)

    def test_select_action_in_valid_range(self, small_grid):
        agent = QLearning(small_grid.n_states, small_grid.N_ACTIONS)
        for _ in range(50):
            a = agent.select_action(0)
            assert 0 <= a < 4

    def test_greedy_action_is_deterministic(self, small_grid):
        agent = QLearning(small_grid.n_states, small_grid.N_ACTIONS)
        agent.Q[0, 2] = 1.0          # manually favour action 2
        actions = {agent.select_action(0, greedy=True) for _ in range(20)}
        assert actions == {2}

    def test_update_returns_float(self, small_grid):
        agent = QLearning(small_grid.n_states, small_grid.N_ACTIONS)
        td = agent.update(0, 1, -0.1, 1, False)
        assert isinstance(td, float)

    def test_update_modifies_q_value(self, small_grid):
        agent = QLearning(small_grid.n_states, small_grid.N_ACTIONS)
        agent.update(0, 1, 10.0, 15, True)
        assert agent.Q[0, 1] != 0.0

    def test_train_episode_returns_float(self, small_grid):
        agent = QLearning(small_grid.n_states, small_grid.N_ACTIONS,
                           random_state=0)
        r = agent.train_episode(small_grid)
        assert isinstance(r, float)

    def test_episode_rewards_recorded(self, small_grid):
        agent = QLearning(small_grid.n_states, small_grid.N_ACTIONS,
                           random_state=0)
        agent.train(small_grid, n_episodes=10)
        assert len(agent.episode_rewards_) == 10

    def test_epsilon_decays(self, small_grid):
        agent = QLearning(small_grid.n_states, small_grid.N_ACTIONS,
                           epsilon=1.0, epsilon_decay=0.9, random_state=0)
        initial_eps = agent.epsilon
        agent.train(small_grid, n_episodes=5)
        assert agent.epsilon < initial_eps

    def test_epsilon_lower_bounded(self, small_grid):
        agent = QLearning(small_grid.n_states, small_grid.N_ACTIONS,
                           epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5,
                           random_state=0)
        agent.train(small_grid, n_episodes=50)
        assert agent.epsilon >= 0.1 - 1e-10

    def test_train_returns_self(self, small_grid):
        agent = QLearning(small_grid.n_states, small_grid.N_ACTIONS)
        assert agent.train(small_grid, n_episodes=2) is agent


# ===================================================================
# QLearning — Correctness
# ===================================================================

class TestQLearningCorrectness:
    def test_terminal_update_ignores_next_state(self, small_grid):
        """Done=True → target = reward only (no γ * max Q')."""
        agent = QLearning(n_states=16, n_actions=4, alpha=1.0, gamma=0.99)
        agent.update(5, 2, 10.0, 15, done=True)
        np.testing.assert_allclose(agent.Q[5, 2], 10.0)

    def test_bellman_update_correctness(self, small_grid):
        """Manual verification of one TD update."""
        agent = QLearning(n_states=16, n_actions=4, alpha=0.5, gamma=0.9)
        agent.Q[1, 3] = 4.0
        agent.update(0, 1, 1.0, 1, done=False)
        # target = 1.0 + 0.9 * max(Q[1]) = 1.0 + 0.9 * 4.0 = 4.6
        # new Q  = 0 + 0.5 * (4.6 - 0) = 2.3
        np.testing.assert_allclose(agent.Q[0, 1], 2.3, atol=1e-8)

    def test_policy_argmax_of_q(self, small_grid):
        agent = QLearning(n_states=16, n_actions=4, random_state=0)
        agent.Q[5, 2] = 99.0
        assert agent.policy()[5] == 2

    def test_value_function_max_of_q(self, small_grid):
        agent = QLearning(n_states=16, n_actions=4, random_state=0)
        agent.Q[3, :] = [1.0, 2.0, 3.0, 4.0]
        assert agent.value_function()[3] == pytest.approx(4.0)

    def test_learns_optimal_policy_simple_grid(self, tiny_grid):
        """On a 2×2 grid the agent should reliably reach the goal."""
        agent = QLearning(n_states=4, n_actions=4, alpha=0.5, gamma=0.99,
                           epsilon=0.5, epsilon_min=0.05, epsilon_decay=0.99,
                           random_state=42)
        agent.train(tiny_grid, n_episodes=300)
        # Evaluate greedily
        rewards = []
        for seed in range(20):
            tiny_grid.reset()
            done = False
            total = 0.0
            steps = 0
            while not done and steps < 20:
                s = tiny_grid._encode(tiny_grid._state)
                a = agent.select_action(s, greedy=True)
                _, r, done = tiny_grid.step(a)
                total += r
                steps += 1
            rewards.append(total)
        mean_reward = np.mean(rewards)
        assert mean_reward > 0, f"Expected positive mean reward, got {mean_reward}"

    def test_q_values_non_zero_after_training(self, small_grid):
        agent = QLearning(n_states=16, n_actions=4, random_state=0)
        agent.train(small_grid, n_episodes=100)
        assert not np.all(agent.Q == 0.0)


# ===================================================================
# DoubleQLearning — Basic API & Properties
# ===================================================================

class TestDoubleQLearning:
    def test_two_q_tables(self, small_grid):
        agent = DoubleQLearning(16, 4)
        assert agent.Q_A.shape == (16, 4)
        assert agent.Q_B.shape == (16, 4)

    def test_combined_q_is_average(self, small_grid):
        agent = DoubleQLearning(16, 4)
        agent.Q_A[0, 0] = 2.0
        agent.Q_B[0, 0] = 4.0
        np.testing.assert_allclose(agent.Q[0, 0], 3.0)

    def test_train_episode_returns_float(self, small_grid):
        agent = DoubleQLearning(16, 4, random_state=0)
        assert isinstance(agent.train_episode(small_grid), float)

    def test_episode_rewards_tracked(self, small_grid):
        agent = DoubleQLearning(16, 4, random_state=0)
        agent.train(small_grid, n_episodes=5)
        assert len(agent.episode_rewards_) == 5

    def test_q_tables_differ_after_training(self, small_grid):
        """Q_A and Q_B should not be identical after training (different updates)."""
        agent = DoubleQLearning(16, 4, random_state=0)
        agent.train(small_grid, n_episodes=50)
        assert not np.allclose(agent.Q_A, agent.Q_B)

    def test_reduces_maximisation_bias_vs_qlearning(self):
        """
        In a biased stochastic environment, Double Q has lower max Q-value
        than standard Q — demonstrating the maximisation-bias reduction.
        """
        # Bandit: state=0, 4 actions, all true values = 0
        # Noisy rewards centered at 0 — QLearning should overestimate
        rng = np.random.default_rng(42)

        class BanditEnv:
            N_ACTIONS = 4
            def reset(self):  return 0
            def step(self, a):
                r = float(rng.normal(0, 1))    # noisy, true value = 0
                return 0, r, True

        env = BanditEnv()
        q_agent  = QLearning(1, 4, alpha=0.1, gamma=0.0, epsilon=0.5,
                              epsilon_decay=1.0, random_state=0)
        dq_agent = DoubleQLearning(1, 4, alpha=0.1, gamma=0.0, epsilon=0.5,
                                    epsilon_decay=1.0, random_state=0)
        for _ in range(1000):
            q_agent.train_episode(env)
            dq_agent.train_episode(env)

        q_max  = q_agent.Q.max()
        dq_max = dq_agent.Q.max()
        assert dq_max <= q_max + 0.5   # Double Q should not over-inflate


# ===================================================================
# LinearQLearning
# ===================================================================

class TestLinearQLearning:
    def test_weight_vector_shape(self, small_grid):
        agent = LinearQLearning(16, 4)
        assert agent.w.shape == (16 * 4,)

    def test_q_table_property_shape(self, small_grid):
        agent = LinearQLearning(16, 4)
        assert agent.Q.shape == (16, 4)

    def test_feature_vector_one_hot(self):
        agent = LinearQLearning(n_states=4, n_actions=3)
        phi = agent._features(1, 2)
        assert phi.shape == (12,)
        assert phi[1 * 3 + 2] == 1.0
        assert phi.sum() == 1.0

    def test_select_action_in_range(self, small_grid):
        agent = LinearQLearning(16, 4, random_state=0)
        for _ in range(20):
            assert 0 <= agent.select_action(0) < 4

    def test_update_changes_weights(self, small_grid):
        agent = LinearQLearning(16, 4, alpha=0.1)
        w_before = agent.w.copy()
        agent.update(0, 1, 5.0, 1, done=True)
        assert not np.allclose(agent.w, w_before)

    def test_train_episode_returns_float(self, small_grid):
        agent = LinearQLearning(16, 4, random_state=0)
        assert isinstance(agent.train_episode(small_grid), float)

    def test_episode_rewards_tracked(self, small_grid):
        agent = LinearQLearning(16, 4, random_state=0)
        agent.train(small_grid, n_episodes=5)
        assert len(agent.episode_rewards_) == 5

    def test_q_non_zero_after_training(self, small_grid):
        agent = LinearQLearning(16, 4, alpha=0.1, random_state=0)
        agent.train(small_grid, n_episodes=100)
        assert not np.all(agent.Q == 0.0)

    def test_returns_self_from_train(self, small_grid):
        agent = LinearQLearning(16, 4)
        assert agent.train(small_grid, n_episodes=2) is agent
