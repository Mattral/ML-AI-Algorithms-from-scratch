"""
mlscratch.reinforcement
========================
From-scratch implementations of Reinforcement Learning algorithms.
Pure numpy — no PyTorch, TensorFlow, or gym dependency required.

Algorithms
----------
QLearning                   – Tabular Q-Learning (Watkins & Dayan, 1992)
DoubleQLearning             – Double Q-Learning (van Hasselt, 2010)
LinearQLearning             – Q-Learning with linear function approximation
DQN                         – Deep Q-Network with Double DQN + Dueling + PER
DuelingMLP                  – Dueling network architecture (stand-alone)
DDPG                        – Deep Deterministic Policy Gradient
TD3                         – Twin Delayed DDPG (Fujimoto et al., 2018)
PPO                         – Proximal Policy Optimization (clip & KL variants)
SAC                         – Soft Actor-Critic with auto-entropy tuning

Shared utilities (mlscratch.reinforcement.utils)
-------------------------------------------------
GridWorld                   – Tabular grid-world environment
ContinuousEnv               – 1-D point-mass continuous control environment
DiscreteEnv                 – Discrete-action wrapper for ContinuousEnv
ReplayBuffer                – Uniform experience replay
PrioritizedReplayBuffer     – Prioritised experience replay (sum-tree)
MLP                         – Pure-numpy MLP with Adam + backprop
OrnsteinUhlenbeckNoise      – Temporally-correlated exploration noise
GaussianNoise               – i.i.d. Gaussian exploration noise
"""

from .q_learning import QLearning, DoubleQLearning, LinearQLearning   # noqa: F401
from .dqn import DQN, DuelingMLP                                       # noqa: F401
from .ddpg import DDPG, TD3                                            # noqa: F401
from .ppo import PPO                                                    # noqa: F401
from .sac import SAC                                                    # noqa: F401
from .utils import (                                                    # noqa: F401
    GridWorld,
    ContinuousEnv,
    DiscreteEnv,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    MLP,
    OrnsteinUhlenbeckNoise,
    GaussianNoise,
)

__all__ = [
    # Q-Learning family
    "QLearning", "DoubleQLearning", "LinearQLearning",
    # Deep RL
    "DQN", "DuelingMLP",
    "DDPG", "TD3",
    "PPO",
    "SAC",
    # Environments & utilities
    "GridWorld", "ContinuousEnv", "DiscreteEnv",
    "ReplayBuffer", "PrioritizedReplayBuffer",
    "MLP", "OrnsteinUhlenbeckNoise", "GaussianNoise",
]
