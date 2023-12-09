import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class Actor(nn.Module):
    """Actor network for DDPG.

    Args:
        state_dim (int): Dimensionality of the input state space.
        action_dim (int): Dimensionality of the output action space.
        max_action (float): Upper bound for the output actions.

    Attributes:
        layer1 (nn.Linear): First linear layer.
        layer2 (nn.Linear): Second linear layer.
        layer3 (nn.Linear): Third linear layer.
        max_action (float): Upper bound for the output actions.
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        """Forward pass of the actor network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output action tensor.
        """
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = self.max_action * torch.tanh(self.layer3(x))
        return x

class Critic(nn.Module):
    """Critic network for DDPG.

    Args:
        state_dim (int): Dimensionality of the input state space.
        action_dim (int): Dimensionality of the input action space.

    Attributes:
        layer1 (nn.Linear): First linear layer.
        layer2 (nn.Linear): Second linear layer.
        layer3 (nn.Linear): Third linear layer.
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

    def forward(self, state, action):
        """Forward pass of the critic network.

        Args:
            state (torch.Tensor): Input state tensor.
            action (torch.Tensor): Input action tensor.

        Returns:
            torch.Tensor: Output value tensor.
        """
        x = torch.cat([state, action], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class DDPG:
    """Deep Deterministic Policy Gradients (DDPG) agent.

    Args:
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Dimensionality of the action space.
        max_action (float): Upper bound for the output actions.

    Attributes:
        actor (Actor): Actor network.
        actor_target (Actor): Target actor network.
        actor_optimizer (torch.optim): Actor optimizer.
        critic (Critic): Critic network.
        critic_target (Critic): Target critic network.
        critic_optimizer (torch.optim): Critic optimizer.
        replay_buffer (deque): Replay buffer for storing experiences.
        gamma (float): Discount factor.
        tau (float): Soft update factor.
    """
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.002)

        self.replay_buffer = deque(maxlen=1000000)
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state):
        """Select an action using the actor network.

        Args:
            state (numpy.ndarray): Input state.

        Returns:
            numpy.ndarray: Selected action.
        """
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=64):
        """Train the actor and critic networks.

        Args:
            batch_size (int, optional): Batch size for training. Defaults to 64.
        """
        if len(self.replay_buffer) < batch_size:
            return

        # Sample a random batch from the replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*batch)

        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        done_batch = torch.FloatTensor(done_batch)

        # Update Critic
        target_actions = self.actor_target(next_state_batch)
        target_values = self.critic_target(next_state_batch, target_actions.detach())
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * target_values
        predicted_values = self.critic(state_batch, action_batch)

        critic_loss = nn.MSELoss()(predicted_values, expected_values.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        predicted_actions = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, predicted_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update(self.actor, self.actor_target, self.tau)
        self.soft_update(self.critic, self.critic_target, self.tau)

    def soft_update(self, source, target, tau):
        """Soft update for target networks.

        Args:
            source (nn.Module): Source network.
            target (nn.Module): Target network.
            tau (float): Soft update factor.
        """
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * source_param.data)

    def add_to_replay_buffer(self, state, action, next_state, reward, done):
        """Add an experience to the replay buffer.

        Args:
            state (numpy.ndarray): Current state.
            action (numpy.ndarray): Taken action.
            next_state (numpy.ndarray): Next state.
            reward (float): Received reward.
            done (bool): Termination flag.
        """
        self.replay_buffer.append((state, action, next_state, reward, done))

# Example usage
state_dim = 3  # Example state dimension
action_dim = 1  # Example action dimension
max_action = 2.0  # Example maximum action value

# Create DDPG agent
ddpg_agent = DDPG(state_dim, action_dim, max_action)

# Training loop
for episode in range(1000):
    state = np.random.rand(state_dim)  # Example: replace with your state input
    for _ in range(100):  # Replace with the desired episode length
        action = ddpg_agent.select_action(state)
        next_state = np.random.rand(state_dim)  # Example: replace with your next state input
        reward = np.random.rand()  # Example: replace with your reward computation
        done = False  # Replace with your termination condition
        ddpg_agent.add_to_replay_buffer(state, action, next_state, reward, done)
        ddpg_agent.train()
        state = next_state

    # Print the total reward for the episode (optional)
    print(f"Episode: {episode}, Total Reward: {np.sum(reward)}")
