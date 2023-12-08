# Deep Deterministic Policy Gradients (DDPG) Implementation

## Overview

This repository contains a Python implementation of the Deep Deterministic Policy Gradients (DDPG) algorithm. DDPG is an off-policy actor-critic algorithm that is particularly well-suited for continuous action spaces in reinforcement learning.

## Contents

- [Actor](#actor)
- [Critic](#critic)
- [DDPG Agent](#ddpg-agent)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Actor

The `Actor` class represents the policy network, which takes the environment state as input and outputs continuous actions. The actor utilizes a fully connected neural network with ReLU activation functions.

## Critic

The `Critic` class represents the value function network, which evaluates the state-action pairs. It consists of separate pathways for processing states and actions, and the results are concatenated before passing through a fully connected layer.

## DDPG Agent

The `DDPG` class brings together the actor and critic networks to create a complete DDPG agent. It includes methods for selecting actions, training the networks, and updating target networks.

### Hyperparameters

- State dimension (`state_dim`): Dimensionality of the environment state.
- Action dimension (`action_dim`): Dimensionality of the action space.
- Maximum action value (`max_action`): Maximum value for scaling the output of the actor.
- Discount factor (`discount`): Discount factor for future rewards.
- Target network update rate (`tau`): Rate at which target networks are updated.
- Actor learning rate (`actor_lr`): Learning rate for the actor network.
- Critic learning rate (`critic_lr`): Learning rate for the critic network.

## Usage

Example usage of the DDPG agent in a simple environment is provided in the [example script](example.py). To use the DDPG algorithm in your custom environment, follow these steps:

1. Define your environment and specify the state and action dimensions.
2. Create an instance of the `DDPG` class with the appropriate dimensions.
3. Implement a replay buffer to store and sample experiences.
4. Train the DDPG agent using the `train` method.

```python
# Example Usage
state_dim = 3
action_dim = 1
max_action = 1.0

# Create DDPG agent
ddpg_agent = DDPG(state_dim, action_dim, max_action)

# Dummy replay buffer (replace with your own implementation)
replay_buffer = DummyReplayBuffer(capacity=10000)

# Dummy environment (replace with your own implementation)
env = DummyEnvironment()

# Training loop
for episode in range(1000):
    state = env.reset()
    total_reward = 0

    for step in range(200):  # Assuming a maximum of 200 steps per episode
        action = ddpg_agent.select_action(state)
        next_state, reward, done = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)

        if len(replay_buffer.buffer) > 64:  # Start training after 64 samples in the buffer
            actor_loss, critic_loss = ddpg_agent.train(replay_buffer, batch_size=64)

        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
