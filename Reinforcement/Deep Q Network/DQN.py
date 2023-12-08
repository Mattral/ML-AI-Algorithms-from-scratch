import numpy as np
from collections import deque
import random

class SimpleDQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon_decay=0.995, epsilon_min=0.01,
                 replay_memory_size=10000, batch_size=32, target_update_frequency=100):
        """
        Initialize the Simple DQN agent.

        Parameters:
        - state_size (int): Dimensionality of the state space.
        - action_size (int): Number of possible actions.
        - learning_rate (float): Learning rate for the neural network.
        - gamma (float): Discount factor for future rewards.
        - epsilon_decay (float): Decay rate for exploration-exploitation trade-off.
        - epsilon_min (float): Minimum exploration probability.
        - replay_memory_size (int): Size of the replay memory buffer.
        - batch_size (int): Number of samples to train the neural network on in each iteration.
        - target_update_frequency (int): Frequency of updating the target Q-network.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0  # Initial exploration probability
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Create the Q-network and target Q-network
        self.q_network = self._build_q_network()
        self.target_q_network = self._build_q_network()
        self._update_target_q_network()

        # Initialize replay memory
        self.replay_memory = deque(maxlen=replay_memory_size)

        # Hyperparameters
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.step_counter = 0

    def _build_q_network(self):
        """Build and return a simple feedforward neural network for Q-values."""
        model = {
            'weights': [np.random.rand(self.state_size, 24), np.random.rand(24, 24), np.random.rand(24, self.action_size)],
            'biases': [np.zeros((1, 24)), np.zeros((1, 24)), np.zeros((1, self.action_size))]
        }
        return model

    def _update_target_q_network(self):
        """Update the weights of the target Q-network with the current Q-network weights."""
        self.target_q_network = self.q_network.copy()

    def act(self, state):
        """
        Choose an action based on the current state using an epsilon-greedy strategy.

        Parameters:
        - state (np.ndarray): The current state.

        Returns:
        - action (int): The chosen action.
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Explore
        else:
            q_values = self._predict_q_values(state.reshape(1, -1))[0]
            return np.argmax(q_values)  # Exploit

    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience in the replay memory.

        Parameters:
        - state (np.ndarray): The current state.
        - action (int): The chosen action.
        - reward (float): The received reward.
        - next_state (np.ndarray): The next state.
        - done (bool): Whether the episode is done.
        """
        self.replay_memory.append((state, action, reward, next_state, done))

    def _train_q_network(self, states, target_q_values):
        """Train the Q-network using the provided states and target Q-values."""
        # Forward pass
        layer1 = np.maximum(0, states @ self.q_network['weights'][0] + self.q_network['biases'][0])
        layer2 = np.maximum(0, layer1 @ self.q_network['weights'][1] + self.q_network['biases'][1])
        q_values = layer2 @ self.q_network['weights'][2] + self.q_network['biases'][2]

        # Backpropagation
        loss = np.mean((q_values - target_q_values) ** 2)
        # You can add gradient descent or any optimization method here, but for simplicity, we'll skip it for now

    def _predict_q_values(self, states):
        """Predict Q-values for the given states using the current Q-network."""
        layer1 = np.maximum(0, states @ self.q_network['weights'][0] + self.q_network['biases'][0])
        layer2 = np.maximum(0, layer1 @ self.q_network['weights'][1] + self.q_network['biases'][1])
        q_values = layer2 @ self.q_network['weights'][2] + self.q_network['biases'][2]
        return q_values

    def experience_replay(self):
        """Train the Q-network using experience replay."""
        if len(self.replay_memory) < self.batch_size:
            return  # Not enough samples for training

        # Sample a random batch from the replay memory
        batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert the lists to NumPy arrays
        states = np.vstack(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.vstack(next_states)
        dones = np.array(dones)

        # Compute Q-values for current and next states
        current_q_values = self._predict_q_values(states)
        next_q_values = self._predict_q_values(next_states)

        # Update Q-values based on the Bellman equation
        target_q_values = current_q_values.copy()
        for i in range(self.batch_size):
            target_q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i]) * (1 - dones[i])

        # Train the Q-network
        self._train_q_network(states, target_q_values)

        # Decay exploration probability
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        # Update target Q-network periodically
        if self.step_counter % self.target_update_frequency == 0:
            self._update_target_q_network()

        self.step_counter += 1

# Create a grid world environment
class GridWorld:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agent_position = [0, 0]
        self.goal_position = [width - 1, height - 1]
        self.done = False

    def reset(self):
        self.agent_position = [0, 0]
        self.done = False
        return np.array(self.agent_position)

    def step(self, action):
        if action == 0:  # Move right
            self.agent_position[0] = min(self.agent_position[0] + 1, self.width - 1)
        elif action == 1:  # Move left
            self.agent_position[0] = max(self.agent_position[0] - 1, 0)
        elif action == 2:  # Move down
            self.agent_position[1] = min(self.agent_position[1] + 1, self.height - 1)
        elif action == 3:  # Move up
            self.agent_position[1] = max(self.agent_position[1] - 1, 0)

        if self.agent_position == self.goal_position:
            self.done = True
            reward = 1.0
        else:
            reward = 0.0

        return np.array(self.agent_position), reward, self.done, {}

# Test the DQN agent in the synthetic environment
env = GridWorld(width=5, height=5)
state_size = 2  # 2D state space for the grid world
action_size = 4  # Four possible actions: right, left, down, up

# Create a DQN agent
dqn_agent = SimpleDQNAgent(state_size, action_size)

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while not env.done:
        # Choose an action and take a step
        action = dqn_agent.act(state)
        next_state, reward, done, _ = env.step(action)

        # Remember the experience and perform experience replay
        dqn_agent.remember(state, action, reward, next_state, done)
        dqn_agent.experience_replay()

        state = next_state
        total_reward += reward

    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# After training, you can test the agent by running a similar loop without exploration
test_episodes = 10
for episode in range(test_episodes):
    state = env.reset()
    total_reward = 0

    while not env.done:
        action = dqn_agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

    print(f"Test Episode {episode}, Total Reward: {total_reward}")
