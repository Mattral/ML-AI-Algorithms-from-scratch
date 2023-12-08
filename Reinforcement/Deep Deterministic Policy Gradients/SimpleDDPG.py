import numpy as np


# Actor class
class Actor:
    def __init__(self, state_dim, action_dim, max_action):
        self.weights = np.random.rand(state_dim, action_dim)
        self.biases = np.zeros((1, action_dim))
        self.max_action = max_action

    def forward(self, state):
        x = np.maximum(0, state @ self.weights + self.biases)
        return self.max_action * x

# Critic class
class Critic:
    def __init__(self, state_dim, action_dim):
        self.state_weights = np.random.rand(state_dim, 400)
        self.state_biases = np.zeros((1, 400))
        self.action_weights = np.random.rand(1, 1)
 
        self.action_biases = np.zeros((1, 300))
        self.fc_weights = np.random.rand(700, 1)
        self.fc_biases = np.zeros((1, 1))

    def forward(self, state, action):
        state_val = np.maximum(0, state @ self.state_weights + self.state_biases)
        action_val = np.maximum(0, action @ self.action_weights + self.action_biases)


        # Check the dimensions before concatenation
        print("state_val shape:", state_val.shape)
        print("action_val shape:", action_val.shape)

        x = np.concatenate([state_val, action_val], axis=1)

        # Check the dimensions after concatenation
        print("concatenated x shape:", x.shape)

        # Check the dimensions of fc_weights and fc_biases
        print("fc_weights shape:", self.fc_weights.shape)
        print("fc_biases shape:", self.fc_biases.shape)

        # Print the shapes and values before matrix multiplication
        print("action shape:", action.shape)
        print("action_weights.T shape:", self.action_weights.T.shape)
        print("action:", action)
        print("action_weights.T:", self.action_weights.T)

        x = np.maximum(0, x @ self.fc_weights + self.fc_biases)

        # Check the dimensions after matrix multiplication
        print("output x shape:", x.shape)

        return x


# DDPG class
class DDPG:
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001, actor_lr=0.001, critic_lr=0.002):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.target_actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)

        # Initialize target networks with the same weights as the online networks
        self.target_actor.weights = self.actor.weights.copy()
        self.target_actor.biases = self.actor.biases.copy()
        self.target_critic.state_weights = self.critic.state_weights.copy()
        self.target_critic.state_biases = self.critic.state_biases.copy()
        self.target_critic.action_weights = self.critic.action_weights.copy()
        self.target_critic.action_biases = self.critic.action_biases.copy()
        self.target_critic.fc_weights = self.critic.fc_weights.copy()
        self.target_critic.fc_biases = self.critic.fc_biases.copy()

        self.discount = discount
        self.tau = tau

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

    def select_action(self, state):
        state = np.reshape(state, [1, -1])
        action = self.actor.forward(state)
        return action.squeeze()

    def train(self, replay_buffer, batch_size):
        # Sample a minibatch from the replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)

        # Convert to NumPy arrays
        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        done_batch = np.array(done_batch)

        # Update critic
        target_actions = self.target_actor.forward(next_state_batch)
        target_q_values = self.target_critic.forward(next_state_batch, target_actions)
        y = reward_batch + self.discount * (1.0 - done_batch) * target_q_values
        critic_loss = np.mean((y - self.critic.forward(state_batch, action_batch)) ** 2)

        # Update actor
        actions = self.actor.forward(state_batch)
        actor_loss = -np.mean(self.critic.forward(state_batch, actions))

        # Update target networks
        self._update_target_networks()

        # Update weights
        self.actor.weights -= self.actor_lr * np.dot(state_batch.T, np.dot((y - self.critic.forward(state_batch, actions)) * actions, self.actor.weights.T))
        self.actor.biases -= self.actor_lr * np.sum((y - self.critic.forward(state_batch, actions)) * actions, axis=0, keepdims=True)
        self.critic.state_weights -= self.critic_lr * np.dot(state_batch.T, np.dot((y - self.critic.forward(state_batch, actions)) * actions, self.critic.state_weights.T))
        self.critic.state_biases -= self.critic_lr * np.sum((y - self.critic.forward(state_batch, actions)) * actions, axis=0, keepdims=True)
        self.critic.action_weights -= self.critic_lr * np.dot(actions.T, np.dot((y - self.critic.forward(state_batch, actions)).T, self.critic.action_weights))
        self.critic.action_biases -= self.critic_lr * np.sum((y - self.critic.forward(state_batch, actions)) * actions, axis=0, keepdims=True)
        self.critic.fc_weights -= self.critic_lr * np.dot(np.concatenate([state_batch, actions], axis=1).T, np.dot((y - self.critic.forward(state_batch, actions)).T, self.critic.fc_weights))
        self.critic.fc_biases -= self.critic_lr * np.sum((y - self.critic.forward(state_batch, actions)) * actions, axis=0, keepdims=True)

        return actor_loss, critic_loss

    def _update_target_networks(self):
        self.target_actor.weights = self.tau * self.actor.weights + (1 - self.tau) * self.target_actor.weights
        self.target_actor.biases = self.tau * self.actor.biases + (1 - self.tau) * self.target_actor.biases
        self.target_critic.state_weights = self.tau * self.critic.state_weights + (1 - self.tau) * self.target_critic.state_weights
        self.target_critic.state_biases = self.tau * self.critic.state_biases + (1 - self.tau) * self.target_critic.state_biases
        self.target_critic.action_weights = self.tau * self.critic.action_weights + (1 - self.tau) * self.target_critic.action_weights
        self.target_critic.action_biases = self.tau * self.critic.action_biases + (1 - self.tau) * self.target_critic.action_biases
        self.target_critic.fc_weights = self.tau * self.critic.fc_weights + (1 - self.tau) * self.target_critic.fc_weights
        self.target_critic.fc_biases = self.tau * self.critic.fc_biases + (1 - self.tau) * self.target_critic.fc_biases

# Example Usage:
state_dim = 3
action_dim = 1
max_action = 1.0

# Create DDPG agent
ddpg_agent = DDPG(state_dim, action_dim, max_action, actor_lr=0.001, critic_lr=0.002)

# Dummy replay buffer (replace with your own implementation)
class DummyReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return zip(*batch)

# Dummy environment (replace with your own implementation)
class DummyEnvironment:
    def __init__(self):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def reset(self):
        return np.zeros(self.state_dim)

    def step(self, action):
        next_state = np.zeros(self.state_dim)
        reward = np.random.rand()
        done = False
        return next_state, reward, done

# Create DDPG agent and replay buffer
ddpg_agent = DDPG(state_dim, action_dim, max_action)
replay_buffer = DummyReplayBuffer(capacity=10000)
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
