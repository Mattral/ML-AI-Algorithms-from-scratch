import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        """
        Initialize the Q-learning agent.

        Parameters:
        - n_states (int): Number of states in the environment.
        - n_actions (int): Number of actions the agent can take.
        - learning_rate (float): The learning rate for updating Q-values.
        - discount_factor (float): Discount factor for future rewards.
        - exploration_prob (float): Probability of exploration during action selection.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        """
        Choose an action for the current state based on the exploration-exploitation trade-off.

        Parameters:
        - state (int): Current state.

        Returns:
        - action (int): Chosen action.
        """
        # Exploration-exploitation trade-off
        if np.random.uniform(0, 1) < self.exploration_prob:
            return np.random.choice(self.n_actions)  # Explore
        else:
            return np.argmax(self.q_table[state, :])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-value for a state-action pair using the Q-learning formula.

        Parameters:
        - state (int): Current state.
        - action (int): Chosen action.
        - reward (float): Received reward.
        - next_state (int): Next state.
        """
        # Q-value update using the Q-learning formula
        current_q_value = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state, :])
        new_q_value = (1 - self.learning_rate) * current_q_value + \
                      self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state, action] = new_q_value

    def train(self, episodes, max_steps_per_episode, initial_state=0):
        """
        Train the Q-learning agent in the environment.

        Parameters:
        - episodes (int): Number of episodes to train.
        - max_steps_per_episode (int): Maximum number of steps per episode.
        - initial_state (int): Initial state for each episode.
        """
        for episode in range(episodes):
            state = initial_state
            total_reward = 0

            for step in range(max_steps_per_episode):
                action = self.choose_action(state)
                # For simplicity, let's assume a reward of +1 for reaching the goal (terminal state)
                reward = 1 if action == self.n_actions - 1 else 0
                next_state = state + 1 if action < self.n_actions - 1 else state  # Move to the next state
                total_reward += reward

                self.update_q_table(state, action, reward, next_state)

                state = next_state

                if state == self.n_states - 1:  # If terminal state reached
                    break

            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

# Example usage:
# Assume a simple environment with 5 states and 2 actions
n_states = 5
n_actions = 2

# Create a Q-learning agent
q_learning_agent = QLearning(n_states, n_actions)

# Train the agent for 100 episodes with a maximum of 10 steps per episode
q_learning_agent.train(episodes=100, max_steps_per_episode=10)
