import numpy as np

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_size=64, gamma=0.99, clip_param=0.2, lr=0.001):
        """
        Initializes the PPO agent.

        Parameters:
        - state_dim: Dimensionality of the state space.
        - action_dim: Dimensionality of the action space.
        - hidden_size: Size of the hidden layer in the neural network.
        - gamma: Discount factor for future rewards.
        - clip_param: PPO clipping parameter.
        - lr: Learning rate for the optimizer.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.clip_param = clip_param
        self.lr = lr

        # Initialize policy and value networks
        self.policy_net = self._build_network()
        self.value_net = self._build_network()

    def _build_network(self):
        """
        Builds a simple neural network model.

        Returns:
        - model: Dictionary containing weights and biases.
        """
        model = {
            'weights': [np.random.rand(self.state_dim, self.hidden_size),
                        np.random.rand(self.hidden_size, self.action_dim)],
            'biases': [np.zeros((1, self.hidden_size)),
                       np.zeros((1, self.action_dim))]
        }
        return model

    def get_action(self, state):
        """
        Samples an action from the policy network.

        Parameters:
        - state: Current state.

        Returns:
        - action: Sampled action.
        - action_prob: Probability of the sampled action.
        """
        logits = self._forward(self.policy_net, state)
        action_probs = self._softmax(logits)
        action = np.random.choice(self.action_dim, p=action_probs.ravel())
        return action, action_probs[:, action]

    def _forward(self, model, x):
        """
        Forward pass through the neural network.

        Parameters:
        - model: Dictionary containing weights and biases.
        - x: Input data.

        Returns:
        - logits: Unnormalized scores from the output layer.
        """
        layer = np.maximum(0, x @ model['weights'][0] + model['biases'][0])
        logits = layer @ model['weights'][1] + model['biases'][1]
        return logits

    def _softmax(self, x):
        """
        Softmax function.

        Parameters:
        - x: Input data.

        Returns:
        - softmax_probs: Probabilities after applying softmax.
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        softmax_probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return softmax_probs

    def update(self, states, actions, rewards, next_states, dones):
        """
        Updates the policy and value networks using PPO.

        Parameters:
        - states: List of states.
        - actions: List of actions.
        - rewards: List of rewards.
        - next_states: List of next states.
        - dones: List of terminal flags.

        Returns:
        - policy_loss: PPO policy loss.
        - value_loss: PPO value loss.
        """
        policy_loss, value_loss = self._ppo_update(states, actions, rewards, next_states, dones)
        return policy_loss, value_loss

    def _ppo_update(self, states, actions, rewards, next_states, dones):
        """
        Placeholder for PPO update rule. Replace with the actual PPO update logic.

        Parameters:
        - states: List of states.
        - actions: List of actions.
        - rewards: List of rewards.
        - next_states: List of next states.
        - dones: List of terminal flags.

        Returns:
        - policy_loss: PPO policy loss.
        - value_loss: PPO value loss.
        """
        policy_loss = 0.0
        value_loss = 0.0
        # Implement PPO update logic here
        # This is a placeholder and should be replaced
        return policy_loss, value_loss


# Test PPOAgent with synthetic data
if __name__ == "__main__":
    state_dim = 4
    action_dim = 2
    agent = PPOAgent(state_dim, action_dim)

    # Generate synthetic data
    num_samples = 1000
    states = np.random.rand(num_samples, state_dim)
    actions = np.random.randint(0, action_dim, size=num_samples)
    rewards = np.random.rand(num_samples)
    next_states = np.random.rand(num_samples, state_dim)
    dones = np.random.choice([False, True], size=num_samples)

    # Test PPO update
    policy_loss, value_loss = agent.update(states, actions, rewards, next_states, dones)
    print(f"PPO Policy Loss: {policy_loss}, PPO Value Loss: {value_loss}")
