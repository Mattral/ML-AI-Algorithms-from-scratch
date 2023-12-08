# Simple DQN Agent

## What is it?

The Simple DQN (Deep Q Network) Agent is a reinforcement learning algorithm implemented in Python. It utilizes a Q-learning approach to train an agent to navigate a grid world environment.

## How it works

The agent employs a deep neural network to approximate the Q-values for different state-action pairs. It uses an epsilon-greedy strategy to balance exploration and exploitation during training. The training involves experience replay, where past experiences are stored and randomly sampled for training the neural network.

## Math

The core mathematical concept behind the Simple DQN Agent is the Q-learning algorithm. The Q-values are updated using the Bellman equation, which models the expected future rewards for each action in a given state. The neural network is trained to minimize the difference between predicted Q-values and target Q-values computed using the Bellman equation.

### Q-learning

Q-learning is a model-free reinforcement learning algorithm that learns a policy, which tells an agent what action to take under what circumstances. In the context of the Simple DQN Agent:

- **State (S):** Represents the current configuration of the environment.
- **Action (A):** Denotes the possible actions the agent can take.
- **Reward (R):** Indicates the immediate reward received after taking an action in a certain state.
- **Q-Value (Q):** Represents the expected cumulative future reward for taking an action in a given state.

The Q-value for a state-action pair is updated iteratively using the Bellman equation:

$$\[ Q(S, A) \leftarrow (1 - \alpha) \cdot Q(S, A) + \alpha \cdot (R + \gamma \cdot \max_{a'} Q(S', a')) \]$$

where:
- \(\alpha\) is the learning rate, controlling the weight given to new information.
- \(\gamma\) is the discount factor, determining the importance of future rewards.

### Deep Q Network (DQN)

The Simple DQN Agent employs a neural network to approximate Q-values, allowing it to handle high-dimensional state spaces. The Q-network is trained to minimize the mean squared difference between predicted Q-values $(\(Q_{\text{predicted}}\))$ and target Q-values $(\(Q_{\text{target}}\))$:

$$\[ \text{Loss} = \frac{1}{N} \sum_{i=1}^{N} (Q_{\text{predicted}}(S_i, A_i) - Q_{\text{target}}(S_i, A_i))^2 \]$$

where:
- \(N\) is the batch size.
- \(S_i\) is the \(i\)-th state in the batch.
- \(A_i\) is the corresponding action taken.

### Experience Replay

To improve stability and efficiency, the Simple DQN Agent uses experience replay. It stores past experiences (state, action, reward, next state) in a replay memory and randomly samples mini-batches during training. This breaks the temporal correlation in the data and helps the agent learn more robust policies.

### Exploration-Exploitation Strategy

The agent balances exploration (trying new actions) and exploitation (choosing actions with the highest expected reward) using an epsilon-greedy strategy. The exploration probability (\(\epsilon\)) decays over time, encouraging the agent to exploit its learned policies as training progresses.

These mathematical concepts collectively enable the Simple DQN Agent to learn and improve its decision-making capabilities over time in a given environment.

## Uses

The Simple DQN Agent can be used for various applications, including:

- Autonomous navigation of agents in grid-like environments.
- Learning optimal strategies for sequential decision-making tasks.
- Training agents in scenarios where a reward signal guides the learning process.

## Pros and Cons

### Pros

- Capable of learning complex decision-making policies.
- Can handle high-dimensional state spaces.
- Suitable for environments with discrete action spaces.

### Cons

- Prone to overestimation bias.
- Requires careful tuning of hyperparameters.
- Training may be computationally intensive for large-scale environments.

