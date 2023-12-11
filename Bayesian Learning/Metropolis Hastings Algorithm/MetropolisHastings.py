import numpy as np
import matplotlib.pyplot as plt

class MetropolisHastings:
    def __init__(self, target_distribution, proposal_distribution, initial_state):
        """
        Initialize the Metropolis-Hastings sampler.

        Parameters:
        - target_distribution (callable): The target distribution function.
        - proposal_distribution (callable): The proposal distribution function.
        - initial_state (float): The initial state of the Markov chain.
        """
        self.target_distribution = target_distribution
        self.proposal_distribution = proposal_distribution
        self.current_state = initial_state
        self.samples = [initial_state]

    def sample(self, num_samples):
        """
        Perform Metropolis-Hastings sampling.

        Parameters:
        - num_samples (int): Number of samples to generate.

        Returns:
        - np.ndarray: Array of generated samples.
        """
        for _ in range(num_samples):
            # Propose a new state
            proposed_state = self.proposal_distribution(self.current_state)

            # Calculate acceptance ratio
            acceptance_ratio = min(1, self.target_distribution(proposed_state) / self.target_distribution(self.current_state))

            # Accept or reject the proposed state
            if np.random.rand() < acceptance_ratio:
                self.current_state = proposed_state

            self.samples.append(self.current_state)

        return np.array(self.samples)

def target_distribution(x):
    # Example: Univariate Gaussian distribution
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def proposal_distribution(current_state):
    # Example: Normal distribution as the proposal distribution
    return current_state + np.random.normal(scale=0.5)

# Initial state
initial_state = 0.0

# Number of samples
num_samples = 5000

# Create Metropolis-Hastings sampler
mh_sampler = MetropolisHastings(target_distribution, proposal_distribution, initial_state)

# Generate samples
samples = mh_sampler.sample(num_samples)

# Plot results
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, label="Metropolis-Hastings Samples")
x_range = np.linspace(-3, 3, 1000)
plt.plot(x_range, target_distribution(x_range), label="Target Distribution", linewidth=2, color='red')
plt.title("Metropolis-Hastings Sampling")
plt.xlabel("Sampled Values")
plt.ylabel("Density")
plt.legend()
plt.show()
