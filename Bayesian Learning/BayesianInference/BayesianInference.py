import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, bernoulli

class BayesianInference:
    def __init__(self, prior_alpha, prior_beta):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.posterior_alpha = prior_alpha
        self.posterior_beta = prior_beta

    def update_posterior(self, data):
        self.posterior_alpha += np.sum(data)
        self.posterior_beta += len(data) - np.sum(data)

    def plot_distributions(self, true_prob_success=None):
        x = np.linspace(0, 1, 1000)
        prior_distribution = beta.pdf(x, self.prior_alpha, self.prior_beta)
        posterior_distribution = beta.pdf(x, self.posterior_alpha, self.posterior_beta)

        plt.figure(figsize=(10, 6))
        plt.plot(x, prior_distribution, label='Prior Distribution', linestyle='--')
        plt.plot(x, posterior_distribution, label='Posterior Distribution')
        
        if true_prob_success is not None:
            plt.axvline(x=true_prob_success, color='red', linestyle='--', label='True Probability of Success')
        
        plt.title('Bayesian Inference: Updating Prior to Posterior')
        plt.xlabel('Probability of Success')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.show()

# True probability of success (unknown in practice)
true_prob_success = 0.7

# Generate synthetic data (observations)
np.random.seed(42)
data = bernoulli.rvs(true_prob_success, size=20)

# Create Bayesian Inference object with a Beta prior
bayesian_inference = BayesianInference(prior_alpha=1, prior_beta=1)

# Update the posterior based on observed data
bayesian_inference.update_posterior(data)

# Plot the prior, posterior, and true probability of success
bayesian_inference.plot_distributions(true_prob_success)
