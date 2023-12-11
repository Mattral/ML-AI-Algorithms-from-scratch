import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class BayesianInference:
    def __init__(self, prior_mean, prior_std):
        """
        Initialize Bayesian Inference with a prior distribution.

        Parameters:
        - prior_mean (float): Mean of the prior distribution.
        - prior_std (float): Standard deviation of the prior distribution.
        """
        self.prior_mean = prior_mean
        self.prior_std = prior_std

    def generate_data(self, true_mean, true_std, sample_size):
        """
        Generate synthetic data based on a true distribution.

        Parameters:
        - true_mean (float): True mean of the distribution generating the data.
        - true_std (float): True standard deviation of the distribution generating the data.
        - sample_size (int): Number of data points to generate.

        Returns:
        - data (numpy.ndarray): Array of synthetic data points.
        """
        return np.random.normal(true_mean, true_std, sample_size)

    def update_posterior(self, prior_mean, prior_std, data):
        """
        Update the posterior distribution based on observed data.

        Parameters:
        - prior_mean (float): Mean of the prior distribution.
        - prior_std (float): Standard deviation of the prior distribution.
        - data (numpy.ndarray): Observed data points.

        Returns:
        - posterior_mean (float): Mean of the posterior distribution.
        - posterior_std (float): Standard deviation of the posterior distribution.
        """
        # Bayes' Theorem: posterior ∝ likelihood * prior
        likelihood_mean = np.mean(data)
        likelihood_std = np.std(data)
        
        # Calculate posterior parameters
        posterior_precision = 1 / (1 / prior_std**2 + len(data) / likelihood_std**2)
        posterior_mean = posterior_precision * (prior_mean / prior_std**2 + np.sum(data) / likelihood_std**2)
        posterior_std = np.sqrt(1 / posterior_precision)

        return posterior_mean, posterior_std

    def plot_distribution(self, true_mean, true_std, prior_mean, prior_std, posterior_mean, posterior_std, data):
        """
        Plot the true distribution, prior distribution, and posterior distribution.

        Parameters:
        - true_mean (float): True mean of the distribution generating the data.
        - true_std (float): True standard deviation of the distribution generating the data.
        - prior_mean (float): Mean of the prior distribution.
        - prior_std (float): Standard deviation of the prior distribution.
        - posterior_mean (float): Mean of the posterior distribution.
        - posterior_std (float): Standard deviation of the posterior distribution.
        - data (numpy.ndarray): Observed data points.
        """
        x = np.linspace(true_mean - 3 * true_std, true_mean + 3 * true_std, 1000)
        plt.plot(x, norm.pdf(x, true_mean, true_std), label='True Distribution', linestyle='--')

        plt.plot(x, norm.pdf(x, prior_mean, prior_std), label='Prior Distribution', linestyle='--')
        plt.scatter(data, np.zeros_like(data), color='red', label='Observed Data')

        plt.plot(x, norm.pdf(x, posterior_mean, posterior_std), label='Posterior Distribution')

        plt.title('Bayesian Inference')
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Parameters for the true distribution
    true_mean = 5.0
    true_std = 2.0

    # Parameters for the prior distribution
    prior_mean = 8.0
    prior_std = 1.5

    # Number of data points
    sample_size = 100

    # Create an instance of BayesianInference
    bayesian_inference = BayesianInference(prior_mean, prior_std)

    # Generate synthetic data
    data = bayesian_inference.generate_data(true_mean, true_std, sample_size)

    # Update posterior distribution based on observed data
    posterior_mean, posterior_std = bayesian_inference.update_posterior(prior_mean, prior_std, data)

    # Plot the true, prior, and posterior distributions
    bayesian_inference.plot_distribution(true_mean, true_std, prior_mean, prior_std, posterior_mean, posterior_std, data)

    # Print results
    print("True Mean:", true_mean)
    print("Prior Mean:", prior_mean, "Prior Standard Deviation:", prior_std)
    print("Posterior Mean:", posterior_mean, "Posterior Standard Deviation:", posterior_std)
