'''
This example generates a dataset with two components,
applies the EM algorithm to fit a Gaussian Mixture Model,
and visualizes the progress at each iteration. The final means,
standard deviations, and weights are printed.
Note that this is a simplified example
'''

import numpy as np
import matplotlib.pyplot as plt

def normal_distribution(x, mean, std):
    """
    Calculate the probability density function of a normal distribution.
    """
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def expectation(data, means, stds, weights):
    """
    Expectation step of the EM algorithm.
    """
    likelihoods = np.array([weights[i] * normal_distribution(data, means[i], stds[i]) for i in range(len(means))])
    total_likelihood = np.sum(likelihoods, axis=0)
    responsibilities = likelihoods / total_likelihood
    return responsibilities

def maximization(data, responsibilities):
    """
    Maximization step of the EM algorithm.
    """
    total_responsibility = np.sum(responsibilities, axis=1)
    means = np.sum(responsibilities * data, axis=1) / total_responsibility
    stds = np.sqrt(np.sum(responsibilities * (data - means[:, np.newaxis]) ** 2, axis=1) / total_responsibility)
    weights = total_responsibility / len(data)
    return means, stds, weights

def em_algorithm(data, num_components, num_iterations):
    """
    Perform the Expectation-Maximization algorithm for Gaussian Mixture Model.

    Parameters:
    - data (numpy array): Input data.
    - num_components (int): Number of Gaussian components.
    - num_iterations (int): Number of iterations.

    Returns:
    - means (numpy array): Final means of the Gaussian components.
    - stds (numpy array): Final standard deviations of the Gaussian components.
    - weights (numpy array): Final weights of the Gaussian components.
    """
    # Initialization
    means = np.random.rand(num_components) * np.max(data)
    stds = np.random.rand(num_components) * np.std(data)
    weights = np.ones(num_components) / num_components

    for iteration in range(num_iterations):
        # Expectation step
        responsibilities = expectation(data, means, stds, weights)

        # Maximization step
        means, stds, weights = maximization(data, responsibilities)

        # Visualization
        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=30, density=True, alpha=0.6, color='g')

        x = np.linspace(np.min(data), np.max(data), 1000)
        for i in range(num_components):
            plt.plot(x, normal_distribution(x, means[i], stds[i]), label=f'Component {i + 1}')

        plt.title(f'EM Algorithm - Iteration {iteration + 1}')
        plt.legend()
        plt.show()

    return means, stds, weights

# Example usage
np.random.seed(42)
data = np.concatenate([np.random.normal(0, 1, 300), np.random.normal(5, 1.5, 200)])

num_components = 2
num_iterations = 5

means, stds, weights = em_algorithm(data, num_components, num_iterations)
print("Final Means:", means)
print("Final Standard Deviations:", stds)
print("Final Weights:", weights)
