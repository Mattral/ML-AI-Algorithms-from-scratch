import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    def __init__(self, n_components, max_iters=100, tol=1e-4):
        """
        Initialize the Gaussian Mixture Model.

        Parameters:
        - n_components: Number of components (clusters) in the mixture model.
        - max_iters: Maximum number of iterations for the EM algorithm.
        - tol: Convergence tolerance for the EM algorithm.
        """
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol

    def initialize_parameters(self, data):
        """
        Initialize model parameters: means, covariances, and weights.

        Parameters:
        - data: Input data.

        Initializes means randomly, covariances as identity matrices, and weights uniformly.
        """
        n_samples, n_features = data.shape
        np.random.seed(42)

        # Initialize means randomly from the data
        self.means = data[np.random.choice(n_samples, self.n_components, replace=False)]

        # Initialize covariances as identity matrices
        self.covariances = [np.identity(n_features) for _ in range(self.n_components)]

        # Initialize weights uniformly
        self.weights = np.ones(self.n_components) / self.n_components

    def compute_probabilities(self, data):
        """
        Compute the probabilities of data points belonging to each component.

        Parameters:
        - data: Input data.

        Returns:
        - probabilities: 2D array of probabilities.
        """
        probabilities = np.zeros((len(data), self.n_components))

        for i in range(self.n_components):
            probabilities[:, i] = multivariate_normal.pdf(data, mean=self.means[i], cov=self.covariances[i])

        return probabilities

    def expectation_step(self, data):
        """
        Perform the Expectation step of the EM algorithm.

        Parameters:
        - data: Input data.

        Returns:
        - responsibilities: 2D array of responsibilities.
        """
        probabilities = self.compute_probabilities(data)

        # Normalize probabilities to get responsibilities
        weighted_probabilities = probabilities * self.weights
        responsibilities = weighted_probabilities / np.sum(weighted_probabilities, axis=1, keepdims=True)

        return responsibilities

    def maximization_step(self, data, responsibilities):
        """
        Perform the Maximization step of the EM algorithm.

        Parameters:
        - data: Input data.
        - responsibilities: Responsibilities from the Expectation step.
        """
        # Update means, covariances, and weights
        Nk = np.sum(responsibilities, axis=0)

        for i in range(self.n_components):
            # Update means
            self.means[i] = np.sum(responsibilities[:, i][:, np.newaxis] * data, axis=0) / Nk[i]

            # Update covariances
            diff = data - self.means[i]
            self.covariances[i] = np.dot((responsibilities[:, i][:, np.newaxis] * diff).T, diff) / Nk[i]

            # Update weights
            self.weights[i] = Nk[i] / len(data)

    def fit(self, data):
        """
        Fit the Gaussian Mixture Model to the input data.

        Parameters:
        - data: Input data.
        """
        self.initialize_parameters(data)

        for iteration in range(self.max_iters):
            old_means = np.copy(self.means)

            # Expectation step
            responsibilities = self.expectation_step(data)

            # Maximization step
            self.maximization_step(data, responsibilities)

            # Check for convergence
            if np.linalg.norm(self.means - old_means) < self.tol:
                print(f"Converged after {iteration + 1} iterations.")
                break

            # Print iteration information
            print(f"Iteration {iteration + 1}/{self.max_iters}")

    def predict(self, data):
        """
        Predict cluster labels for input data.

        Parameters:
        - data: Input data.

        Returns:
        - predictions: Array of cluster labels.
        """
        responsibilities = self.expectation_step(data)
        return np.argmax(responsibilities, axis=1)

    def plot_clusters(self, data):
        """
        Plot the clustered data.

        Parameters:
        - data: Input data.
        """
        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        predictions = self.predict(data)

        for i in range(self.n_components):
            cluster_points = data[predictions == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i + 1}')

        plt.scatter(self.means[:, 0], self.means[:, 1], c='black', marker='x', s=100, label='Cluster Centers')
        plt.legend()
        plt.title('Gaussian Mixture Model Clustering')
        plt.show()

# Generate synthetic data
np.random.seed(42)
data1 = np.random.multivariate_normal(mean=[3, 3], cov=[[1, 0.5], [0.5, 1]], size=100)
data2 = np.random.multivariate_normal(mean=[7, 7], cov=[[1, -0.5], [-0.5, 1]], size=100)
data3 = np.random.multivariate_normal(mean=[10, 2], cov=[[1, 0], [0, 1]], size=100)
data = np.concatenate([data1, data2, data3])

# Instantiate and fit the Gaussian Mixture Model
gmm = GaussianMixtureModel(n_components=3)
gmm.fit(data)

# Plot the clustered data
gmm.plot_clusters(data)
