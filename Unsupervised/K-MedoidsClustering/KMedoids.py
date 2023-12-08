import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KMedoids:
    """
    K-Medoids Clustering Algorithm.

    Parameters:
    - k (int): Number of clusters.
    - max_iterations (int): Maximum number of iterations for convergence.

    Methods:
    - fit(X): Fit the K-Medoids model to the input data.
    """

    def __init__(self, k=3, max_iterations=100):
        """
        Initialize the K-Medoids clustering algorithm.

        Parameters:
        - k (int): Number of clusters.
        - max_iterations (int): Maximum number of iterations for convergence.
        """
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        """
        Fit the K-Medoids model to the input data.

        Parameters:
        - X (numpy.ndarray): Input data with shape (n_samples, n_features).
        """
        self.X = X
        self.medoids = self._initialize_medoids()
        self.clusters = np.zeros(len(X))

        for _ in range(self.max_iterations):
            # Assign each data point to the nearest medoid
            self._assign_clusters()

            # Update medoids
            new_medoids = self._update_medoids()

            # Check for convergence
            if np.array_equal(self.medoids, new_medoids):
                break

            self.medoids = new_medoids

    def _initialize_medoids(self):
        """
        Initialize medoids by randomly selecting k data points.

        Returns:
        - medoids (numpy.ndarray): Initial medoids.
        """
        return self.X[np.random.choice(len(self.X), self.k, replace=False)]

    def _assign_clusters(self):
        """
        Assign each data point to the nearest medoid.
        Updates the 'clusters' attribute.
        """
        for i, x in enumerate(self.X):
            # Find the index of the nearest medoid
            medoid_indices = np.argmin(np.linalg.norm(self.medoids - x, axis=1))
            self.clusters[i] = medoid_indices

    def _update_medoids(self):
        """
        Update medoids based on the data points assigned to each cluster.

        Returns:
        - new_medoids (numpy.ndarray): Updated medoids.
        """
        new_medoids = np.copy(self.medoids)
        for i in range(self.k):
            # Indices of data points assigned to the current medoid
            cluster_indices = np.where(self.clusters == i)[0]

            # Calculate the total cost for each data point in the cluster
            total_costs = np.sum(np.linalg.norm(self.X[cluster_indices] - self.X[cluster_indices][:, np.newaxis], axis=2), axis=0)

            # Find the index of the data point with the lowest total cost (new medoid)
            new_medoid_index = cluster_indices[np.argmin(total_costs)]

            # Update the medoid
            new_medoids[i] = self.X[new_medoid_index]

        return new_medoids

# Load the housing dataset from CSV file
df = pd.read_csv('housing.csv')

# Extract features
X = df[['median_income', 'median_house_value']].values

# Instantiate and fit the KMedoids model
kmedoids = KMedoids(k=3)
kmedoids.fit(X)

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=kmedoids.clusters, cmap='viridis', alpha=0.5)
plt.scatter(kmedoids.medoids[:, 0], kmedoids.medoids[:, 1], c='red', marker='x', s=200)
plt.title('K-Medoids Clustering')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.show()
