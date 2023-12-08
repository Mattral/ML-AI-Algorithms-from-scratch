import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances

class DBSCAN:
    def __init__(self, eps, min_samples):
        """
        DBSCAN (Density-Based Spatial Clustering of Applications with Noise) Algorithm.

        Parameters:
        - eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        - min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

        Attributes:
        - labels (numpy.ndarray): Cluster labels assigned to each data point. -1 represents noise.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def _eps_neighborhood(self, point, data):
        """
        Find the indices of data points within the epsilon neighborhood of a given point.

        Parameters:
        - point (numpy.ndarray): The target data point.
        - data (numpy.ndarray): The dataset.

        Returns:
        - numpy.ndarray: Indices of data points within the epsilon neighborhood.
        """
        distances = euclidean_distances([point], data)[0]
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, point, neighbors, cluster_id, data):
        """
        Expand a cluster starting from a core point.

        Parameters:
        - point (int): Index of the core point.
        - neighbors (numpy.ndarray): Indices of points within the epsilon neighborhood of the core point.
        - cluster_id (int): Cluster label assigned to the current cluster.
        - data (numpy.ndarray): The dataset.
        """
        self.labels[point] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if self.labels[neighbor] == -1:
                self.labels[neighbor] = cluster_id
                new_neighbors = self._eps_neighborhood(data[neighbor], data)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate([neighbors, new_neighbors])
            elif self.labels[neighbor] == 0:
                self.labels[neighbor] = cluster_id
            i += 1

    def fit_predict(self, data):
        """
        Fit the DBSCAN model to the dataset and predict cluster labels.

        Parameters:
        - data (numpy.ndarray): The input dataset.

        Returns:
        - numpy.ndarray: Cluster labels assigned to each data point.
        """
        n = data.shape[0]
        self.labels = np.zeros(n, dtype=int)

        cluster_id = 0
        for i in range(n):
            if self.labels[i] != 0:
                continue

            neighbors = self._eps_neighborhood(data[i], data)
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1
            else:
                cluster_id += 1
                self._expand_cluster(i, neighbors, cluster_id, data)

        return self.labels

# Example usage with visualization
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels = dbscan.fit_predict(X)

# Visualization
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolors='k')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
