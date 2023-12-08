import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the housing dataset from CSV file
df = pd.read_csv('housing.csv')

# Extract features and target variable
X = df[['median_income', 'median_house_value']].values

class KMeansPlusPlus:
    def __init__(self, k, max_iters=100):
        """
        Initialize the KMeansPlusPlus object.

        Parameters:
        - k (int): Number of clusters.
        - max_iters (int): Maximum number of iterations for convergence.
        """
        self.k = k
        self.max_iters = max_iters

    def initialize_centroids(self, X):
        """
        Initialize cluster centroids using K-Means++.

        Parameters:
        - X (numpy.ndarray): Input data.

        Returns:
        - centroids (numpy.ndarray): Initial cluster centroids.
        """
        centroids = [X[np.random.choice(len(X))]]

        for _ in range(1, self.k):
            distances = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in X])
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            rand = np.random.rand()
            new_centroid_index = np.searchsorted(cumulative_probabilities, rand)
            centroids.append(X[new_centroid_index])

        return np.array(centroids)

    def assign_clusters(self, X, centroids):
        """
        Assign data points to the nearest cluster.

        Parameters:
        - X (numpy.ndarray): Input data.
        - centroids (numpy.ndarray): Current cluster centroids.

        Returns:
        - clusters (numpy.ndarray): Cluster assignments for each data point.
        """
        distances = np.array([np.linalg.norm(X - c, axis=1) for c in centroids])
        clusters = np.argmin(distances, axis=0)
        return clusters

    def update_centroids(self, X, clusters):
        """
        Update cluster centroids based on the mean of assigned data points.

        Parameters:
        - X (numpy.ndarray): Input data.
        - clusters (numpy.ndarray): Cluster assignments for each data point.

        Returns:
        - new_centroids (numpy.ndarray): Updated cluster centroids.
        """
        new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(self.k)])
        return new_centroids

    def fit(self, X):
        """
        Fit the K-Means++ model to the data.

        Parameters:
        - X (numpy.ndarray): Input data.
        """
        # Initialize centroids using K-Means++
        centroids = self.initialize_centroids(X)

        for _ in range(self.max_iters):
            # Assign data points to clusters
            clusters = self.assign_clusters(X, centroids)

            # Update centroids
            new_centroids = self.update_centroids(X, clusters)

            # Check for convergence
            if np.all(centroids == new_centroids):
                break

            centroids = new_centroids

        self.centroids = centroids
        self.clusters = clusters

# Instantiate the KMeansPlusPlus model with the desired number of clusters (k)
kmeans_plus_plus = KMeansPlusPlus(k=3)

# Fit the model to the housing data
kmeans_plus_plus.fit(X)

# Visualize the results
plt.scatter(X[:, 0], X[:, 1], c=kmeans_plus_plus.clusters, cmap='viridis', alpha=0.5)
plt.scatter(kmeans_plus_plus.centroids[:, 0], kmeans_plus_plus.centroids[:, 1], marker='X', s=200, color='red', label='Centroids')
plt.title('K-Means++ Clustering on Housing Data')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.legend()
plt.show()
