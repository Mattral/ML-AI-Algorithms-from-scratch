import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def kmeans(X, k, max_iters=100, tol=1e-4):
    """
    K-means clustering algorithm.

    Parameters:
    - X: numpy array, shape (n_samples, n_features)
        Input data.
    - k: int
        Number of clusters.
    - max_iters: int, optional (default=100)
        Maximum number of iterations.
    - tol: float, optional (default=1e-4)
        Tolerance to declare convergence.

    Returns:
    - centroids: numpy array, shape (k, n_features)
        Final cluster centers.
    - labels: numpy array, shape (n_samples,)
        Index of the cluster each sample belongs to.
    """

    # Get the number of samples and features from the input data
    n_samples, n_features = X.shape

    # Initialize centroids randomly by choosing k data points from the input
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    # Iterative optimization loop
    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        # np.linalg.norm computes the Euclidean distance between each data point and each centroid
        # np.argmin finds the index of the minimum distance, indicating the closest centroid
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

        # Update centroids by computing the mean of the data points assigned to each cluster
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence by comparing the change in centroids
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        # Update centroids for the next iteration
        centroids = new_centroids

    # Return the final centroids and cluster assignments
    return centroids, labels

# Load your dataset from a CSV file
# Assuming the first two columns are the features of interest
our_data = pd.read_csv('housing.csv').values # replace housing with your own dataset

# Specify the number of clusters (you need to choose this based on your data)
k = 2

# Apply K-means algorithm to your own dataset
centroids, labels = kmeans(our_data, k)

# Print the final centroids and cluster assignments
print("Final centroids:")
print(centroids)
print("Cluster assignments:")
print(labels)


# Plot the data points with color-coded clusters
plt.scatter(our_data[:, 0], our_data[:, 1], c=labels, cmap='viridis', marker='o', edgecolors='k', label='Data Points')

# Plot the final centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')

plt.title('K-means Clustering')
plt.xlabel('Income')
plt.ylabel('House value')
plt.legend()
plt.show()


"""
link to dataset I used here
https://www.kaggle.com/datasets/camnugent/california-housing-prices/
"""
