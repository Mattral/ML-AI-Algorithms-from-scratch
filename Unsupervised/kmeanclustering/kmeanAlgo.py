"""
Basic Info:

Input Data (X): The input data should be a 2D numpy array where each row
                corresponds to a data point and each column corresponds to a feature.

Number of Clusters (k): This parameter determines the number of clusters
                        the algorithm will try to identify in the data.

Maximum Iterations (max_iters): This parameter limits the number of iterations
                                the algorithm will perform to update the centroids.
                                If convergence is not reached within this limit, the algorithm stops.

Tolerance (tol): This parameter sets the convergence threshold.
                If the change in centroids between consecutive iterations is below this threshold,
                the algorithm considers that it has converged and stops.

Output: The function returns the final centroids (cluster centers) and the labels,
        indicating which cluster each data point belongs to.

Experiment with different values of k, max_iters, and tol to observe how they affect the clustering results.

"""

#ReadMe
    
"""

Suggestion about Adjusting X (Input Data):

    The K-means algorithm is sensitive to the scale and distribution of features.
    If the features have different scales,
    the algorithm might give more weight to features with larger magnitudes.
    It's often a good practice to standardize or normalize the data before applying K-means.

    The distribution of data also plays a role. K-means assumes that clusters are spherical and equally sized.
    If your data violates these assumptions (e.g., clusters have different shapes or densities),
    K-means might not perform well.

    Outliers can also impact K-means. Since the algorithm minimizes the sum of squared distances,
    outliers can disproportionately influence the cluster centers.
    Preprocessing steps like outlier removal may be necessary.
"""
    

import numpy as np
import pandas as pd
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


# Example usage:
np.random.seed(42)
"""
    ine sets the seed for the random number generator in NumPy.
    This ensures that the random initialization of centroids is reproducible.
    If you keep the seed constant (e.g., np.random.seed(42)),
    the initial centroids will be the same every time you run the algorithm.
    This can be useful for reproducibility when you want consistent results for debugging
    or comparison purposes.
    If you change the seed value, the initial centroids will be different.
    However,
    the algorithm's convergence and final clusters should still be consistent
    given the same data and parameters.
"""



# Generate some random data with two clusters
#( You can replace this line with your own dataset. )
data = np.concatenate([np.random.randn(100, 2), np.random.randn(100, 2) + [5, 5]])

"""
     In above line, We are creating a synthetic dataset for the purpose of testing the K-means algorithm.
     The dataset is created by concatenating two sets of points:

    np.random.randn(100, 2): This generates 100 data points with 2 features each.
    The data is drawn from a standard normal distribution (mean=0, variance=1). This simulates one cluster.

    np.random.randn(100, 2) + [5, 5]: This generates another 100 data points with 2 features each,
    but it adds [5, 5] to each point. This simulates a second cluster that is shifted from the first one.

"""
#if you have a dataset in a different format,
#make sure it's a 2D NumPy array where each row represents a data point and each column represents a feature.




# Apply K-means algorithm

k = 2

centroids, labels = kmeans(data, k)
"""
    Adjusting k (Number of Clusters):

    The k parameter determines the number of clusters the algorithm should identify.
    If you set k too low, the algorithm may fail to capture the underlying structure of the data.
    Conversely, if you set k too high, the algorithm may create artificial clusters.

    A common approach to determining the optimal k is to use techniques
    like the elbow method or silhouette analysis.
    These methods involve running the algorithm for different values of k
    and evaluating the clustering performance.
"""

   



# Print the final centroids and cluster assignments
print("Final centroids:")
print(centroids)
print("Cluster assignments:")
print(labels)

#_________________________________________________________________#
#__________________________ Plotting _____________________________#
#_________________________________________________________________#

# Plot the data points with color-coded clusters
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolors='k', label='Data Points')

# Plot the final centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')

plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()




####################################################################################
##________________________________ Working with your own dataset _________________##
####################################################################################


"""
 if you have a CSV file containing your dataset,
 you might use a library like pandas to load it into a NumPy array
 
```
    your_data = pd.read_csv('your_dataset.csv').values
```

Then, you can use your_data in place of the data variable in the K-means algorithm.

```
    k = 2
    centroids, labels = kmeans(your_data, k)
```
 Make sure it's a 2D NumPy array
 where each row represents a data point and
 each column represents a feature.

"""

    
