# K-Medoids Clustering from Scratch

## Overview

The K-Medoids clustering algorithm is a variation of K-Means that uses the medoid (the most representative point) as the center of each cluster. This README covers the implementation of K-Medoids from scratch using Python. It explains the workings of the algorithm, its use cases, mathematical equations, and provides insights into the implementation.

## How K-Medoids Clustering Works

K-Medoids clustering follows these steps:

1. **Initialization:**
   - Randomly select k data points as initial medoids.

2. **Assignment:**
   - Assign each data point to the nearest medoid.

3. **Update:**
   - Update medoids by selecting the data point within each cluster that minimizes the total dissimilarity to other points in the cluster.

4. **Convergence:**
   - Iterate the assignment and update steps until convergence (when medoids no longer change significantly).

## Mathematical Formulation

The cost function for K-Medoids is defined as the sum of dissimilarities between data points and their assigned medoids:

$$\[ J = \sum_{i=1}^{k} \sum_{j=1}^{n_i} d(x_{ij}, m_i) \]$$

where:
- $\( J \)$ is the cost function.
- $\( x_{ij} \)$ is the j-th data point in the i-th cluster.
- $\( m_i \)$ is the medoid of the i-th cluster.
- $\( d(a, b) \)$ is the dissimilarity (distance) measure between points a and b.

## Implementation from Scratch

The provided Python script (`KMedoids.py`) includes a `KMedoids` class that implements the K-Medoids clustering algorithm without relying on external libraries. It consists of methods for initialization, assignment, and updating of medoids.

## Pros and Cons

### Pros

- **Robust to Outliers:** K-Medoids is less sensitive to outliers compared to K-Means.
- **Handles Non-Gaussian Distributions:** Suitable for clusters with non-Gaussian distributions or irregular shapes.
- **Interpretable Results:** The medoid provides a representative point that is an actual data point in the dataset.

### Cons

- **Computationally Expensive:** The assignment and update steps make it computationally more expensive than K-Means.
- **Sensitive to Initial Medoids:** Results can vary based on the initial selection of medoids.
- **Limited Scalability:** May struggle with large datasets.

## Example Usage

```python
# Instantiate and fit the KMedoids model
kmedoids = KMedoids(k=3)
kmedoids.fit(X)

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=kmedoids.clusters, cmap='viridis', alpha=0.5)
plt.scatter(kmedoids.medoids[:, 0], kmedoids.medoids[:, 1], c='red', marker='x', s=200)
plt.title('K-Medoids Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```
