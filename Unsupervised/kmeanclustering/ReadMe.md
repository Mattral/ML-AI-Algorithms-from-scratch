# K-means Clustering Algorithm

## Overview

K-means is a popular unsupervised machine learning algorithm used for clustering data points into groups or clusters based on their similarities. The algorithm aims to partition the data into k clusters, where each cluster is represented by its centroid. Data points within a cluster are more similar to each other than to those in other clusters.

## Algorithm Steps

1. **Initialization:**
   - Select k data points as initial centroids.

2. **Assignment:**
   - Assign each data point to the nearest centroid, creating k clusters.

3. **Update Centroids:**
   - Recalculate the centroids by taking the mean of all data points in each cluster.

4. **Convergence Check:**
   - Check for convergence by measuring the change in centroids.
   - If the change is below a predefined tolerance or a maximum number of iterations is reached, stop.

5. **Repeat Assignment and Update:**
   - Repeat steps 2-4 until convergence or the maximum number of iterations is reached.

## Mathematical Perspective

### Notation

- \( X \): Input data matrix, where each row represents a data point, and each column represents a feature.
- \( k \): Number of clusters.
- \( \text{centroids} \): Matrix representing the current centroid locations.
- \( \text{labels} \): Vector representing the cluster assignment for each data point.
- \( \text{max\_iters} \): Maximum number of iterations.
- \( \text{tol} \): Tolerance for convergence.

### Step 2: Assignment

- **Euclidean Distance:** Assign each data point to the cluster with the nearest centroid based on Euclidean distance:
  
  $$\[ \text{labels}[i] = \arg \min_j \| X[i] - \text{centroids}[j] \|_2 \]$$
  
### Step 3: Update Centroids

- **Mean Calculation:** Update centroids by computing the mean of the data points assigned to each cluster:
  ```math
  \[ \text{new\_centroids}[j] = \frac{1}{\text{count}[j]} \sum_{i=1}^{n} \text{centroids}[j] \]
  ```

### Step 4: Convergence Check

- **Convergence Criterion:** Check for convergence by comparing the change in centroids:
  ```math
  \[ \|\text{new\_centroids} - \text{centroids}\|_2 < \text{tol} \]
  ````

## Experimentation

- Experiment with different values of \( k \), \( \text{max\_iters} \), and \( \text{tol} \) to observe their impact on clustering results.
- Adjust input data preprocessing, such as standardization or normalization, for improved performance.

## Example Usage

```python
import numpy as np

# Example usage with random data
data = np.random.randn(100, 2)
k = 3
centroids, labels = kmeans(data, k)
