# t-Distributed Stochastic Neighbor Embedding (t-SNE)

## Overview:

t-SNE is a dimensionality reduction technique that maps high-dimensional data to a lower-dimensional space while preserving pairwise similarities between data points. It is particularly useful for visualizing high-dimensional data in two or three dimensions.

## Mathematical Expressions:

1. **Pairwise Euclidean Distances:**

   $$\[ \text{distances}_{ij} = \lVert \text{data}_i - \text{data}_j \rVert \]$$

2. **Conditional Probabilities:**

   The conditional probability \(P_{j|i}\) is computed using a Student's t-distribution with perplexity \(Perp\):
```math
   \[ P_{j|i} = \frac{\exp\left(-\frac{\lVert \text{data}_i - \text{data}_j \rVert^2}{2\sigma_i^2}\right)}{\sum_{k \neq i}\exp\left(-\frac{\lVert \text{data}_i - \text{data}_k \rVert^2}{2\sigma_i^2}\right)} \]
```
   where \(\sigma_i\) is adjusted to achieve the desired perplexity.

3. **Gradient of t-SNE:**

   The gradient is computed as:
```math
   \[ \text{grad}_i = 4 \sum_j (P_{j|i} - Q_{j|i}) \left(\text{data}_i - \text{data}_j\right) \left(1 + \lVert \text{data}_i - \text{data}_j \rVert^2\right)^{-1} \]
```
   where $\(Q_{j|i}\)$ is the low-dimensional similarity.

## Algorithm Steps:

1. **Initialize Low-Dimensional Representation:**
   - Initialize the low-dimensional representation randomly.

2. **Compute Pairwise Distances:**
   - Compute pairwise Euclidean distances between data points.

3. **Optimize Conditional Probabilities:**
   - Use binary search to adjust $\(\sigma_i\)$ for each data point to achieve the target perplexity.
   - Compute conditional probabilities $\(P_{j|i}\)$ for each pair of data points.

4. **Optimize Low-Dimensional Representation:**
   - Minimize the mismatch between $\(P_{j|i}\)$ and low-dimensional similarities $\(Q_{j|i}\)$ using gradient descent.
   - Update the low-dimensional representation.

5. **Repeat Optimization:**
   - Repeat steps 3 and 4 until convergence.

## Uses:

- **Visualization:** t-SNE is commonly used for visualizing high-dimensional data in a reduced-dimensional space.
  
- **Clustering Analysis:** t-SNE can reveal clusters and patterns in the data.

## Pros and Cons:

**Pros:**
- **Preservation of Local Structure:** t-SNE effectively preserves local similarities.
- **Visualization:** Provides a visually appealing representation of high-dimensional data.

**Cons:**
- **Computational Complexity:** t-SNE can be computationally expensive, especially for large datasets.
- **Sensitivity to Parameters:** The performance may depend on the choice of perplexity and learning rate.
