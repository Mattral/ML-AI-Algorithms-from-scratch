# Self-Organizing Map (SOM)

## How it Works
The Self-Organizing Map (SOM) is a type of unsupervised learning neural network designed for clustering and visualization of high-dimensional data. The algorithm organizes input data into a lower-dimensional grid, preserving the topological relationships between data points.

1. **Initialization:**
   - Randomly initialize weights for each neuron in the grid.

2. **Training:**
   - For each input vector, find the Best Matching Unit (BMU) – the neuron with the closest weight to the input vector.
   - Update the weights of the BMU and its neighboring neurons using a learning rate and a neighborhood function.
   - Repeat this process for multiple epochs.

3. **Neighborhood Function:**
   - The neighborhood function defines the influence of neighboring neurons during weight updates.
   - Commonly, a Gaussian function is used, where the influence decreases with distance from the BMU.

4. **Learning Rate Decay:**
   - Reduce the learning rate over time to fine-tune the convergence.

## Math Expression
The update rule for weights (ΔW) at each iteration can be expressed as follows:
$$\[ \Delta W_{ij} = \eta \cdot h(i, b) \cdot (X - W_{ij}) \]$$
   - $\( \eta \)$: Learning rate.
   - $\( h(i, b) \)$: Neighborhood function.
   - $\( X \)$: Input vector.
   - $\( W_{ij} \)$: Weight vector of neuron at position (i, j) in the grid.

## Uses
- **Clustering:** SOM can be used for clustering similar data points.
- **Dimensionality Reduction:** Visualization of high-dimensional data in a lower-dimensional space.
- **Feature Mapping:** Mapping input features to specific regions in the grid.

## Pros and Cons
### Pros:
- **Topological Preservation:** Preserves the topological relationships of input data.
- **Visualization:** Enables visual interpretation of complex data structures.
- **Simple Implementation:** Relatively simple and intuitive compared to other neural network architectures.

### Cons:
- **Computational Cost:** Can be computationally expensive for large datasets.
- **Sensitivity to Parameters:** Performance is influenced by parameters like learning rate and neighborhood size.
- **Initialization Dependency:** Sensitive to initial weight configurations.

## Implementation Details
- **Euclidean Distance Calculation:** Used to find the BMU.
- **Agglomerative Approach:** Neurons are updated iteratively, starting with random weights.
- **Visualization:** Matplotlib is commonly used for visualizing the SOM and input data.

This implementation is a basic example, and fine-tuning parameters and using a more sophisticated initialization method could enhance performance for specific datasets.

Please note that the above explanation is an overview, and detailed mathematics may vary based on specific implementations.
