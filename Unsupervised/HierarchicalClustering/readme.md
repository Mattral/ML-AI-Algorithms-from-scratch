# Hierarchical Clustering

Hierarchical Clustering is a clustering algorithm that organizes data into a tree-like structure based on the similarity between data points. In this implementation, we use the agglomerative approach, where each data point starts as its own cluster and clusters are successively merged based on their similarity.

## How it Works

1. **Load the Dataset:**
   - The algorithm begins by loading the dataset from a CSV file. In this example, we use the 'housing.csv' dataset, focusing on the 'median_income' feature.

2. **Euclidean Distance:**
   - We define a function to calculate the Euclidean distance between two points. This distance metric measures the straight-line distance between two points in space.
# Real-Life Uses

1. **Customer Segmentation:**
   - Identifying similar groups of customers based on their purchasing behavior.

2. **Biology:**
   - Classifying species based on genetic similarities.

3. **Image Segmentation:**
   - Grouping pixels in an image based on color or intensity.

# Pros and Cons

## Pros:

- **Simple and Intuitive:**
  - Easy to understand and implement.

- **Hierarchy Representation:**
  - Provides a hierarchical structure, allowing analysis at different levels.

## Cons:

- **Computational Cost:**
  - Can be computationally expensive, especially for large datasets.

- **Memory Usage:**
  - Requires storage of candidate and frequent itemsets, leading to high memory usage.

# Implementation Details

- **Euclidean Distance Calculation:**
  - We use the Euclidean distance (\(d\)) as a measure of similarity:
    $$\[ d(x_1, x_2) = \sqrt{\sum_{i=1}^{n} (x_{1i} - x_{2i})^2} \]$$

- **Agglomerative Approach:**
  - The algorithm starts with each data point as a singleton cluster and successively merges the closest clusters.

- **Visualization:**
  - Matplotlib is used for visualizing the clusters.

Feel free to experiment with different linkage methods and parameters for a better understanding of the algorithm's behavior.
