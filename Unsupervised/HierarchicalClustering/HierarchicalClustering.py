import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the housing dataset from CSV file
df = pd.read_csv('housing.csv')

# Extract features and target variable
X = df[['median_income']].values

def euclidean_distance(x1, x2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    - x1 (numpy.ndarray): First point.
    - x2 (numpy.ndarray): Second point.

    Returns:
    - float: Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

def hierarchical_clustering(data, linkage='single'):
    """
    Perform hierarchical clustering on the given data.

    Parameters:
    - data (numpy.ndarray): Input data.
    - linkage (str): Linkage method ('single', 'complete', 'average', etc.).

    Returns:
    - list: List of clusters, where each cluster is a list of indices.
    """
    clusters = [[i] for i in range(len(data))]
    
    while len(clusters) > 1:
        min_distance = float('inf')
        merge_indices = (0, 0)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                for m in clusters[i]:
                    for n in clusters[j]:
                        distance = euclidean_distance(data[m], data[n])
                        if distance < min_distance:
                            min_distance = distance
                            merge_indices = (i, j)

        i, j = merge_indices
        clusters[i].extend(clusters[j])
        del clusters[j]

    return clusters

# Perform hierarchical clustering
clusters = hierarchical_clustering(X)

# Visualize the results
colors = ['red', 'green', 'blue', 'purple', 'orange']
for i, cluster in enumerate(clusters):
    cluster_points = X[cluster]
    plt.scatter(cluster_points, np.zeros_like(cluster_points), color=colors[i % len(colors)], label=f'Cluster {i+1}')

plt.title('Hierarchical Clustering')
plt.xlabel('Median Income')
plt.legend()
plt.show()
