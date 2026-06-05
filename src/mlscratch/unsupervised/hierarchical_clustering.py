"""
Hierarchical Agglomerative Clustering (HAC)
============================================
Builds a hierarchy of clusters bottom-up: each sample starts as its own
cluster; at each step the two closest clusters are merged, until all
samples belong to one cluster or a stopping criterion is met.

Linkage criteria implemented
-----------------------------
- 'single'   : distance = min pairwise distance between clusters
- 'complete' : distance = max pairwise distance between clusters
- 'average'  : distance = mean pairwise distance between clusters
- 'ward'     : distance = increase in total within-cluster variance on merge

Only numpy is used.
"""

import numpy as np


class AgglomerativeClustering:
    """
    Hierarchical agglomerative clustering.

    Parameters
    ----------
    n_clusters : int
        Target number of clusters to cut the dendrogram to.
    linkage : str
        One of {'single', 'complete', 'average', 'ward'}.
    """

    def __init__(self, n_clusters: int = 2, linkage: str = "ward"):
        if linkage not in {"single", "complete", "average", "ward"}:
            raise ValueError(
                f"linkage must be one of 'single', 'complete', 'average', "
                f"'ward'. Got '{linkage}'."
            )
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pairwise_distances(X: np.ndarray) -> np.ndarray:
        """Return symmetric Euclidean distance matrix (n x n)."""
        n = len(X)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                D[i, j] = D[j, i] = d
        return D

    def _cluster_distance(
        self,
        c1: list,
        c2: list,
        X: np.ndarray,
        D: np.ndarray,
    ) -> float:
        """Compute linkage distance between two clusters."""
        dists = [D[i, j] for i in c1 for j in c2]

        if self.linkage == "single":
            return min(dists)
        if self.linkage == "complete":
            return max(dists)
        if self.linkage == "average":
            return sum(dists) / len(dists)
        if self.linkage == "ward":
            # Increase in total within-cluster variance
            combined = c1 + c2
            centroid_c1 = X[c1].mean(axis=0)
            centroid_c2 = X[c2].mean(axis=0)
            centroid_merged = X[combined].mean(axis=0)
            wcv_c1 = np.sum((X[c1] - centroid_c1) ** 2)
            wcv_c2 = np.sum((X[c2] - centroid_c2) ** 2)
            wcv_merged = np.sum((X[combined] - centroid_merged) ** 2)
            return wcv_merged - wcv_c1 - wcv_c2
        raise ValueError(f"Unknown linkage '{self.linkage}'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "AgglomerativeClustering":
        """
        Perform hierarchical clustering on X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        """
        n_samples = len(X)
        D = self._pairwise_distances(X)

        # Each sample starts as its own cluster (stored as list of indices)
        clusters = [[i] for i in range(n_samples)]

        while len(clusters) > self.n_clusters:
            min_dist = np.inf
            merge_i, merge_j = 0, 1

            # Find closest pair of clusters
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    d = self._cluster_distance(clusters[i], clusters[j], X, D)
                    if d < min_dist:
                        min_dist = d
                        merge_i, merge_j = i, j

            # Merge
            merged = clusters[merge_i] + clusters[merge_j]
            # Remove old clusters (higher index first to preserve positions)
            clusters = [
                c for idx, c in enumerate(clusters)
                if idx != merge_i and idx != merge_j
            ]
            clusters.append(merged)

        # Assign labels
        labels = np.empty(n_samples, dtype=int)
        for cluster_id, indices in enumerate(clusters):
            for idx in indices:
                labels[idx] = cluster_id

        self.labels_ = labels
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_
