"""
Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
=====================================================================
Groups together points that are closely packed (high density regions)
and marks points in low-density regions as outliers (noise).

Key ideas
---------
- eps      : neighbourhood radius
- min_samples : minimum number of points to form a dense region (core point)
- Core point   : has >= min_samples neighbours within eps
- Border point : within eps of a core point, but not a core point itself
- Noise point  : neither core nor border

Time complexity : O(n^2) with the naive distance matrix approach used here.
Only numpy and basic Python stdlib are used.
"""

import numpy as np


class DBSCAN:
    """
    DBSCAN clustering.

    Parameters
    ----------
    eps : float
        Maximum distance between two samples to be considered neighbours.
    min_samples : int
        Minimum number of samples in a neighbourhood for a point to be
        labelled a core point (includes the point itself).
    """

    NOISE = -1
    UNVISITED = 0

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None          # cluster label per sample (-1 = noise)
        self.core_sample_indices_ = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance between two 1-D vectors."""
        return np.sqrt(np.sum((a - b) ** 2))

    def _region_query(self, X: np.ndarray, point_idx: int) -> list:
        """Return indices of all points within eps of X[point_idx]."""
        neighbours = []
        for idx in range(len(X)):
            if self._euclidean_distance(X[point_idx], X[idx]) <= self.eps:
                neighbours.append(idx)
        return neighbours

    def _expand_cluster(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        point_idx: int,
        neighbours: list,
        cluster_id: int,
    ) -> None:
        """Grow a cluster starting from point_idx."""
        labels[point_idx] = cluster_id
        seed_set = list(neighbours)          # mutable working queue

        i = 0
        while i < len(seed_set):
            current = seed_set[i]

            # If this was previously labelled noise, reassign to cluster
            if labels[current] == self.NOISE:
                labels[current] = cluster_id

            # If unvisited, visit it now
            if labels[current] == self.UNVISITED:
                labels[current] = cluster_id
                current_neighbours = self._region_query(X, current)

                # If it is itself a core point, add its neighbours to the queue
                if len(current_neighbours) >= self.min_samples:
                    seed_set += current_neighbours   # may add duplicates; OK

            i += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """
        Fit DBSCAN on dataset X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        n_samples = len(X)
        labels = np.full(n_samples, self.UNVISITED, dtype=int)
        cluster_id = 0

        for idx in range(n_samples):
            if labels[idx] != self.UNVISITED:
                continue  # already processed

            neighbours = self._region_query(X, idx)

            if len(neighbours) < self.min_samples:
                labels[idx] = self.NOISE          # mark as noise for now
            else:
                cluster_id += 1
                self._expand_cluster(X, labels, idx, neighbours, cluster_id)

        self.labels_ = labels
        self.core_sample_indices_ = np.array(
            [i for i in range(n_samples)
             if len(self._region_query(X, i)) >= self.min_samples],
            dtype=int,
        )
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and return cluster labels.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            -1 for noise, integers >= 1 for clusters.
        """
        self.fit(X)
        return self.labels_
