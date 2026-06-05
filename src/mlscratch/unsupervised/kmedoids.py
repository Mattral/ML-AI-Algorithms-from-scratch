"""
K-Medoids Clustering (PAM — Partitioning Around Medoids)
=========================================================
Similar to K-Means but the cluster representatives (medoids) must be actual
data points, making the algorithm more robust to outliers and compatible
with non-Euclidean distances.

Algorithm (simplified PAM)
--------------------------
1. Randomly initialise K medoids from the dataset.
2. Assign every point to its nearest medoid.
3. For each cluster, try every non-medoid point as a new medoid;
   keep the swap if it reduces the total cluster cost.
4. Repeat steps 2-3 until no swap improves the cost.

Only numpy and Python stdlib are used.
"""

import numpy as np


class KMedoids:
    """
    K-Medoids clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters / medoids.
    max_iter : int
        Maximum number of swap iterations.
    random_state : int or None
        Seed for reproducible medoid initialisation.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 100,
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.medoid_indices_ = None   # indices into X of the medoids
        self.labels_ = None
        self.inertia_ = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pairwise_distances(X: np.ndarray) -> np.ndarray:
        """Return Euclidean distance matrix of shape (n, n)."""
        n = len(X)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                D[i, j] = D[j, i] = d
        return D

    def _assign_labels(self, D: np.ndarray, medoids: list) -> np.ndarray:
        """Assign each point to its nearest medoid."""
        dist_to_medoids = D[:, medoids]          # (n, K)
        return np.argmin(dist_to_medoids, axis=1)

    def _total_cost(self, D: np.ndarray, medoids: list, labels: np.ndarray) -> float:
        """Sum of distances from each point to its medoid."""
        cost = 0.0
        for k, m in enumerate(medoids):
            members = np.where(labels == k)[0]
            cost += D[members, m].sum()
        return float(cost)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "KMedoids":
        """
        Fit K-Medoids to X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        """
        rng = np.random.default_rng(self.random_state)
        n_samples = len(X)
        D = self._pairwise_distances(X)

        # 1. Initialise medoids
        medoids = rng.choice(n_samples, self.n_clusters, replace=False).tolist()

        for _ in range(self.max_iter):
            labels = self._assign_labels(D, medoids)
            current_cost = self._total_cost(D, medoids, labels)
            improved = False

            for k in range(self.n_clusters):
                cluster_members = np.where(labels == k)[0].tolist()
                for candidate in cluster_members:
                    if candidate in medoids:
                        continue
                    new_medoids = medoids.copy()
                    new_medoids[k] = candidate
                    new_labels = self._assign_labels(D, new_medoids)
                    new_cost = self._total_cost(D, new_medoids, new_labels)

                    if new_cost < current_cost:
                        medoids = new_medoids
                        labels = new_labels
                        current_cost = new_cost
                        improved = True

            if not improved:
                break

        self.medoid_indices_ = np.array(medoids, dtype=int)
        self.labels_ = self._assign_labels(D, medoids)
        self.inertia_ = self._total_cost(D, medoids, self.labels_)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Return the actual medoid data points."""
        return None  # set by fit via fit_predict path; use medoid_indices_
