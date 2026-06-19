"""
K-Nearest Neighbours (KNN)
===========================
Instance-based (lazy) learning — no training phase; prediction queries
the k most similar training samples.

Distance metrics supported:  euclidean, manhattan, minkowski
Weighting:  'uniform' (vote equally) or 'distance' (weight by 1/d)

KNeighboursClassifier — majority-vote (weighted) classification
KNeighboursRegressor  — mean (weighted) of k nearest target values

Time complexity: O(n·d) per prediction (brute-force).
Only numpy and Python stdlib are used.
"""

from __future__ import annotations
import numpy as np
from collections import Counter


def _pairwise_distances(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "euclidean",
    p: float = 2.0,
) -> np.ndarray:
    """Return (n_X, n_Y) distance matrix."""
    if metric == "euclidean":
        # ||x-y||² = ||x||² + ||y||² - 2 x·y
        sq_X = np.sum(X ** 2, axis=1, keepdims=True)
        sq_Y = np.sum(Y ** 2, axis=1, keepdims=True)
        dist2 = sq_X + sq_Y.T - 2 * X @ Y.T
        return np.sqrt(np.maximum(dist2, 0.0))
    elif metric == "manhattan":
        return np.sum(np.abs(X[:, np.newaxis, :] - Y[np.newaxis, :, :]), axis=2)
    elif metric == "minkowski":
        return np.sum(
            np.abs(X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** p, axis=2
        ) ** (1.0 / p)
    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose euclidean, manhattan, minkowski.")


class KNeighboursClassifier:
    """
    K-Nearest Neighbours classifier.

    Parameters
    ----------
    n_neighbors : int
    weights     : str   'uniform' | 'distance'
    metric      : str   'euclidean' | 'manhattan' | 'minkowski'
    p           : float Minkowski order (only when metric='minkowski')
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "euclidean",
        p: float = 2.0,
    ) -> None:
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'.")
        self.n_neighbors = n_neighbors
        self.weights     = weights
        self.metric      = metric
        self.p           = p
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self.classes_:   np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNeighboursClassifier":
        self._X_train = X.copy()
        self._y_train = y.copy()
        self.classes_ = np.unique(y)
        return self

    def _get_knn(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, distances) of k nearest neighbours for each row of X."""
        D = _pairwise_distances(X, self._X_train, self.metric, self.p)
        # Partition (faster than full sort for large n)
        k = min(self.n_neighbors, len(self._X_train))
        idx  = np.argpartition(D, k - 1, axis=1)[:, :k]
        dists = D[np.arange(len(X))[:, np.newaxis], idx]
        # Sort within the k neighbours
        order = np.argsort(dists, axis=1)
        idx   = idx[np.arange(len(X))[:, np.newaxis], order]
        dists = dists[np.arange(len(X))[:, np.newaxis], order]
        return idx, dists

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx, dists = self._get_knn(X)
        neighbours = self._y_train[idx]              # (n_test, k)

        if self.weights == "uniform":
            preds = [Counter(row).most_common(1)[0][0] for row in neighbours]
        else:
            preds = []
            for i in range(len(X)):
                d = dists[i]
                # Avoid division by zero for exact matches
                if np.any(d == 0):
                    exact = neighbours[i][d == 0]
                    preds.append(Counter(exact).most_common(1)[0][0])
                else:
                    w = 1.0 / d
                    vote: dict = {}
                    for cls, wi in zip(neighbours[i], w):
                        vote[cls] = vote.get(cls, 0.0) + wi
                    preds.append(max(vote, key=vote.get))
        return np.array(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        idx, dists = self._get_knn(X)
        n_test  = len(X)
        n_cls   = len(self.classes_)
        cls_idx = {c: i for i, c in enumerate(self.classes_)}
        proba   = np.zeros((n_test, n_cls))

        for i in range(n_test):
            labels = self._y_train[idx[i]]
            if self.weights == "uniform":
                for lbl in labels:
                    proba[i, cls_idx[lbl]] += 1.0
            else:
                d = dists[i]
                if np.any(d == 0):
                    for lbl in labels[d == 0]:
                        proba[i, cls_idx[lbl]] += 1.0
                else:
                    for lbl, wi in zip(labels, 1.0 / d):
                        proba[i, cls_idx[lbl]] += wi
            proba[i] /= proba[i].sum() + 1e-12

        return proba


class KNeighboursRegressor:
    """
    K-Nearest Neighbours regressor.

    Parameters
    ----------
    n_neighbors : int
    weights     : str   'uniform' | 'distance'
    metric      : str   'euclidean' | 'manhattan' | 'minkowski'
    p           : float Minkowski order
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "euclidean",
        p: float = 2.0,
    ) -> None:
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'.")
        self.n_neighbors = n_neighbors
        self.weights     = weights
        self.metric      = metric
        self.p           = p
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNeighboursRegressor":
        self._X_train = X.copy()
        self._y_train = y.astype(float).copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        clf = KNeighboursClassifier(self.n_neighbors, self.weights,
                                     self.metric, self.p)
        clf._X_train = self._X_train
        clf._y_train = self._y_train
        idx, dists = clf._get_knn(X)
        neighbours = self._y_train[idx]

        if self.weights == "uniform":
            return neighbours.mean(axis=1)

        preds = np.zeros(len(X))
        for i in range(len(X)):
            d = dists[i]
            if np.any(d == 0):
                preds[i] = neighbours[i][d == 0].mean()
            else:
                w = 1.0 / d
                preds[i] = np.dot(w, neighbours[i]) / w.sum()
        return preds
