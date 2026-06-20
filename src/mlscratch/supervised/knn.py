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

from collections import Counter

import numpy as np


def _validate_x(X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    return X_arr


def _validate_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_arr = _validate_x(X)
    y_arr = np.asarray(y).flatten()
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}.")
    return X_arr, y_arr


def _pairwise_distances(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "euclidean",
    p: float = 2.0,
) -> np.ndarray:
    """Return (n_X, n_Y) distance matrix."""
    if metric == "euclidean":
        # ||x-y||² = ||x||² + ||y||² - 2 x·y
        sq_X = np.sum(X**2, axis=1, keepdims=True)
        sq_Y = np.sum(Y**2, axis=1, keepdims=True)
        dist2 = sq_X + sq_Y.T - 2 * X @ Y.T
        return np.sqrt(np.maximum(dist2, 0.0))
    elif metric == "manhattan":
        return np.sum(np.abs(X[:, np.newaxis, :] - Y[np.newaxis, :, :]), axis=2)
    elif metric == "minkowski":
        return np.sum(np.abs(X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** p, axis=2) ** (1.0 / p)
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
        self.weights = weights
        self.metric = metric
        self.p = p
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNeighboursClassifier:
        X_arr, y_arr = _validate_xy(X, y)
        self._X_train = X_arr
        self._y_train = y_arr
        self.classes_ = np.unique(y_arr)
        return self

    def _get_knn(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, distances) of k nearest neighbours for each row of X."""
        if self._X_train is None:
            raise RuntimeError("Call fit() before predict().")
        D = _pairwise_distances(X, self._X_train, self.metric, self.p)
        # Partition (faster than full sort for large n)
        k = min(self.n_neighbors, len(self._X_train))
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        dists = D[np.arange(len(X))[:, np.newaxis], idx]
        # Sort within the k neighbours
        order = np.argsort(dists, axis=1)
        idx = idx[np.arange(len(X))[:, np.newaxis], order]
        dists = dists[np.arange(len(X))[:, np.newaxis], order]
        return idx, dists

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = _validate_x(X)
        idx, dists = self._get_knn(X_arr)
        neighbours = self._y_train[idx]  # (n_test, k)

        if self.weights == "uniform":
            preds = [Counter(row).most_common(1)[0][0] for row in neighbours]
        else:
            preds = []
            for i in range(len(X_arr)):
                d = dists[i]
                # Avoid division by zero for exact matches
                if np.any(d == 0):
                    exact = neighbours[i][d == 0]
                    preds.append(Counter(exact).most_common(1)[0][0])
                else:
                    w = 1.0 / d
                    vote: dict = {}
                    for cls, wi in zip(neighbours[i], w, strict=True):
                        vote[cls] = vote.get(cls, 0.0) + wi
                    preds.append(max(vote, key=vote.get))
        return np.array(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = _validate_x(X)
        idx, dists = self._get_knn(X_arr)
        n_test = len(X_arr)
        n_cls = len(self.classes_)
        cls_idx = {c: i for i, c in enumerate(self.classes_)}
        proba = np.zeros((n_test, n_cls))

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
                    for lbl, wi in zip(labels, 1.0 / d, strict=True):
                        proba[i, cls_idx[lbl]] += wi
            proba[i] /= proba[i].sum() + 1e-12

        return proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy on the given data."""
        X_arr, y_arr = _validate_xy(X, y)
        return float(np.mean(self.predict(X_arr) == y_arr))


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
        self.weights = weights
        self.metric = metric
        self.p = p
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNeighboursRegressor:
        X_arr, y_arr = _validate_xy(X, y)
        self._X_train = X_arr
        self._y_train = y_arr.astype(float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._X_train is None:
            raise RuntimeError("Call fit() before predict().")
        X_arr = _validate_x(X)
        clf = KNeighboursClassifier(self.n_neighbors, self.weights, self.metric, self.p)
        clf._X_train = self._X_train
        clf._y_train = self._y_train
        idx, dists = clf._get_knn(X_arr)
        neighbours = self._y_train[idx]

        if self.weights == "uniform":
            return neighbours.mean(axis=1)

        preds = np.zeros(len(X_arr))
        for i in range(len(X_arr)):
            d = dists[i]
            if np.any(d == 0):
                preds[i] = neighbours[i][d == 0].mean()
            else:
                w = 1.0 / d
                preds[i] = np.dot(w, neighbours[i]) / w.sum()
        return preds

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        X_arr, y_arr = _validate_xy(X, y)
        preds = self.predict(X_arr)
        y_arr = y_arr.astype(float)
        ss_res = float(np.sum((y_arr - preds) ** 2))
        ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


# American-spelling aliases (sklearn-style) for ergonomics.
KNeighborsClassifier = KNeighboursClassifier
KNeighborsRegressor = KNeighboursRegressor
