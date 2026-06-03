"""
Decision Tree Classifier
========================

A from-scratch CART decision tree classifier using Gini impurity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _validate_classification_inputs(
    X: ArrayLike, y: ArrayLike,
) -> tuple[FloatArray, NDArray[np.int64]]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int).flatten()
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}."
        )
    return X_arr, y_arr


class DecisionTreeClassifier:
    r"""A binary or multiclass decision tree classifier.

    The CART decision rule splits nodes to minimize weighted Gini impurity:

    .. math::
        G = \sum_{k=1}^K p_k (1 - p_k)

    At each node, the best split minimizes:

    .. math::
        \frac{n_{left}}{n} G_{left} + \frac{n_{right}}{n} G_{right}
    """

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2) -> None:
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.n_classes_: int | None = None
        self.n_features_in_: int | None = None
        self.tree_: dict | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "DecisionTreeClassifier":
        """Grow the decision tree from training data."""
        X_arr, y_arr = _validate_classification_inputs(X, y)
        self.n_features_in_ = X_arr.shape[1]
        self.n_classes_ = int(len(np.unique(y_arr)))
        self.tree_ = self._grow_tree(X_arr, y_arr, depth=0)
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Predict class labels for X."""
        if self.tree_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")
        return np.array([self._predict_row(row, self.tree_) for row in X_arr], dtype=np.int64)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy on the given data."""
        X_arr, y_arr = _validate_classification_inputs(X, y)
        y_pred = self.predict(X_arr)
        return float(np.mean(y_pred == y_arr))

    def _gini(self, y: NDArray[np.int64]) -> float:
        if y.size == 0:
            return 0.0
        proportions = np.bincount(y, minlength=self.n_classes_) / y.size
        return float(np.sum(proportions * (1.0 - proportions)))

    def _best_split(self, X: FloatArray, y: NDArray[np.int64]) -> tuple[int | None, float | None]:
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None

        best_idx = None
        best_thr = None
        best_impurity = self._gini(y)

        for idx in range(n_features):
            sorted_indices = np.argsort(X[:, idx])
            thresholds = X[sorted_indices, idx]
            labels = y[sorted_indices]

            left_counts = np.zeros(self.n_classes_, dtype=int)
            right_counts = np.bincount(labels, minlength=self.n_classes_)

            for i in range(1, n_samples):
                label = labels[i - 1]
                left_counts[label] += 1
                right_counts[label] -= 1
                if thresholds[i] == thresholds[i - 1]:
                    continue

                left_size = i
                right_size = n_samples - i
                if left_size < self.min_samples_split or right_size < self.min_samples_split:
                    continue

                left_gini = 1.0 - np.sum((left_counts / left_size) ** 2)
                right_gini = 1.0 - np.sum((right_counts / right_size) ** 2)
                impurity = (left_size * left_gini + right_size * right_gini) / n_samples

                if impurity < best_impurity:
                    best_impurity = float(impurity)
                    best_idx = idx
                    best_thr = float((thresholds[i] + thresholds[i - 1]) / 2.0)

        return best_idx, best_thr

    def _grow_tree(self, X: FloatArray, y: NDArray[np.int64], depth: int) -> dict:
        node = {
            "n_samples": X.shape[0],
            "n_classes": int(np.bincount(y, minlength=self.n_classes_).argmax()),
            "class": int(np.bincount(y, minlength=self.n_classes_).argmax()),
        }

        if self.max_depth is None or depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                mask = X[:, idx] < thr
                left = self._grow_tree(X[mask], y[mask], depth + 1)
                right = self._grow_tree(X[~mask], y[~mask], depth + 1)
                node.update({"feature_index": idx, "threshold": thr, "left": left, "right": right})
        return node

    def _predict_row(self, x: FloatArray, node: dict) -> int:
        while "feature_index" in node:
            if x[node["feature_index"]] < node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return int(node["class"])
