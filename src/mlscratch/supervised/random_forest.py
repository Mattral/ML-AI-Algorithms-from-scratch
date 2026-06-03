"""
Random Forest Classifier
========================

A from-scratch random forest ensemble built from decision tree classifiers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .decision_tree import DecisionTreeClassifier

FloatArray = NDArray[np.float64]


def _validate_classification_inputs(
    X: ArrayLike, y: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int).flatten()
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}."
        )
    return X_arr, y_arr


class RandomForestClassifier:
    """Random forest classifier using bootstrap aggregation of decision trees."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | str | None = "sqrt",
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_: list[tuple[DecisionTreeClassifier, NDArray[np.int64]]] = []
        self.feature_importances_: FloatArray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RandomForestClassifier":
        X_arr, y_arr = _validate_classification_inputs(X, y)
        rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = X_arr.shape[1]
        features_per_tree = self._resolve_max_features(self.n_features_in_)

        self.estimators_ = []
        importances: FloatArray = np.zeros(self.n_features_in_, dtype=np.float64)

        for _ in range(self.n_estimators):
            indices = rng.choice(X_arr.shape[0], size=X_arr.shape[0], replace=True)
            feature_indices = rng.choice(self.n_features_in_, size=features_per_tree, replace=False)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_arr[indices][:, feature_indices], y_arr[indices])
            self.estimators_.append((tree, feature_indices))
            importances += self._tree_importances(tree, feature_indices)

        self.feature_importances_ = importances / float(self.n_estimators)
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict().")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")
        predictions = np.vstack(
            [tree.predict(X_arr[:, features]) for tree, features in self.estimators_]
        )
        return np.array([np.bincount(row).argmax() for row in predictions.T], dtype=np.int64)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        X_arr, y_arr = _validate_classification_inputs(X, y)
        return float(np.mean(self.predict(X_arr) == y_arr))

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if self.max_features == "log2":
                return max(1, int(np.log2(n_features)))
            raise ValueError("max_features must be None, int, 'sqrt', or 'log2'.")
        return int(self.max_features)

    def _tree_importances(
        self, tree: DecisionTreeClassifier, feature_indices: NDArray[np.int64]
    ) -> FloatArray:
        counts = np.zeros(self.n_features_in_, dtype=np.float64)
        self._accumulate_importance(tree.tree_, counts, feature_indices)
        return counts

    def _accumulate_importance(
        self,
        node: dict | None,
        counts: FloatArray,
        feature_indices: NDArray[np.int64],
    ) -> None:
        if node is None or "feature_index" not in node:
            return
        counts[feature_indices[node["feature_index"]]] += 1.0
        self._accumulate_importance(node["left"], counts, feature_indices)
        self._accumulate_importance(node["right"], counts, feature_indices)
