r"""
Random Forest
=============
Bootstrap-aggregated ("bagged") ensembles of :class:`DecisionTreeClassifier`
/ :class:`DecisionTreeRegressor` trees, decorrelated by also restricting
each tree to a random subset of features (the "random subspace" method).

Algorithm
---------
For each of ``n_estimators`` trees:

1. Draw a bootstrap sample of ``n`` rows with replacement (if
   ``bootstrap=True``).
2. Draw ``max_features`` columns without replacement.
3. Fit a full (or depth-limited) tree on that bootstrap sample restricted
   to those columns.

``RandomForestClassifier`` combines trees by averaging their
``predict_proba`` output (soft voting) and taking the arg-max; rows
where a particular tree never saw a class during its bootstrap draw are
naturally handled because that tree's probability for the missing
class is implicitly zero, not undefined.

``RandomForestRegressor`` combines trees by averaging their scalar
predictions.

Out-of-bag (OOB) estimation
----------------------------
When ``oob_score=True``, each tree's prediction is also collected for
the ``~37%`` of rows it never trained on (the rows not drawn by its
bootstrap sample), giving an unbiased estimate of generalisation
performance without held-out data.

Complexity
----------
- Training : O(n_estimators * n d log n)
- Inference: O(n_estimators * depth)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_EPS = 1e-12


def _validate_x(X: ArrayLike) -> FloatArray:
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    return X_arr


def _validate_xy(X: ArrayLike, y: ArrayLike) -> tuple[FloatArray, NDArray]:
    X_arr = _validate_x(X)
    y_arr = np.asarray(y).flatten()
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}.")
    return X_arr, y_arr


def _resolve_max_features(max_features: int | float | str | None, n_features: int) -> int:
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError("max_features must be None, int, float, 'sqrt', or 'log2'.")
    if isinstance(max_features, float):
        if not (0.0 < max_features <= 1.0):
            raise ValueError("max_features as a float must be in (0, 1].")
        return max(1, int(round(max_features * n_features)))
    return max(1, min(int(max_features), n_features))


# ──────────────────────────────────────────────────────────────────────────
# RandomForestClassifier
# ──────────────────────────────────────────────────────────────────────────


class RandomForestClassifier:
    """Bagged ensemble of decision-tree classifiers with feature subsampling.

    Parameters
    ----------
    n_estimators : int, default=100
    max_depth : int | None, default=None
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    criterion : str, default='gini'
        ``'gini'`` or ``'entropy'``, forwarded to each tree.
    max_features : int | float | str | None, default='sqrt'
        Number of features considered by each tree: an int (exact count),
        a float in (0, 1] (fraction), ``'sqrt'``, ``'log2'``, or ``None``
        (use all features).
    bootstrap : bool, default=True
        Whether each tree is trained on a bootstrap resample.
    oob_score : bool, default=False
        Whether to compute an out-of-bag accuracy estimate (``oob_score_``).
    random_state : int | None, default=None

    Attributes
    ----------
    estimators_ : list of (tree, feature_indices) tuples
    classes_ : sorted unique labels seen during fit
    feature_importances_ : mean impurity-decrease importance across trees
    oob_score_ : float, only set when ``oob_score=True``
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        max_features: int | float | str | None = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
    ) -> None:
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be >= 1.")
        if oob_score and not bootstrap:
            raise ValueError("oob_score requires bootstrap=True.")
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

        self.estimators_: list[tuple[DecisionTreeClassifier, IntArray]] = []
        self.classes_: IntArray | None = None
        self.n_features_in_: int | None = None
        self.feature_importances_: FloatArray | None = None
        self.oob_score_: float | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> RandomForestClassifier:
        X_arr, y_raw = _validate_xy(X, y)
        self.classes_, y_idx = np.unique(y_raw, return_inverse=True)
        y_idx = y_idx.astype(np.int64)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features
        n_classes = self.classes_.size
        n_feat_sub = _resolve_max_features(self.max_features, n_features)

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        importances = np.zeros(n_features, dtype=np.float64)

        oob_proba = np.zeros((n_samples, n_classes)) if self.oob_score else None
        oob_count = np.zeros(n_samples, dtype=np.int64) if self.oob_score else None

        for _ in range(self.n_estimators):
            sample_idx = (
                rng.integers(0, n_samples, n_samples) if self.bootstrap else np.arange(n_samples)
            )
            feat_idx = rng.choice(n_features, size=n_feat_sub, replace=False)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
            )
            tree.fit(X_arr[sample_idx][:, feat_idx], y_idx[sample_idx])
            self.estimators_.append((tree, feat_idx))
            importances[feat_idx] += tree.feature_importances_

            if self.oob_score:
                in_bag = np.zeros(n_samples, dtype=bool)
                in_bag[sample_idx] = True
                oob_idx = np.flatnonzero(~in_bag)
                if oob_idx.size:
                    proba = tree.predict_proba(X_arr[oob_idx][:, feat_idx])
                    oob_proba[np.ix_(oob_idx, tree.classes_)] += proba
                    oob_count[oob_idx] += 1

        importances /= self.n_estimators
        total = importances.sum()
        self.feature_importances_ = importances / total if total > _EPS else importances

        if self.oob_score:
            has_oob = oob_count > 0
            if np.any(has_oob):
                pred_idx = np.argmax(oob_proba[has_oob], axis=1)
                self.oob_score_ = float(np.mean(pred_idx == y_idx[has_oob]))
            else:
                self.oob_score_ = float("nan")
        return self

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict_proba().")
        X_arr = _validate_x(X)
        n_classes = self.classes_.size
        proba = np.zeros((X_arr.shape[0], n_classes), dtype=np.float64)
        for tree, feat_idx in self.estimators_:
            p = tree.predict_proba(X_arr[:, feat_idx])
            proba[:, tree.classes_] += p
        proba /= len(self.estimators_)
        return proba

    def predict(self, X: ArrayLike) -> NDArray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        X_arr, y_arr = _validate_xy(X, y)
        return float(np.mean(self.predict(X_arr) == y_arr))


# ──────────────────────────────────────────────────────────────────────────
# RandomForestRegressor
# ──────────────────────────────────────────────────────────────────────────


class RandomForestRegressor:
    """Bagged ensemble of decision-tree regressors with feature subsampling.

    Parameters mirror :class:`RandomForestClassifier`, except
    ``max_features`` defaults to ``1.0`` (consider all features at every
    split, the conventional bagging-regressor default) and there is no
    ``criterion`` choice (trees always split on weighted MSE).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
    ) -> None:
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be >= 1.")
        if oob_score and not bootstrap:
            raise ValueError("oob_score requires bootstrap=True.")
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

        self.estimators_: list[tuple[DecisionTreeRegressor, IntArray]] = []
        self.n_features_in_: int | None = None
        self.feature_importances_: FloatArray | None = None
        self.oob_score_: float | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> RandomForestRegressor:
        X_arr, y_arr = _validate_xy(X, y)
        y_arr = y_arr.astype(np.float64)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features
        n_feat_sub = _resolve_max_features(self.max_features, n_features)

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        importances = np.zeros(n_features, dtype=np.float64)

        oob_sum = np.zeros(n_samples) if self.oob_score else None
        oob_count = np.zeros(n_samples, dtype=np.int64) if self.oob_score else None

        for _ in range(self.n_estimators):
            sample_idx = (
                rng.integers(0, n_samples, n_samples) if self.bootstrap else np.arange(n_samples)
            )
            feat_idx = rng.choice(n_features, size=n_feat_sub, replace=False)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_arr[sample_idx][:, feat_idx], y_arr[sample_idx])
            self.estimators_.append((tree, feat_idx))
            importances[feat_idx] += tree.feature_importances_

            if self.oob_score:
                in_bag = np.zeros(n_samples, dtype=bool)
                in_bag[sample_idx] = True
                oob_idx = np.flatnonzero(~in_bag)
                if oob_idx.size:
                    oob_sum[oob_idx] += tree.predict(X_arr[oob_idx][:, feat_idx])
                    oob_count[oob_idx] += 1

        importances /= self.n_estimators
        total = importances.sum()
        self.feature_importances_ = importances / total if total > _EPS else importances

        if self.oob_score:
            has_oob = oob_count > 0
            if np.any(has_oob):
                oob_pred = oob_sum[has_oob] / oob_count[has_oob]
                y_true = y_arr[has_oob]
                ss_res = np.sum((y_true - oob_pred) ** 2)
                ss_tot = np.sum((y_true - y_true.mean()) ** 2)
                self.oob_score_ = float(1.0 - ss_res / ss_tot) if ss_tot > _EPS else 0.0
            else:
                self.oob_score_ = float("nan")
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict().")
        X_arr = _validate_x(X)
        preds = np.zeros(X_arr.shape[0], dtype=np.float64)
        for tree, feat_idx in self.estimators_:
            preds += tree.predict(X_arr[:, feat_idx])
        return preds / len(self.estimators_)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        X_arr, y_arr = _validate_xy(X, y)
        preds = self.predict(X_arr)
        ss_res = np.sum((y_arr - preds) ** 2)
        ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > _EPS else 0.0
