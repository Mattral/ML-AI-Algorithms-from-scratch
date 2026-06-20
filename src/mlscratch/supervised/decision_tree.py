r"""
Decision Trees
==============
From-scratch CART (Classification And Regression Trees), pure numpy.

DecisionTreeClassifier
-----------------------
Binary or multiclass classification. At each node the split that
minimises the weighted child impurity is chosen, where impurity is
either Gini:

.. math::
    G = 1 - \sum_{k=1}^K p_k^2

or Shannon entropy:

.. math::
    H = -\sum_{k=1}^K p_k \log_2 p_k

DecisionTreeRegressor
-----------------------
Minimises weighted variance (mean squared error) of the target within
each child:

.. math::
    \mathrm{MSE} = \frac{1}{W}\sum_i w_i (y_i - \bar y)^2

Both trees support per-sample weights (``sample_weight``), which is
what lets :mod:`mlscratch.supervised.adaboost` and
:mod:`mlscratch.supervised.random_forest` reuse the exact same split
logic instead of re-deriving it.

Algorithm
---------
For each candidate feature the rows are sorted once (`O(n log n)`),
then a single left-to-right vectorised sweep evaluates every possible
split point in `O(n)` using running (weighted) cumulative sums — no
per-split re-scan of the data. Overall a node with `n` samples and
`d` features costs `O(n d log n)`.

Complexity
----------
- Training : O(n d log n) per node, O(depth) nodes on the root path
- Inference: O(depth) per sample
- Space    : O(n_nodes)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────────
# Shared node / validation helpers
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class _Node:
    """A single node of a binary decision tree.

    ``value`` holds the leaf prediction: a class-probability vector for
    :class:`DecisionTreeClassifier`, or a scalar mean for
    :class:`DecisionTreeRegressor`. Internal nodes additionally carry
    ``feature_index`` / ``threshold`` and child references.
    """

    n_samples: int
    weighted_n_samples: float
    impurity: float
    value: object
    feature_index: int | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


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


def _validate_sample_weight(sample_weight: ArrayLike | None, n_samples: int) -> FloatArray:
    if sample_weight is None:
        return np.ones(n_samples, dtype=np.float64)
    w = np.asarray(sample_weight, dtype=np.float64).flatten()
    if w.shape[0] != n_samples:
        raise ValueError(f"sample_weight has {w.shape[0]} entries but X has {n_samples} samples.")
    if np.any(w < 0):
        raise ValueError("sample_weight entries must be non-negative.")
    return w


def _apply(root: _Node, X: FloatArray) -> list[_Node]:
    """Route every row of X to its terminal leaf node and return the leaves."""
    leaves: list[_Node] = []
    for row in X:
        node = root
        while not node.is_leaf:
            node = node.left if row[node.feature_index] <= node.threshold else node.right
        leaves.append(node)
    return leaves


def group_by_leaf(leaves: list[_Node]) -> dict[int, tuple[_Node, list[int]]]:
    """Group sample indices by which leaf object they were routed to.

    Used internally by gradient boosting to re-fit leaf values after the
    tree structure has been chosen by the (cheaper) variance criterion.
    """
    groups: dict[int, tuple[_Node, list[int]]] = {}
    for i, leaf in enumerate(leaves):
        key = id(leaf)
        if key not in groups:
            groups[key] = (leaf, [])
        groups[key][1].append(i)
    return groups


# ──────────────────────────────────────────────────────────────────────────
# DecisionTreeClassifier
# ──────────────────────────────────────────────────────────────────────────


class DecisionTreeClassifier:
    """A binary or multiclass CART decision tree classifier.

    Parameters
    ----------
    max_depth : int | None, default=None
        Maximum tree depth. ``None`` grows nodes until they are pure or
        too small to split.
    min_samples_split : int, default=2
        Minimum number of samples a node must have to be eligible for
        splitting.
    min_samples_leaf : int, default=1
        Minimum number of samples required in each child of a split.
    criterion : str, default='gini'
        Split quality measure: ``'gini'`` or ``'entropy'``.
    random_state : int | None, default=None
        Unused by the splitting rule itself (which is deterministic);
        accepted for API symmetry with ensembles that seed their trees.

    Attributes
    ----------
    tree_ : the fitted root node
    classes_ : sorted unique labels seen during fit
    n_classes_ : number of classes
    n_features_in_ : number of features seen during fit
    feature_importances_ : impurity-decrease-based importances, sums to 1
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        random_state: int | None = None,
    ) -> None:
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if int(min_samples_split) < 2:
            raise ValueError("min_samples_split must be >= 2.")
        if int(min_samples_leaf) < 1:
            raise ValueError("min_samples_leaf must be >= 1.")
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.criterion = criterion
        self.random_state = random_state

        self.tree_: _Node | None = None
        self.classes_: IntArray | None = None
        self.n_classes_: int | None = None
        self.n_features_in_: int | None = None
        self.feature_importances_: FloatArray | None = None

    # -- public API ---------------------------------------------------------

    def fit(
        self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> DecisionTreeClassifier:
        """Grow the decision tree from training data."""
        X_arr, y_raw = _validate_xy(X, y)
        self.classes_, y_idx = np.unique(y_raw, return_inverse=True)
        y_idx = y_idx.astype(np.int64)
        self.n_classes_ = int(self.classes_.size)
        self.n_features_in_ = X_arr.shape[1]
        w = _validate_sample_weight(sample_weight, X_arr.shape[0])

        importances = np.zeros(self.n_features_in_, dtype=np.float64)
        self.tree_ = self._grow(X_arr, y_idx, w, depth=0, importances=importances)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > _EPS else importances
        return self

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Return class-probability estimates, columns ordered as ``classes_``."""
        if self.tree_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X_arr = _validate_x(X)
        leaves = _apply(self.tree_, X_arr)
        return np.vstack([leaf.value for leaf in leaves])

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict the most likely class label for each row of X."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
        """Return (optionally weighted) classification accuracy."""
        X_arr, y_arr = _validate_xy(X, y)
        w = _validate_sample_weight(sample_weight, X_arr.shape[0])
        preds = self.predict(X_arr)
        return float(np.average(preds == y_arr, weights=w))

    def apply(self, X: ArrayLike) -> list[_Node]:
        """Return the terminal leaf node each row of X is routed to."""
        if self.tree_ is None:
            raise RuntimeError("Call fit() before apply().")
        return _apply(self.tree_, _validate_x(X))

    # -- tree construction ----------------------------------------------------

    def _impurity(self, weighted_counts: FloatArray, total_w: float) -> float:
        if total_w <= _EPS:
            return 0.0
        p = weighted_counts / total_w
        if self.criterion == "gini":
            return float(1.0 - np.sum(p**2))
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.log2(np.where(p > 0, p, 1.0))
        return float(-np.sum(np.where(p > 0, p * log_p, 0.0)))

    def _grow(
        self, X: FloatArray, y: IntArray, w: FloatArray, depth: int, importances: FloatArray
    ) -> _Node:
        weighted_counts = np.bincount(y, weights=w, minlength=self.n_classes_)
        total_w = float(weighted_counts.sum())
        if total_w > _EPS:
            proba = weighted_counts / total_w
        else:
            proba = np.full(self.n_classes_, 1.0 / self.n_classes_)
        impurity = self._impurity(weighted_counts, total_w)

        node = _Node(
            n_samples=X.shape[0], weighted_n_samples=total_w, impurity=impurity, value=proba
        )

        can_split = (
            X.shape[0] >= self.min_samples_split
            and impurity > _EPS
            and self.n_classes_ > 1
            and (self.max_depth is None or depth < self.max_depth)
        )
        if can_split:
            feat_idx, threshold, gain = self._best_split(X, y, w, total_w, impurity)
            if feat_idx is not None:
                mask = X[:, feat_idx] <= threshold
                importances[feat_idx] += gain * total_w
                node.feature_index = feat_idx
                node.threshold = threshold
                node.left = self._grow(X[mask], y[mask], w[mask], depth + 1, importances)
                node.right = self._grow(X[~mask], y[~mask], w[~mask], depth + 1, importances)
        return node

    def _best_split(
        self, X: FloatArray, y: IntArray, w: FloatArray, total_w: float, parent_impurity: float
    ) -> tuple[int | None, float | None, float]:
        best_feat, best_thr, best_impurity = None, None, parent_impurity
        for feat in range(self.n_features_in_):
            thr, impurity = self._best_split_feature(X[:, feat], y, w)
            if thr is not None and impurity < best_impurity - _EPS:
                best_impurity, best_feat, best_thr = impurity, feat, thr
        if best_feat is None:
            return None, None, 0.0
        return best_feat, best_thr, parent_impurity - best_impurity

    def _best_split_feature(
        self, col: FloatArray, y: IntArray, w: FloatArray
    ) -> tuple[float | None, float]:
        n = col.shape[0]
        if n < 2:
            return None, np.inf

        order = np.argsort(col, kind="mergesort")
        xs, ys, ws = col[order], y[order], w[order]

        one_hot = np.zeros((n, self.n_classes_), dtype=np.float64)
        one_hot[np.arange(n), ys] = ws
        left_cum = np.cumsum(one_hot, axis=0)
        W_left = np.cumsum(ws)
        total_counts, W_total = left_cum[-1], W_left[-1]
        right_cum = total_counts - left_cum
        W_right = W_total - W_left

        left_sizes = np.arange(1, n)
        right_sizes = n - left_sizes
        valid = (
            (xs[1:] != xs[:-1])
            & (left_sizes >= self.min_samples_leaf)
            & (right_sizes >= self.min_samples_leaf)
        )
        if not np.any(valid):
            return None, np.inf

        Wl, Wr = W_left[:-1], W_right[:-1]
        safe_Wl = np.where(Wl > _EPS, Wl, 1.0)
        safe_Wr = np.where(Wr > _EPS, Wr, 1.0)
        pl = left_cum[:-1] / safe_Wl[:, None]
        pr = right_cum[:-1] / safe_Wr[:, None]

        if self.criterion == "gini":
            imp_l = 1.0 - np.sum(pl**2, axis=1)
            imp_r = 1.0 - np.sum(pr**2, axis=1)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_pl = np.log2(np.where(pl > 0, pl, 1.0))
                log_pr = np.log2(np.where(pr > 0, pr, 1.0))
            imp_l = -np.sum(np.where(pl > 0, pl * log_pl, 0.0), axis=1)
            imp_r = -np.sum(np.where(pr > 0, pr * log_pr, 0.0), axis=1)

        weighted_impurity = np.where(valid, (Wl * imp_l + Wr * imp_r) / W_total, np.inf)
        best_i = int(np.argmin(weighted_impurity))
        if not np.isfinite(weighted_impurity[best_i]):
            return None, np.inf
        threshold = float((xs[best_i] + xs[best_i + 1]) / 2.0)
        return threshold, float(weighted_impurity[best_i])


# ──────────────────────────────────────────────────────────────────────────
# DecisionTreeRegressor
# ──────────────────────────────────────────────────────────────────────────


class DecisionTreeRegressor:
    """A CART decision tree regressor minimising weighted MSE.

    Parameters
    ----------
    max_depth : int | None, default=None
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    random_state : int | None, default=None
        Unused by the (deterministic) splitting rule; kept for API
        symmetry with ensembles.

    Attributes
    ----------
    tree_ : the fitted root node
    n_features_in_ : number of features seen during fit
    feature_importances_ : impurity-decrease-based importances, sums to 1
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
    ) -> None:
        if int(min_samples_split) < 2:
            raise ValueError("min_samples_split must be >= 2.")
        if int(min_samples_leaf) < 1:
            raise ValueError("min_samples_leaf must be >= 1.")
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = random_state

        self.tree_: _Node | None = None
        self.n_features_in_: int | None = None
        self.feature_importances_: FloatArray | None = None

    # -- public API ---------------------------------------------------------

    def fit(
        self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> DecisionTreeRegressor:
        """Grow the regression tree from training data."""
        X_arr, y_arr = _validate_xy(X, y)
        y_arr = y_arr.astype(np.float64)
        self.n_features_in_ = X_arr.shape[1]
        w = _validate_sample_weight(sample_weight, X_arr.shape[0])

        importances = np.zeros(self.n_features_in_, dtype=np.float64)
        self.tree_ = self._grow(X_arr, y_arr, w, depth=0, importances=importances)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > _EPS else importances
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict the target value for each row of X."""
        if self.tree_ is None:
            raise RuntimeError("Call fit() before predict().")
        leaves = self.apply(X)
        return np.array([leaf.value for leaf in leaves], dtype=np.float64)

    def score(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        X_arr, y_arr = _validate_xy(X, y)
        w = _validate_sample_weight(sample_weight, X_arr.shape[0])
        preds = self.predict(X_arr)
        y_mean = float(np.average(y_arr, weights=w))
        ss_res = float(np.sum(w * (y_arr - preds) ** 2))
        ss_tot = float(np.sum(w * (y_arr - y_mean) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > _EPS else 0.0

    def apply(self, X: ArrayLike) -> list[_Node]:
        """Return the terminal leaf node each row of X is routed to."""
        if self.tree_ is None:
            raise RuntimeError("Call fit() before apply().")
        return _apply(self.tree_, _validate_x(X))

    # -- tree construction ----------------------------------------------------

    def _grow(
        self, X: FloatArray, y: FloatArray, w: FloatArray, depth: int, importances: FloatArray
    ) -> _Node:
        total_w = float(w.sum())
        mean = float(np.average(y, weights=w)) if total_w > _EPS else float(np.mean(y))
        variance = float(np.average((y - mean) ** 2, weights=w)) if total_w > _EPS else 0.0

        node = _Node(
            n_samples=X.shape[0], weighted_n_samples=total_w, impurity=variance, value=mean
        )

        can_split = (
            X.shape[0] >= self.min_samples_split
            and variance > _EPS
            and (self.max_depth is None or depth < self.max_depth)
        )
        if can_split:
            feat_idx, threshold, gain = self._best_split(X, y, w, total_w, variance)
            if feat_idx is not None:
                mask = X[:, feat_idx] <= threshold
                importances[feat_idx] += gain * total_w
                node.feature_index = feat_idx
                node.threshold = threshold
                node.left = self._grow(X[mask], y[mask], w[mask], depth + 1, importances)
                node.right = self._grow(X[~mask], y[~mask], w[~mask], depth + 1, importances)
        return node

    def _best_split(
        self, X: FloatArray, y: FloatArray, w: FloatArray, total_w: float, parent_variance: float
    ) -> tuple[int | None, float | None, float]:
        best_feat, best_thr, best_var = None, None, parent_variance
        for feat in range(self.n_features_in_):
            thr, var = self._best_split_feature(X[:, feat], y, w)
            if thr is not None and var < best_var - _EPS:
                best_var, best_feat, best_thr = var, feat, thr
        if best_feat is None:
            return None, None, 0.0
        return best_feat, best_thr, parent_variance - best_var

    def _best_split_feature(
        self, col: FloatArray, y: FloatArray, w: FloatArray
    ) -> tuple[float | None, float]:
        n = col.shape[0]
        if n < 2:
            return None, np.inf

        order = np.argsort(col, kind="mergesort")
        xs, ys, ws = col[order], y[order], w[order]

        wy, wy2 = ys * ws, (ys**2) * ws
        left_sum, left_sum2 = np.cumsum(wy), np.cumsum(wy2)
        W_left = np.cumsum(ws)
        total_sum, total_sum2, W_total = left_sum[-1], left_sum2[-1], W_left[-1]
        right_sum, right_sum2 = total_sum - left_sum, total_sum2 - left_sum2
        W_right = W_total - W_left

        left_sizes = np.arange(1, n)
        right_sizes = n - left_sizes
        valid = (
            (xs[1:] != xs[:-1])
            & (left_sizes >= self.min_samples_leaf)
            & (right_sizes >= self.min_samples_leaf)
        )
        if not np.any(valid):
            return None, np.inf

        Wl, Wr = W_left[:-1], W_right[:-1]
        safe_Wl = np.where(Wl > _EPS, Wl, 1.0)
        safe_Wr = np.where(Wr > _EPS, Wr, 1.0)
        mean_l = left_sum[:-1] / safe_Wl
        mean_r = right_sum[:-1] / safe_Wr
        var_l = np.maximum(left_sum2[:-1] / safe_Wl - mean_l**2, 0.0)
        var_r = np.maximum(right_sum2[:-1] / safe_Wr - mean_r**2, 0.0)

        weighted_var = np.where(valid, (Wl * var_l + Wr * var_r) / W_total, np.inf)
        best_i = int(np.argmin(weighted_var))
        if not np.isfinite(weighted_var[best_i]):
            return None, np.inf
        threshold = float((xs[best_i] + xs[best_i + 1]) / 2.0)
        return threshold, float(weighted_var[best_i])
