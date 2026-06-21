r"""
Gradient Boosting
==================
Friedman's TreeBoost: an additive ensemble of shallow regression trees,
each fit to the negative gradient ("pseudo-residual") of a loss
function evaluated at the current ensemble prediction, then shrunk by
a learning rate and added to the running prediction.

GradientBoostingRegressor
--------------------------
``loss='squared_error'``:

.. math::
    F_m(x) = F_{m-1}(x) + \eta\, h_m(x), \qquad h_m \text{ fit to } y - F_{m-1}

A regression tree's leaf-mean is already the exact minimiser of
squared error, so no further leaf adjustment is needed — residual
fitting alone implements gradient descent in function space.

``loss='absolute_error'`` (LAD-TreeBoost):
:math:`h_m` is fit to :math:`\mathrm{sign}(y-F_{m-1})` to choose split
*structure*, then every leaf value is replaced by the **median**
residual of the samples routed there — the closed-form minimiser of
absolute error within a leaf.

GradientBoostingClassifier (binary)
-------------------------------------
Minimises binomial deviance. :math:`F_0 = \mathrm{logit}(\bar y)`.
At stage *m*:

.. math::
    p_i = \sigma(F_{m-1}(x_i)), \qquad r_i = y_i - p_i

:math:`h_m` is fit to :math:`r_i` to choose split structure, then each
leaf is replaced by a single Newton-Raphson step (Friedman, 2001):

.. math::
    \gamma_{\text{leaf}} = \frac{\sum_{i \in \text{leaf}} r_i}
                                  {\sum_{i \in \text{leaf}} p_i(1-p_i)}

Predictions: :math:`\sigma(F_M(x)) \ge 0.5 \Rightarrow` positive class.

Design note
-----------
Both models choose tree *structure* with the plain weighted-MSE
criterion (cheap, already implemented by ``DecisionTreeRegressor``)
rather than the more elaborate "Friedman MSE" split-quality score.
Only the leaf *values* use the loss-specific closed-form update. This
is a standard, well-documented simplification that keeps the tree
code shared and dependency-free while still giving each loss its
correct, optimal leaf prediction.

Complexity
----------
- Training : O(n_estimators * n log n * d)
- Inference: O(n_estimators * depth)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import validate_x, validate_xy
from .decision_tree import DecisionTreeRegressor, group_by_leaf

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _sigmoid(z: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _newton_leaf_refit(
    tree: DecisionTreeRegressor, X: FloatArray, numerator: FloatArray, denominator: FloatArray
) -> None:
    """Overwrite each leaf's value with sum(numerator)/sum(denominator)
    over the samples routed to that leaf — the GBM Newton-step update."""
    leaves = tree.apply(X)
    for leaf, idxs in group_by_leaf(leaves).values():
        idx_arr = np.asarray(idxs)
        den = float(denominator[idx_arr].sum())
        leaf.value = float(numerator[idx_arr].sum() / den) if den > _EPS else 0.0


def _median_leaf_refit(tree: DecisionTreeRegressor, X: FloatArray, residual: FloatArray) -> None:
    """Overwrite each leaf's value with the median residual of the
    samples routed there — the LAD-TreeBoost update."""
    leaves = tree.apply(X)
    for leaf, idxs in group_by_leaf(leaves).values():
        leaf.value = float(np.median(residual[np.asarray(idxs)]))


def _check_common_params(n_estimators: int, learning_rate: float, subsample: float) -> None:
    if int(n_estimators) < 1:
        raise ValueError("n_estimators must be >= 1.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if not (0.0 < subsample <= 1.0):
        raise ValueError("subsample must be in (0, 1].")


# ──────────────────────────────────────────────────────────────────────────
# GradientBoostingRegressor
# ──────────────────────────────────────────────────────────────────────────


class GradientBoostingRegressor:
    """Gradient-boosted ensemble of regression trees.

    Parameters
    ----------
    n_estimators : int, default=100
    learning_rate : float, default=0.1
        Shrinkage applied to every tree's contribution.
    max_depth : int, default=3
        Trees are deliberately shallow ("weak learners").
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    subsample : float, default=1.0
        Fraction of rows (sampled without replacement) used to fit
        each tree. ``< 1.0`` gives stochastic gradient boosting.
    loss : str, default='squared_error'
        ``'squared_error'`` or ``'absolute_error'``.
    random_state : int | None, default=None

    Attributes
    ----------
    estimators_ : the fitted sequence of trees
    init_ : the constant initial prediction (mean of y)
    train_score_ : loss value after each boosting stage
    feature_importances_ : mean impurity-decrease importance across trees
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        loss: str = "squared_error",
        random_state: int | None = None,
    ) -> None:
        if loss not in ("squared_error", "absolute_error"):
            raise ValueError("loss must be 'squared_error' or 'absolute_error'.")
        _check_common_params(n_estimators, learning_rate, subsample)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.loss = loss
        self.random_state = random_state

        self.estimators_: list[DecisionTreeRegressor] = []
        self.init_: float | None = None
        self.train_score_: FloatArray | None = None
        self.feature_importances_: FloatArray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> GradientBoostingRegressor:
        X_arr, y_arr = validate_xy(X, y)
        y_arr = y_arr.astype(np.float64)
        n = X_arr.shape[0]
        self.n_features_in_ = X_arr.shape[1]
        rng = np.random.default_rng(self.random_state)

        self.init_ = float(np.mean(y_arr))
        F = np.full(n, self.init_, dtype=np.float64)
        self.estimators_ = []
        self.train_score_ = np.empty(self.n_estimators, dtype=np.float64)

        for m in range(self.n_estimators):
            residual = y_arr - F
            target = residual if self.loss == "squared_error" else np.sign(residual)

            idx = (
                rng.choice(n, size=max(1, int(round(self.subsample * n))), replace=False)
                if self.subsample < 1.0
                else np.arange(n)
            )

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_arr[idx], target[idx])
            if self.loss == "absolute_error":
                _median_leaf_refit(tree, X_arr[idx], residual[idx])

            F = F + self.learning_rate * tree.predict(X_arr)
            self.estimators_.append(tree)
            self.train_score_[m] = float(np.mean((y_arr - F) ** 2))

        self.feature_importances_ = np.mean(
            [t.feature_importances_ for t in self.estimators_], axis=0
        )
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict().")
        X_arr = validate_x(X)
        F = np.full(X_arr.shape[0], self.init_, dtype=np.float64)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X_arr)
        return F

    def staged_predict(self, X: ArrayLike):
        """Yield the running prediction after each boosting stage."""
        if not self.estimators_:
            raise RuntimeError("Call fit() before staged_predict().")
        X_arr = validate_x(X)
        F = np.full(X_arr.shape[0], self.init_, dtype=np.float64)
        for tree in self.estimators_:
            F = F + self.learning_rate * tree.predict(X_arr)
            yield F.copy()

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the coefficient of determination R^2."""
        X_arr, y_arr = validate_xy(X, y)
        preds = self.predict(X_arr)
        ss_res = np.sum((y_arr - preds) ** 2)
        ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > _EPS else 0.0


# ──────────────────────────────────────────────────────────────────────────
# GradientBoostingClassifier (binary)
# ──────────────────────────────────────────────────────────────────────────


class GradientBoostingClassifier:
    """Gradient-boosted ensemble for binary classification (binomial deviance).

    Parameters mirror :class:`GradientBoostingRegressor` minus ``loss``
    (binomial deviance is the only supported objective).

    Attributes
    ----------
    estimators_, init_, train_score_, feature_importances_ — see
    :class:`GradientBoostingRegressor`.
    classes_ : sorted unique labels seen during fit (exactly 2)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        _check_common_params(n_estimators, learning_rate, subsample)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.random_state = random_state

        self.estimators_: list[DecisionTreeRegressor] = []
        self.init_: float | None = None
        self.train_score_: FloatArray | None = None
        self.feature_importances_: FloatArray | None = None
        self.classes_: NDArray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> GradientBoostingClassifier:
        X_arr, y_raw = validate_xy(X, y)
        self.classes_ = np.unique(y_raw)
        if self.classes_.size != 2:
            raise ValueError("GradientBoostingClassifier supports only binary classification.")
        y_bin = (y_raw == self.classes_[1]).astype(np.float64)

        n = X_arr.shape[0]
        self.n_features_in_ = X_arr.shape[1]
        rng = np.random.default_rng(self.random_state)

        p0 = float(np.clip(y_bin.mean(), 1e-6, 1.0 - 1e-6))
        self.init_ = float(np.log(p0 / (1.0 - p0)))
        F = np.full(n, self.init_, dtype=np.float64)
        self.estimators_ = []
        self.train_score_ = np.empty(self.n_estimators, dtype=np.float64)

        for m in range(self.n_estimators):
            p = _sigmoid(F)
            residual = y_bin - p
            denom = p * (1.0 - p)

            idx = (
                rng.choice(n, size=max(1, int(round(self.subsample * n))), replace=False)
                if self.subsample < 1.0
                else np.arange(n)
            )

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_arr[idx], residual[idx])
            _newton_leaf_refit(tree, X_arr[idx], residual[idx], denom[idx])

            F = F + self.learning_rate * tree.predict(X_arr)
            self.estimators_.append(tree)

            p_now = np.clip(_sigmoid(F), 1e-12, 1.0 - 1e-12)
            self.train_score_[m] = float(
                -np.mean(y_bin * np.log(p_now) + (1 - y_bin) * np.log(1 - p_now))
            )

        self.feature_importances_ = np.mean(
            [t.feature_importances_ for t in self.estimators_], axis=0
        )
        return self

    def decision_function(self, X: ArrayLike) -> FloatArray:
        """Return the raw (pre-sigmoid) ensemble score."""
        if not self.estimators_:
            raise RuntimeError("Call fit() before decision_function().")
        X_arr = validate_x(X)
        F = np.full(X_arr.shape[0], self.init_, dtype=np.float64)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X_arr)
        return F

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Return class probabilities, columns ordered as ``classes_``."""
        p1 = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: ArrayLike) -> NDArray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        X_arr, y_arr = validate_xy(X, y)
        return float(np.mean(self.predict(X_arr) == y_arr))
