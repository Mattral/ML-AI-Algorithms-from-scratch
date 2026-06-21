r"""
AdaBoost
========
Adaptive Boosting: a weighted ensemble of shallow decision-tree "weak
learners" (stumps by default), where each successive learner is fit
on a re-weighted version of the training set that emphasises the
samples the ensemble-so-far gets wrong.

Two algorithms are supported, both natively multiclass
(Zhu, Rosset, Zhu & Hastie, 2009, "Multi-class AdaBoost"):

``'SAMME'`` (discrete)
-----------------------
Uses only each weak learner's hard predictions.

.. math::
    \alpha_m = \eta \left[ \ln\frac{1-\mathrm{err}_m}{\mathrm{err}_m} + \ln(K-1) \right]

.. math::
    w_i \leftarrow w_i \exp\!\big(\alpha_m \cdot \mathbb{1}[\hat y_i \ne y_i]\big),
    \quad \text{then renormalise}

``'SAMME.R'`` (real-valued, the modern default)
--------------------------------------------------
Uses each weak learner's class-probability estimates directly, which
typically converges in fewer rounds:

.. math::
    h_k^{(m)}(x) = (K-1)\left(\log p_k(x) - \frac1K\sum_{k'} \log p_{k'}(x)\right)

.. math::
    w_i \leftarrow w_i \exp\!\left(-\eta\,\frac{K-1}{K}\, y_i^{\mathsf T} \log p(x_i)\right)

where :math:`y_i` uses the symmetric :math:`\{-1/(K-1), 1\}` class
coding. The ensemble decision is :math:`\arg\max_k \sum_m h_k^{(m)}(x)`.

For :math:`K=2` both algorithms reduce to the classic binary AdaBoost.

Complexity
----------
- Training : O(n_estimators * weak-learner fit cost)
- Inference: O(n_estimators * weak-learner predict cost)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import validate_x, validate_xy
from .decision_tree import DecisionTreeClassifier

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_EPS = 1e-10


class AdaBoostClassifier:
    """Adaptive Boosting classifier (SAMME / SAMME.R), natively multiclass.

    Parameters
    ----------
    n_estimators : int, default=50
        Maximum number of weak learners. Boosting may stop earlier if
        a weak learner achieves zero training error or (SAMME only)
        becomes worse than random guessing.
    learning_rate : float, default=1.0
        Shrinks the contribution of each weak learner.
    algorithm : str, default='SAMME.R'
        ``'SAMME'`` (discrete) or ``'SAMME.R'`` (real-valued).
    max_depth : int, default=1
        Depth of each weak learner; ``1`` gives the classic "decision
        stump".
    random_state : int | None, default=None

    Attributes
    ----------
    estimators_ : the fitted sequence of weak learners
    estimator_weights_ : per-estimator combination weight (alpha)
    estimator_errors_ : per-estimator weighted training error
    classes_ : sorted unique labels seen during fit
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        algorithm: str = "SAMME.R",
        max_depth: int = 1,
        random_state: int | None = None,
    ) -> None:
        if algorithm not in ("SAMME", "SAMME.R"):
            raise ValueError("algorithm must be 'SAMME' or 'SAMME.R'.")
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be >= 1.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.algorithm = algorithm
        self.max_depth = int(max_depth)
        self.random_state = random_state

        self.estimators_: list[DecisionTreeClassifier] = []
        self.estimator_weights_: FloatArray | None = None
        self.estimator_errors_: FloatArray | None = None
        self.classes_: IntArray | None = None
        self.n_features_in_: int | None = None

    # -- public API -----------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> AdaBoostClassifier:
        X_arr, y_raw = validate_xy(X, y)
        self.classes_, y_idx = np.unique(y_raw, return_inverse=True)
        y_idx = y_idx.astype(np.int64)
        n_classes = self.classes_.size
        if n_classes < 2:
            raise ValueError("AdaBoostClassifier requires at least 2 classes.")
        n = X_arr.shape[0]
        self.n_features_in_ = X_arr.shape[1]
        rng = np.random.default_rng(self.random_state)

        sample_weight = np.full(n, 1.0 / n, dtype=np.float64)
        self.estimators_ = []
        weights: list[float] = []
        errors: list[float] = []

        for m in range(self.n_estimators):
            stump = DecisionTreeClassifier(
                max_depth=self.max_depth, random_state=int(rng.integers(0, 2**31 - 1))
            )
            stump.fit(X_arr, y_idx, sample_weight=sample_weight)

            if self.algorithm == "SAMME":
                pred = stump.predict(X_arr)
                incorrect = pred != y_idx
                err = float(np.average(incorrect, weights=sample_weight))

                if err >= 1.0 - 1.0 / n_classes:
                    if not self.estimators_:
                        raise RuntimeError(
                            "BaseEstimator is worse than random guessing on the first "
                            "boosting round; AdaBoost cannot be fit."
                        )
                    break

                err_clipped = float(np.clip(err, _EPS, 1.0 - _EPS))
                alpha = self.learning_rate * (
                    np.log((1.0 - err_clipped) / err_clipped) + np.log(n_classes - 1)
                )

                self.estimators_.append(stump)
                weights.append(float(alpha))
                errors.append(err)

                if err <= 0.0 or m == self.n_estimators - 1:
                    break
                sample_weight = sample_weight * np.exp(alpha * incorrect)

            else:  # SAMME.R
                proba = self._safe_proba(stump, X_arr, n_classes)
                logp = np.log(proba)
                pred = np.argmax(proba, axis=1)
                incorrect = pred != y_idx
                err = float(np.average(incorrect, weights=sample_weight))

                self.estimators_.append(stump)
                weights.append(1.0)  # SAMME.R combines via h(x) directly, not a scalar alpha
                errors.append(err)

                if err <= 0.0 or m == self.n_estimators - 1:
                    break

                y_coding = np.full((n, n_classes), -1.0 / (n_classes - 1))
                y_coding[np.arange(n), y_idx] = 1.0
                contrib = (
                    -self.learning_rate
                    * (n_classes - 1)
                    / n_classes
                    * np.sum(y_coding * logp, axis=1)
                )
                sample_weight = sample_weight * np.exp(contrib)

            sample_weight = np.maximum(sample_weight, _EPS)
            sample_weight /= sample_weight.sum()

        self.estimator_weights_ = np.array(weights, dtype=np.float64)
        self.estimator_errors_ = np.array(errors, dtype=np.float64)
        return self

    def decision_function(self, X: ArrayLike) -> FloatArray:
        """Return the per-class ensemble score, shape ``(n_samples, n_classes)``."""
        if not self.estimators_:
            raise RuntimeError("Call fit() before decision_function().")
        X_arr = validate_x(X)
        n_classes = self.classes_.size
        scores = np.zeros((X_arr.shape[0], n_classes), dtype=np.float64)

        if self.algorithm == "SAMME":
            for stump, alpha in zip(self.estimators_, self.estimator_weights_, strict=True):
                pred = stump.predict(X_arr)
                scores[np.arange(X_arr.shape[0]), pred] += alpha
        else:
            for stump in self.estimators_:
                proba = self._safe_proba(stump, X_arr, n_classes)
                logp = np.log(proba)
                scores += (n_classes - 1) * (logp - logp.mean(axis=1, keepdims=True))
        return scores

    def predict(self, X: ArrayLike) -> NDArray:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Softmax of the (rescaled) ensemble decision scores."""
        scores = self.decision_function(X)
        n_classes = self.classes_.size
        scaled = scores / max(1, n_classes - 1)
        e = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def staged_predict(self, X: ArrayLike):
        """Yield the ensemble's predicted labels after each boosting round."""
        if not self.estimators_:
            raise RuntimeError("Call fit() before staged_predict().")
        X_arr = validate_x(X)
        n_classes = self.classes_.size
        cum_scores = np.zeros((X_arr.shape[0], n_classes), dtype=np.float64)
        for i, stump in enumerate(self.estimators_):
            if self.algorithm == "SAMME":
                pred = stump.predict(X_arr)
                cum_scores[np.arange(X_arr.shape[0]), pred] += self.estimator_weights_[i]
            else:
                proba = self._safe_proba(stump, X_arr, n_classes)
                logp = np.log(proba)
                cum_scores += (n_classes - 1) * (logp - logp.mean(axis=1, keepdims=True))
            yield self.classes_[np.argmax(cum_scores, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        X_arr, y_arr = validate_xy(X, y)
        return float(np.mean(self.predict(X_arr) == y_arr))

    # -- internals --------------------------------------------------------------

    @staticmethod
    def _safe_proba(stump: DecisionTreeClassifier, X: FloatArray, n_classes: int) -> FloatArray:
        """Map a stump's predict_proba (over its own classes_ subset, which may
        be missing classes absent from its bootstrap/weighted sample) into a
        full, strictly-positive (n_samples, n_classes) probability matrix."""
        p_sub = stump.predict_proba(X)
        proba = np.full((X.shape[0], n_classes), _EPS)
        proba[:, stump.classes_] = np.maximum(p_sub, _EPS)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba
