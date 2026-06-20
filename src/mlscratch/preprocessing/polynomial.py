r"""
Polynomial Features
=====================
Expand the input features into all polynomial and interaction
combinations up to a given degree. For ``degree=2`` and input
features ``[a, b]``, the output (with bias) is::

    [1, a, b, a^2, a*b, b^2]

Used to let a linear model fit non-linear relationships in the
original feature space.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations, combinations_with_replacement

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class PolynomialFeatures:
    """Generate polynomial and interaction features up to ``degree``.

    Parameters
    ----------
    degree : int, default=2
    include_bias : bool, default=True
        Whether to prepend a constant ``1`` column.
    interaction_only : bool, default=False
        If True, only products of *distinct* input features are
        produced (no ``x_i^2``, ``x_i^3``, ... pure powers).

    Attributes
    ----------
    powers_ : list of tuples
        For each output column, the tuple of input-feature indices
        multiplied together (an empty tuple is the bias column).
    """

    def __init__(
        self, degree: int = 2, include_bias: bool = True, interaction_only: bool = False
    ) -> None:
        if degree < 1:
            raise ValueError("degree must be >= 1.")
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.n_features_in_: int | None = None
        self.powers_: list[tuple[int, ...]] | None = None

    def fit(self, X: ArrayLike) -> PolynomialFeatures:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        self.n_features_in_ = X_arr.shape[1]

        terms: list[tuple[int, ...]] = [()] if self.include_bias else []
        combo_fn = combinations if self.interaction_only else combinations_with_replacement
        for d in range(1, self.degree + 1):
            terms.extend(combo_fn(range(self.n_features_in_), d))
        self.powers_ = terms
        return self

    def transform(self, X: ArrayLike) -> FloatArray:
        if self.powers_ is None:
            raise RuntimeError("Call fit() before transform().")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features but transformer was fit on {self.n_features_in_}."
            )
        out = np.empty((X_arr.shape[0], len(self.powers_)), dtype=np.float64)
        for j, combo in enumerate(self.powers_):
            out[:, j] = 1.0 if not combo else np.prod(X_arr[:, combo], axis=1)
        return out

    def fit_transform(self, X: ArrayLike) -> FloatArray:
        return self.fit(X).transform(X)

    def get_feature_names(self, input_features: list[str] | None = None) -> list[str]:
        """Return human-readable names such as ``"x0 x1^2"`` for each output column."""
        if self.powers_ is None:
            raise RuntimeError("Call fit() before get_feature_names().")
        names_in = (
            input_features
            if input_features is not None
            else [f"x{i}" for i in range(self.n_features_in_)]
        )
        names = []
        for combo in self.powers_:
            if not combo:
                names.append("1")
                continue
            counts = Counter(combo)
            parts = [
                f"{names_in[i]}^{c}" if c > 1 else names_in[i] for i, c in sorted(counts.items())
            ]
            names.append(" ".join(parts))
        return names
