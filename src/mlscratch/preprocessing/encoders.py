r"""
Categorical Encoders
======================
LabelEncoder
-------------
Maps a 1-D array of arbitrary hashable labels to integer codes
``0..K-1`` (sorted) — typically used to encode a target column.

OneHotEncoder
--------------
Maps each column of a 2-D categorical array independently to a block
of binary indicator columns, one per observed category.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class LabelEncoder:
    """Encode a single 1-D array of labels as integers ``0..K-1``.

    Attributes
    ----------
    classes_ : sorted unique labels seen during fit
    """

    def __init__(self) -> None:
        self.classes_: NDArray | None = None

    def fit(self, y: ArrayLike) -> LabelEncoder:
        y_arr = np.asarray(y).flatten()
        if y_arr.shape[0] == 0:
            raise ValueError("y must not be empty.")
        self.classes_ = np.unique(y_arr)
        return self

    def transform(self, y: ArrayLike) -> NDArray[np.int64]:
        if self.classes_ is None:
            raise RuntimeError("Call fit() before transform().")
        y_arr = np.asarray(y).flatten()
        label_to_idx = {label: i for i, label in enumerate(self.classes_)}
        unseen = sorted(set(np.unique(y_arr).tolist()) - set(self.classes_.tolist()))
        if unseen:
            raise ValueError(f"y contains previously unseen labels: {unseen}")
        return np.array([label_to_idx[v] for v in y_arr], dtype=np.int64)

    def fit_transform(self, y: ArrayLike) -> NDArray[np.int64]:
        return self.fit(y).transform(y)

    def inverse_transform(self, y: ArrayLike) -> NDArray:
        if self.classes_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        y_arr = np.asarray(y, dtype=np.int64).flatten()
        if np.any((y_arr < 0) | (y_arr >= self.classes_.size)):
            raise ValueError("y contains codes outside the range of known classes.")
        return self.classes_[y_arr]


class OneHotEncoder:
    """One-hot encode every column of a 2-D categorical array independently.

    Parameters
    ----------
    drop_first : bool, default=False
        If True, drop the first category of each column to avoid the
        "dummy variable trap" (perfect multicollinearity for linear models).
    handle_unknown : str, default='error'
        ``'error'`` raises on categories not seen during fit;
        ``'ignore'`` encodes them as an all-zero row.

    Attributes
    ----------
    categories_ : list of per-column arrays of observed categories
    """

    def __init__(self, drop_first: bool = False, handle_unknown: str = "error") -> None:
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'.")
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown
        self.categories_: list[NDArray] | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike) -> OneHotEncoder:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        self.n_features_in_ = X_arr.shape[1]
        self.categories_ = [np.unique(X_arr[:, j]) for j in range(X_arr.shape[1])]
        return self

    def transform(self, X: ArrayLike) -> FloatArray:
        if self.categories_ is None:
            raise RuntimeError("Call fit() before transform().")
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} columns but encoder was fit on {self.n_features_in_}."
            )

        blocks = []
        for j, cats in enumerate(self.categories_):
            start = 1 if self.drop_first else 0
            col = X_arr[:, j]
            cat_to_idx = {c: i for i, c in enumerate(cats)}
            block = np.zeros((X_arr.shape[0], cats.size), dtype=np.float64)
            for row, value in enumerate(col):
                idx = cat_to_idx.get(value)
                if idx is None:
                    if self.handle_unknown == "error":
                        raise ValueError(f"Unknown category {value!r} in column {j}.")
                    continue
                block[row, idx] = 1.0
            blocks.append(block[:, start:])
        return np.hstack(blocks)

    def fit_transform(self, X: ArrayLike) -> FloatArray:
        return self.fit(X).transform(X)

    def get_feature_names(self, input_features: list[str] | None = None) -> list[str]:
        """Return human-readable ``"<feature>_<category>"`` output column names."""
        if self.categories_ is None:
            raise RuntimeError("Call fit() before get_feature_names().")
        prefixes = (
            input_features
            if input_features is not None
            else [f"x{i}" for i in range(self.n_features_in_)]
        )
        names = []
        for prefix, cats in zip(prefixes, self.categories_, strict=True):
            start = 1 if self.drop_first else 0
            for cat in cats[start:]:
                names.append(f"{prefix}_{cat}")
        return names
