r"""
Train/Test Splitting
======================
A minimal, dependency-free re-implementation of the classic
``train_test_split`` utility, with optional stratification.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def train_test_split(
    *arrays: ArrayLike,
    test_size: float | int = 0.25,
    train_size: float | int | None = None,
    shuffle: bool = True,
    stratify: ArrayLike | None = None,
    random_state: int | None = None,
) -> list[np.ndarray]:
    """Split one or more array-likes into random train/test subsets.

    Parameters
    ----------
    *arrays : array-like
        One or more arrays, all sharing the same length along axis 0
        (e.g. ``X, y``).
    test_size : float | int, default=0.25
        Fraction in ``(0, 1)`` or an absolute sample count for the test split.
    train_size : float | int | None, default=None
        Complementary to ``test_size`` if given; otherwise inferred as
        everything not in the test split.
    shuffle : bool, default=True
        Whether to shuffle before splitting (ignored if ``stratify`` is given,
        which always shuffles within each class).
    stratify : array-like | None, default=None
        If given, the split preserves this array's class proportions.
    random_state : int | None, default=None

    Returns
    -------
    A flat list ``[arr1_train, arr1_test, arr2_train, arr2_test, ...]``,
    in the same order as the inputs (mirrors scikit-learn's signature).
    """
    if len(arrays) == 0:
        raise ValueError("At least one array is required.")
    arrays_np = [np.asarray(a) for a in arrays]
    n = arrays_np[0].shape[0]
    for a in arrays_np[1:]:
        if a.shape[0] != n:
            raise ValueError("All input arrays must have the same first dimension.")

    n_test = _resolve_size(test_size, n, "test_size")
    n_train = _resolve_size(train_size, n, "train_size") if train_size is not None else n - n_test
    if n_train + n_test > n:
        raise ValueError("train_size + test_size exceeds the number of available samples.")
    if n_train <= 0 or n_test <= 0:
        raise ValueError("Both the train and test splits must contain at least one sample.")

    rng = np.random.default_rng(random_state)

    if stratify is not None:
        strat = np.asarray(stratify).flatten()
        if strat.shape[0] != n:
            raise ValueError("stratify must have the same length as the input arrays.")
        train_idx, test_idx = _stratified_indices(strat, n_train, n_test, rng)
    else:
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)
        test_idx = idx[:n_test]
        train_idx = idx[n_test : n_test + n_train]

    result: list[np.ndarray] = []
    for a in arrays_np:
        result.append(a[train_idx])
        result.append(a[test_idx])
    return result


def _resolve_size(size: float | int, n: int, name: str) -> int:
    if isinstance(size, float):
        if not (0.0 < size < 1.0):
            raise ValueError(f"{name} as a float must be in (0, 1).")
        return int(np.ceil(size * n))
    if isinstance(size, (int, np.integer)):
        if not (0 < size <= n):
            raise ValueError(f"{name} as an int must be in (0, {n}].")
        return int(size)
    raise ValueError(f"{name} must be a float or an int.")


def _stratified_indices(
    strat: np.ndarray, n_train: int, n_test: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices class-by-class so train/test class proportions
    approximately match the proportions of ``strat`` as a whole."""
    classes, y_idx = np.unique(strat, return_inverse=True)
    n = strat.shape[0]
    train_parts, test_parts = [], []
    for k in range(classes.size):
        cls_idx = np.flatnonzero(y_idx == k)
        rng.shuffle(cls_idx)
        cls_n_test = min(cls_idx.size, max(1, int(round(n_test * cls_idx.size / n))))
        test_parts.append(cls_idx[:cls_n_test])
        train_parts.append(cls_idx[cls_n_test:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    # Trim to the exact requested sizes where the per-class rounding overshot.
    if train_idx.size > n_train:
        train_idx = train_idx[:n_train]
    if test_idx.size > n_test:
        test_idx = test_idx[:n_test]
    return train_idx, test_idx
