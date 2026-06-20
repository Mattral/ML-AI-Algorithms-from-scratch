"""
02_random_forest_vs_single_tree.py
====================================
RandomForestClassifier on noisy, partly-irrelevant-feature blob data:
shows how bagging many deep (individually overfit-prone) trees, each
restricted to a random feature subset, generalises better than any
single deep tree — and demonstrates out-of-bag (OOB) scoring as a
"free" validation estimate that needs no held-out split.

Run:
    python examples/02_random_forest_vs_single_tree.py
"""

import numpy as np
from _datasets import make_blobs

from mlscratch.preprocessing import train_test_split
from mlscratch.supervised import DecisionTreeClassifier, RandomForestClassifier


def add_label_noise(y: np.ndarray, fraction: float, random_state: int) -> np.ndarray:
    """Flip a fraction of labels at random to simulate a noisy dataset."""
    rng = np.random.default_rng(random_state)
    y_noisy = y.copy()
    n_flip = int(fraction * len(y))
    flip_idx = rng.choice(len(y), n_flip, replace=False)
    classes = np.unique(y)
    for i in flip_idx:
        y_noisy[i] = rng.choice(classes[classes != y_noisy[i]])
    return y_noisy


def add_noise_features(X: np.ndarray, n_noise: int, random_state: int) -> np.ndarray:
    """Append irrelevant random columns — real datasets rarely have only
    informative features, and this is where bagging + feature subsampling
    genuinely pays off versus a single tree."""
    rng = np.random.default_rng(random_state)
    noise = rng.normal(size=(X.shape[0], n_noise))
    return np.hstack([X, noise])


def main() -> None:
    X, y = make_blobs(
        n_samples=500, centers=((-3, -3), (3, 3), (-3, 3), (3, -3)), cluster_std=2.5, random_state=1
    )
    y = add_label_noise(y, fraction=0.08, random_state=1)
    X = add_noise_features(X, n_noise=6, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0
    )

    print("=== Single (deep, overfit-prone) tree vs forest of the same depth ===")
    tree = DecisionTreeClassifier(max_depth=8).fit(X_train, y_train)
    print(f"  single tree   test accuracy: {tree.score(X_test, y_test):.3f}")

    for n_estimators in (1, 5, 25, 100):
        forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=8, random_state=0).fit(
            X_train, y_train
        )
        print(f"  forest(n={n_estimators:<3})  test accuracy: {forest.score(X_test, y_test):.3f}")

    print("\n=== Out-of-bag score approximates the held-out test score ===")
    forest = RandomForestClassifier(
        n_estimators=150, max_depth=8, oob_score=True, random_state=0
    ).fit(X_train, y_train)
    print(f"  OOB score:  {forest.oob_score_:.3f}")
    print(f"  test score: {forest.score(X_test, y_test):.3f}")

    print("\n=== Feature importances (forest vs single tree) ===")
    print("  (features 0-1 are informative, 2-7 are pure noise)")
    print(f"  tree:   {tree.feature_importances_.round(3)}")
    print(f"  forest: {forest.feature_importances_.round(3)}")


if __name__ == "__main__":
    main()
