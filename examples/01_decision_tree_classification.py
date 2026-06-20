"""
01_decision_tree_classification.py
====================================
DecisionTreeClassifier on a 3-blob dataset: compares the 'gini' and
'entropy' split criteria, and shows how max_depth trades off training
fit against the held-out test score (the classic overfitting curve).

Run:
    python examples/01_decision_tree_classification.py
"""

from _datasets import make_blobs

from mlscratch.metrics import accuracy_score, classification_report
from mlscratch.preprocessing import train_test_split
from mlscratch.supervised import DecisionTreeClassifier


def main() -> None:
    X, y = make_blobs(
        n_samples=450, centers=((-4, -4), (4, 4), (-4, 4)), cluster_std=2.0, random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0
    )

    print("=== Criterion comparison (max_depth=4) ===")
    for criterion in ("gini", "entropy"):
        model = DecisionTreeClassifier(max_depth=4, criterion=criterion).fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"  {criterion:<8} test accuracy: {acc:.3f}")

    print("\n=== Overfitting curve: deeper trees fit train data, but test score plateaus ===")
    for depth in (1, 2, 3, 5, 8, None):
        model = DecisionTreeClassifier(max_depth=depth).fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        label = "None (unbounded)" if depth is None else str(depth)
        print(f"  max_depth={label:<17} train={train_acc:.3f}  test={test_acc:.3f}")

    print("\n=== Full classification report (max_depth=4, gini) ===")
    model = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))

    print("Per-feature importance:", model.feature_importances_.round(3))


if __name__ == "__main__":
    main()
