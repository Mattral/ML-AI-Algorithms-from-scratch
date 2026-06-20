"""
05_adaboost_multiclass.py
============================
AdaBoostClassifier on 3-class blob data: compares the discrete
'SAMME' algorithm against the real-valued 'SAMME.R' (which typically
converges in fewer rounds since it uses each stump's full probability
estimate, not just its hard prediction), and tracks the ensemble's
accuracy round-by-round via staged_predict.

Run:
    python examples/05_adaboost_multiclass.py
"""

from _datasets import make_blobs

from mlscratch.metrics import accuracy_score
from mlscratch.preprocessing import train_test_split
from mlscratch.supervised import AdaBoostClassifier


def main() -> None:
    X, y = make_blobs(
        n_samples=400, centers=((-4, -4), (4, 4), (0, 5)), cluster_std=2.2, random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0
    )

    print("=== SAMME vs SAMME.R, same number of decision-stump rounds ===")
    for algorithm in ("SAMME", "SAMME.R"):
        model = AdaBoostClassifier(
            n_estimators=60, algorithm=algorithm, max_depth=1, random_state=0
        )
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(
            f"  {algorithm:<8} test accuracy: {acc:.3f}  (stumps actually used: {len(model.estimators_)})"
        )

    print("\n=== Round-by-round accuracy via staged_predict ('SAMME.R') ===")
    model = AdaBoostClassifier(n_estimators=40, algorithm="SAMME.R", max_depth=1, random_state=0)
    model.fit(X_train, y_train)
    for round_idx, staged_pred in enumerate(model.staged_predict(X_test), start=1):
        if round_idx in (1, 2, 5, 10, 20, 40):
            print(
                f"  after {round_idx:<3} rounds: accuracy = {accuracy_score(y_test, staged_pred):.3f}"
            )

    print("\n=== Deeper weak learners need fewer boosting rounds ===")
    for max_depth in (1, 2, 3):
        model = AdaBoostClassifier(
            n_estimators=20, algorithm="SAMME.R", max_depth=max_depth, random_state=0
        )
        model.fit(X_train, y_train)
        print(f"  max_depth={max_depth}  test accuracy: {model.score(X_test, y_test):.3f}")


if __name__ == "__main__":
    main()
