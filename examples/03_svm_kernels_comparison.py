"""
03_svm_kernels_comparison.py
==============================
Kernel SVC on the classic two-moons benchmark: a linear kernel cannot
separate this data, while rbf and poly kernels can. Also shows how
the regularisation strength C trades off margin width against
training error, and demonstrates SVC's transparent one-vs-rest
multiclass support.

Run:
    python examples/03_svm_kernels_comparison.py
"""

from _datasets import make_blobs, make_moons

from mlscratch.preprocessing import StandardScaler, train_test_split
from mlscratch.supervised import SVC


def main() -> None:
    X, y = make_moons(n_samples=300, noise=0.18, random_state=0)
    # SVC's 'scale' gamma already adapts to feature variance, but
    # standardising first is still good practice for kernel methods.
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0
    )

    print("=== Kernel comparison on two interleaving moons ===")
    for kernel in ("linear", "poly", "rbf"):
        model = SVC(C=1.0, kernel=kernel, gamma="scale", max_iter=30, random_state=0).fit(
            X_train, y_train
        )
        acc = model.score(X_test, y_test)
        print(
            f"  kernel={kernel:<7}  test accuracy: {acc:.3f}  n_support_vectors: {model.n_support_}"
        )

    print("\n=== Effect of C (regularisation strength) on an RBF SVM ===")
    for C in (0.01, 0.1, 1.0, 10.0, 100.0):
        model = SVC(C=C, kernel="rbf", gamma="scale", max_iter=30, random_state=0).fit(
            X_train, y_train
        )
        acc = model.score(X_test, y_test)
        print(f"  C={C:<6}  test accuracy: {acc:.3f}  n_support_vectors: {model.n_support_}")

    print("\n=== Multiclass via transparent one-vs-rest ===")
    X3, y3 = make_blobs(
        n_samples=240, centers=((-4, -4), (4, 4), (-4, 4)), cluster_std=1.5, random_state=2
    )
    X3 = StandardScaler().fit_transform(X3)
    X3_train, X3_test, y3_train, y3_test = train_test_split(
        X3, y3, test_size=0.25, stratify=y3, random_state=0
    )
    multi_model = SVC(kernel="rbf", gamma="scale", max_iter=15, random_state=0).fit(
        X3_train, y3_train
    )
    print(f"  3-class accuracy: {multi_model.score(X3_test, y3_test):.3f}")
    print(
        f"  multiclass_: {multi_model.multiclass_}  (one binary SVM trained per class: {len(multi_model.classes_)})"
    )


if __name__ == "__main__":
    main()
