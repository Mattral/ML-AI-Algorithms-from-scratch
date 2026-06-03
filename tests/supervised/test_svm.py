import numpy as np
import pytest

from mlscratch.supervised.svm import LinearSVMClassifier


def test_linear_svm_accuracy_on_separable_data():
    """LinearSVMClassifier should fit a clean binary classification task."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=250,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        class_sep=2.0,
        random_state=42,
    )

    model = LinearSVMClassifier(learning_rate=1e-3, n_epochs=1000, C=1.0, random_state=42).fit(X, y)
    assert model.score(X, y) >= 0.90


def test_linear_svm_agrees_with_sklearn_hinge():
    """Linear SVM should agree with sklearn's hinge-loss SGDClassifier."""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import SGDClassifier

    X, y = make_classification(
        n_samples=220,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        class_sep=1.8,
        random_state=1,
    )

    model = LinearSVMClassifier(learning_rate=1e-3, n_epochs=1000, C=1.0, random_state=1).fit(X, y)
    theirs = SGDClassifier(
        loss="hinge",
        alpha=1.0 / (model.C * X.shape[0]),
        learning_rate="constant",
        eta0=1e-3,
        max_iter=1000,
        tol=None,
        random_state=1,
    ).fit(X, y)
    assert np.mean(model.predict(X) == theirs.predict(X)) >= 0.85


def test_linear_svm_predict_before_fit_raises():
    """Predict before fit must raise RuntimeError."""
    model = LinearSVMClassifier()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.ones((5, 2)))


def test_linear_svm_invalid_input_raises():
    """Mismatched X and y shapes must raise ValueError."""
    model = LinearSVMClassifier()
    with pytest.raises(ValueError, match="samples"):
        model.fit(np.ones((10, 2)), np.ones(5))
