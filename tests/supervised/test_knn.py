import numpy as np
import pytest

from mlscratch.supervised.knn import KNeighborsClassifier


def test_knn_accuracy_on_separable_data():
    """KNN must achieve ≥95% accuracy on a clean binary classification task."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        class_sep=2.0,
        random_state=42,
    )
    model = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    assert model.score(X, y) >= 0.95


def test_knn_agrees_with_sklearn():
    """KNN labels should match sklearn's KNeighborsClassifier within 2%."""
    from sklearn.datasets import make_classification
    from sklearn.neighbors import KNeighborsClassifier as SKKNN

    X, y = make_classification(
        n_samples=200,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        class_sep=1.5,
        random_state=0,
    )
    model = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    ours = model.predict(X)

    theirs = SKKNN(n_neighbors=5).fit(X, y)
    assert np.mean(ours == theirs.predict(X)) >= 0.98


def test_knn_predict_before_fit_raises():
    """predict() before fit() must raise RuntimeError."""
    model = KNeighborsClassifier(n_neighbors=3)
    try:
        model.predict(np.ones((5, 2)))
    except RuntimeError as exc:
        assert "fit" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for predict before fit")


def test_knn_invalid_input_raises():
    """Mismatched X and y shapes must raise ValueError."""
    model = KNeighborsClassifier(n_neighbors=3)
    with pytest.raises(ValueError, match="samples"):
        model.fit(np.ones((10, 2)), np.ones(5))
