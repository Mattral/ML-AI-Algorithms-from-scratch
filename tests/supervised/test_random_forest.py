import numpy as np
import pytest

from mlscratch.supervised.random_forest import RandomForestClassifier


def test_random_forest_accuracy_on_clean_data():
    """Random forest must fit clean binary classification data accurately."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        class_sep=2.0,
        random_state=42,
    )

    model = RandomForestClassifier(n_estimators=25, random_state=42).fit(X, y)
    assert model.score(X, y) >= 0.95


def test_random_forest_agrees_with_sklearn():
    """Predictions should agree with sklearn's RandomForestClassifier on the same data."""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier as SKRF

    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        class_sep=1.5,
        random_state=0,
    )

    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    ours = model.predict(X)

    theirs = SKRF(n_estimators=20, random_state=0).fit(X, y)
    assert np.mean(ours == theirs.predict(X)) >= 0.85


def test_random_forest_predict_before_fit_raises():
    """predict() must raise RuntimeError if fit() was not called."""
    model = RandomForestClassifier(n_estimators=10)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.ones((5, 3)))


def test_random_forest_invalid_input_raises():
    """Mismatched X and y shapes must raise ValueError."""
    model = RandomForestClassifier(n_estimators=5)
    with pytest.raises(ValueError, match="samples"):
        model.fit(np.ones((10, 2)), np.ones(5))
