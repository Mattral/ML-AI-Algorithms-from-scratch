import numpy as np
import pytest

from mlscratch.supervised.decision_tree import DecisionTreeClassifier


def test_decision_tree_accuracy_on_iris():
    """Decision tree should classify Iris with ≥95% accuracy."""
    from sklearn.datasets import load_iris

    data = load_iris()
    X, y = data.data, data.target
    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X, y)
    assert model.score(X, y) >= 0.95


def test_decision_tree_agrees_with_sklearn():
    """Decision tree predictions should match sklearn DecisionTreeClassifier within 2%."""
    from sklearn.datasets import make_classification
    from sklearn.tree import DecisionTreeClassifier as SKDecisionTree

    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X, y)
    ours = model.predict(X)

    theirs = SKDecisionTree(max_depth=5, random_state=42)
    theirs.fit(X, y)
    assert np.mean(ours == theirs.predict(X)) >= 0.98


def test_decision_tree_predict_before_fit_raises():
    """predict() before fit() must raise RuntimeError."""
    model = DecisionTreeClassifier(max_depth=3)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.ones((5, 2)))


def test_decision_tree_shape_mismatch_raises():
    """Mismatched X and y shapes must raise ValueError."""
    model = DecisionTreeClassifier(max_depth=3)
    with pytest.raises(ValueError, match="samples"):
        model.fit(np.ones((10, 2)), np.ones(5))
