import numpy as np
import pytest

from mlscratch.supervised.naive_bayes import GaussianNB


def test_gaussian_nb_agrees_with_sklearn():
    """GaussianNB predictions should match sklearn's GaussianNB on the same data."""
    from sklearn.datasets import make_classification
    from sklearn.naive_bayes import GaussianNB as SKGaussianNB

    X, y = make_classification(
        n_samples=250,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=0,
    )

    model = GaussianNB().fit(X, y)
    theirs = SKGaussianNB().fit(X, y)

    assert model.score(X, y) == pytest.approx(theirs.score(X, y), rel=1e-6)
    assert np.mean(model.predict(X) == theirs.predict(X)) == pytest.approx(1.0, rel=1e-6)


def test_gaussian_nb_predict_before_fit_raises():
    """Predict before fit must raise RuntimeError."""
    model = GaussianNB()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.ones((5, 2)))


def test_gaussian_nb_invalid_input_raises():
    """Mismatched X and y shapes must raise ValueError."""
    model = GaussianNB()
    with pytest.raises(ValueError, match="samples"):
        model.fit(np.ones((10, 2)), np.ones(5))
