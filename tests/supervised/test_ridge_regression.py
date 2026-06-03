import numpy as np
import pytest

from mlscratch.supervised.ridge_regression import RidgeRegression


def test_ridge_agrees_with_sklearn():
    """Ridge predictions should agree with sklearn's Ridge within tolerance."""
    from sklearn.linear_model import Ridge as SKRidge

    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 4))
    y = X @ np.array([1.2, -0.8, 0.0, 0.5]) + rng.normal(scale=0.1, size=200)

    model = RidgeRegression(alpha=0.5)
    model.fit(X, y)
    ours = model.predict(X)

    theirs = SKRidge(alpha=0.5, fit_intercept=True, solver="auto", max_iter=10000)
    theirs.fit(X, y)
    np.testing.assert_allclose(ours, theirs.predict(X), rtol=1e-5, atol=1e-2)


def test_ridge_r2_high():
    """Ridge should score >0.99 on clean linear data."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y = X @ np.array([2.0, -1.5]) + rng.normal(scale=0.05, size=300)
    model = RidgeRegression(alpha=0.1)
    model.fit(X, y)
    assert model.score(X, y) > 0.99


def test_ridge_predict_before_fit_raises():
    """predict() before fit() must raise RuntimeError."""
    model = RidgeRegression(alpha=0.5)
    try:
        model.predict(np.ones((5, 2)))
    except RuntimeError as exc:
        assert "fit" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for predict before fit")


def test_ridge_shape_mismatch_raises():
    """Mismatched X and y shapes must raise ValueError."""
    model = RidgeRegression(alpha=0.5)
    with pytest.raises(ValueError, match="samples"):
        model.fit(np.ones((10, 2)), np.ones(5))
