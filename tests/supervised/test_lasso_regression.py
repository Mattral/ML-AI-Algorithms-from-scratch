import numpy as np
import pytest

from mlscratch.supervised.lasso_regression import LassoRegression


def test_lasso_agrees_with_sklearn():
    """Lasso predictions should agree with sklearn's Lasso within tolerance."""
    from sklearn.linear_model import Lasso as SKLasso

    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 5))
    true_coef = np.array([1.5, 0.0, -2.0, 0.0, 0.5])
    y = X @ true_coef + rng.normal(scale=0.1, size=200)

    model = LassoRegression(alpha=0.1, max_iter=2000, tol=1e-5)
    model.fit(X, y)
    ours = model.predict(X)

    theirs = SKLasso(alpha=0.1, fit_intercept=True, max_iter=10000, tol=1e-8)
    theirs.fit(X, y)
    np.testing.assert_allclose(ours, theirs.predict(X), rtol=1e-3, atol=1e-2)


def test_lasso_r2_high():
    """Lasso should fit clean data with R² above 0.95."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([2.0, -1.0, 0.0]) + rng.normal(scale=0.05, size=200)
    model = LassoRegression(alpha=0.05, max_iter=2000, tol=1e-5)
    model.fit(X, y)
    assert model.score(X, y) > 0.95


def test_lasso_predict_before_fit_raises():
    """predict() before fit() must raise RuntimeError."""
    model = LassoRegression(alpha=0.1)
    try:
        model.predict(np.ones((5, 2)))
    except RuntimeError as exc:
        assert "fit" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for predict before fit")


def test_lasso_shape_mismatch_raises():
    """Mismatched X and y shapes must raise ValueError."""
    model = LassoRegression(alpha=0.1)
    with pytest.raises(ValueError, match="samples"):
        model.fit(np.ones((10, 2)), np.ones(5))
