import numpy as np
import pytest

from mlscratch.supervised.linear_regression import (
    GradientDescentRegressor,
    OrdinaryLeastSquares,
)


def test_ols_coef_recovery(linear_dataset):
    """OLS must recover true coefficients [2.0, 3.0] to within 0.1."""
    X, y = linear_dataset
    model = OrdinaryLeastSquares()
    model.fit(X, y)
    np.testing.assert_allclose(model.coef_, [2.0, 3.0], atol=0.1)


def test_ols_agrees_with_sklearn(linear_dataset):
    """OLS predictions must match sklearn LinearRegression within 1e-6."""
    from sklearn.linear_model import LinearRegression as SKLearn

    X, y = linear_dataset
    ours = OrdinaryLeastSquares().fit(X, y)
    theirs = SKLearn().fit(X, y)
    np.testing.assert_allclose(ours.predict(X), theirs.predict(X), rtol=1e-6)


def test_ols_r2_high(linear_dataset):
    """R² on clean linear data must be > 0.99."""
    X, y = linear_dataset
    r2 = OrdinaryLeastSquares().fit(X, y).score(X, y)
    assert r2 > 0.99


def test_gradient_descent_converges(linear_dataset):
    """GD loss must decrease monotonically for 50+ consecutive epochs."""
    X, y = linear_dataset
    model = GradientDescentRegressor(n_epochs=200, learning_rate=0.01)
    model.fit(X, y)
    losses = model.loss_history_
    diffs = np.diff(losses[-50:])
    assert (diffs <= 1e-6).all(), "Loss not monotonically decreasing"


def test_gd_agrees_with_ols(linear_dataset):
    """After sufficient epochs, GD predictions approximate OLS within 5%."""
    X, y = linear_dataset
    ols_pred = OrdinaryLeastSquares().fit(X, y).predict(X)
    gd_pred = GradientDescentRegressor(n_epochs=5000, learning_rate=0.005).fit(X, y).predict(X)
    np.testing.assert_allclose(gd_pred, ols_pred, rtol=0.05, atol=1e-2)


def test_shape_mismatch_raises():
    """Mismatched X and y shapes must raise ValueError."""
    model = OrdinaryLeastSquares()
    with pytest.raises(ValueError, match="samples"):
        model.fit(np.ones((10, 2)), np.ones(5))


def test_predict_before_fit_raises():
    """predict() before fit() must raise RuntimeError."""
    with pytest.raises(RuntimeError, match="fit"):
        OrdinaryLeastSquares().predict(np.ones((5, 2)))
