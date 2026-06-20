"""
Tests for mlscratch.metrics.regression
"""

import numpy as np
import pytest

from mlscratch.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


@pytest.fixture
def regression_predictions():
    rng = np.random.default_rng(0)
    y_true = rng.normal(10, 3, 100)
    y_pred = y_true + rng.normal(0, 1, 100)
    return y_true, y_pred


class TestMeanSquaredError:
    def test_zero_for_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error(y, y) == 0.0

    def test_matches_manual_computation(self, regression_predictions):
        y_true, y_pred = regression_predictions
        expected = np.mean((y_true - y_pred) ** 2)
        assert mean_squared_error(y_true, y_pred) == pytest.approx(expected)

    def test_agrees_with_sklearn(self, regression_predictions):
        from sklearn.metrics import mean_squared_error as sk_mse

        y_true, y_pred = regression_predictions
        assert mean_squared_error(y_true, y_pred) == pytest.approx(sk_mse(y_true, y_pred))

    def test_squared_false_gives_rmse(self, regression_predictions):
        y_true, y_pred = regression_predictions
        mse = mean_squared_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        assert rmse == pytest.approx(np.sqrt(mse))

    def test_root_mean_squared_error_helper(self, regression_predictions):
        y_true, y_pred = regression_predictions
        assert root_mean_squared_error(y_true, y_pred) == mean_squared_error(
            y_true, y_pred, squared=False
        )

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="samples"):
            mean_squared_error(np.ones(5), np.ones(3))


class TestMeanAbsoluteError:
    def test_zero_for_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_absolute_error(y, y) == 0.0

    def test_agrees_with_sklearn(self, regression_predictions):
        from sklearn.metrics import mean_absolute_error as sk_mae

        y_true, y_pred = regression_predictions
        assert mean_absolute_error(y_true, y_pred) == pytest.approx(sk_mae(y_true, y_pred))

    def test_less_sensitive_to_outliers_than_mse(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 0.0])
        # the single huge error dominates MSE far more than MAE (relative to scale)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        assert mse / mae > 10


class TestMeanAbsolutePercentageError:
    def test_agrees_with_sklearn(self, regression_predictions):
        from sklearn.metrics import mean_absolute_percentage_error as sk_mape

        y_true, y_pred = regression_predictions
        assert mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(
            sk_mape(y_true, y_pred), abs=1e-6
        )

    def test_zero_for_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_absolute_percentage_error(y, y) == 0.0

    def test_handles_near_zero_true_values_without_error(self):
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.1, 1.1, 2.1])
        result = mean_absolute_percentage_error(y_true, y_pred)
        assert np.isfinite(result)


class TestR2Score:
    def test_one_for_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_zero_for_mean_baseline(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full(4, y_true.mean())
        assert r2_score(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)

    def test_agrees_with_sklearn(self, regression_predictions):
        from sklearn.metrics import r2_score as sk_r2

        y_true, y_pred = regression_predictions
        assert r2_score(y_true, y_pred) == pytest.approx(sk_r2(y_true, y_pred))

    def test_negative_for_worse_than_mean_baseline(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([10.0, -10.0, 20.0, -20.0])
        assert r2_score(y_true, y_pred) < 0.0

    def test_zero_variance_target_returns_zero_not_nan(self):
        y_true = np.full(5, 3.0)
        y_pred = np.full(5, 3.0)
        assert r2_score(y_true, y_pred) == 0.0


class TestExplainedVarianceScore:
    def test_one_for_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert explained_variance_score(y, y) == pytest.approx(1.0)

    def test_agrees_with_sklearn(self, regression_predictions):
        from sklearn.metrics import explained_variance_score as sk_evs

        y_true, y_pred = regression_predictions
        assert explained_variance_score(y_true, y_pred) == pytest.approx(sk_evs(y_true, y_pred))

    def test_zero_variance_target_returns_zero(self):
        y_true = np.full(5, 3.0)
        y_pred = np.full(5, 3.0)
        assert explained_variance_score(y_true, y_pred) == 0.0
