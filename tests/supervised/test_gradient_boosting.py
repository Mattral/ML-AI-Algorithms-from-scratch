"""
Tests for mlscratch.supervised.gradient_boosting
"""

import numpy as np
import pytest

from mlscratch.supervised.gradient_boosting import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def regression_data():
    from sklearn.datasets import make_regression

    return make_regression(n_samples=300, n_features=5, noise=8.0, random_state=0)


@pytest.fixture
def binary_classification_data():
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        class_sep=1.2,
        random_state=1,
    )


@pytest.fixture
def multiclass_classification_data():
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=150,
        n_features=4,
        n_classes=3,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=2,
    )


# ============================================================
# GradientBoostingRegressor
# ============================================================


class TestGBRegressorBasic:
    def test_fit_returns_self(self, regression_data):
        X, y = regression_data
        model = GradientBoostingRegressor(n_estimators=10, random_state=0)
        assert model.fit(X, y) is model

    def test_n_estimators_trees_grown(self, regression_data):
        X, y = regression_data
        model = GradientBoostingRegressor(n_estimators=13, random_state=0).fit(X, y)
        assert len(model.estimators_) == 13

    def test_predict_shape(self, regression_data):
        X, y = regression_data
        model = GradientBoostingRegressor(n_estimators=10, random_state=0).fit(X, y)
        assert model.predict(X).shape == (X.shape[0],)

    def test_train_score_monotonically_improves(self, regression_data):
        X, y = regression_data
        model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, random_state=0).fit(
            X, y
        )
        # MSE should be (weakly) decreasing on average over the course of training
        assert model.train_score_[-1] < model.train_score_[0]

    def test_staged_predict_final_matches_predict(self, regression_data):
        X, y = regression_data
        model = GradientBoostingRegressor(n_estimators=20, random_state=0).fit(X, y)
        staged = list(model.staged_predict(X[:10]))
        assert len(staged) == 20
        np.testing.assert_allclose(staged[-1], model.predict(X[:10]))

    def test_feature_importances_sum_to_one(self, regression_data):
        X, y = regression_data
        model = GradientBoostingRegressor(n_estimators=10, random_state=0).fit(X, y)
        np.testing.assert_allclose(model.feature_importances_.sum(), 1.0, atol=1e-8)

    def test_subsample_runs(self, regression_data):
        X, y = regression_data
        model = GradientBoostingRegressor(n_estimators=20, subsample=0.5, random_state=0).fit(X, y)
        assert model.score(X, y) > 0.5


class TestGBRegressorCorrectness:
    def test_squared_error_matches_sklearn_closely(self, regression_data):
        from sklearn.ensemble import GradientBoostingRegressor as SKGBR

        X, y = regression_data
        model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0
        ).fit(X, y)
        theirs = SKGBR(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(X, y)
        np.testing.assert_allclose(model.score(X, y), theirs.score(X, y), atol=1e-6)

    def test_more_estimators_improves_training_fit(self, regression_data):
        X, y = regression_data
        few = GradientBoostingRegressor(n_estimators=5, random_state=0).fit(X, y)
        many = GradientBoostingRegressor(n_estimators=100, random_state=0).fit(X, y)
        assert many.score(X, y) > few.score(X, y)

    def test_absolute_error_robust_to_outliers(self):
        rng = np.random.default_rng(4)
        X = np.linspace(-3, 3, 100).reshape(-1, 1)
        y = 2.0 * X.ravel() + rng.normal(0, 0.3, 100)
        y[0], y[1] = 200.0, -200.0  # gross outliers

        sq = GradientBoostingRegressor(
            n_estimators=50, max_depth=2, loss="squared_error", random_state=0
        ).fit(X, y)
        mae_model = GradientBoostingRegressor(
            n_estimators=50, max_depth=2, loss="absolute_error", random_state=0
        ).fit(X, y)

        clean_mask = np.ones(100, dtype=bool)
        clean_mask[:2] = False
        mae_sq = np.mean(np.abs(sq.predict(X[clean_mask]) - y[clean_mask]))
        mae_lad = np.mean(np.abs(mae_model.predict(X[clean_mask]) - y[clean_mask]))
        assert mae_lad <= mae_sq * 1.5  # LAD should not be drastically worse, typically better


class TestGBRegressorEdgeCases:
    def test_predict_before_fit_raises(self):
        model = GradientBoostingRegressor()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(np.ones((5, 2)))

    def test_shape_mismatch_raises(self):
        model = GradientBoostingRegressor()
        with pytest.raises(ValueError, match="samples"):
            model.fit(np.ones((10, 2)), np.ones(5))

    def test_invalid_loss_raises(self):
        with pytest.raises(ValueError, match="loss"):
            GradientBoostingRegressor(loss="hinge")

    def test_invalid_subsample_raises(self):
        with pytest.raises(ValueError, match="subsample"):
            GradientBoostingRegressor(subsample=1.5)

    def test_non_positive_learning_rate_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            GradientBoostingRegressor(learning_rate=0.0)


# ============================================================
# GradientBoostingClassifier
# ============================================================


class TestGBClassifierBasic:
    def test_fit_returns_self(self, binary_classification_data):
        X, y = binary_classification_data
        model = GradientBoostingClassifier(n_estimators=10, random_state=0)
        assert model.fit(X, y) is model

    def test_predict_proba_sums_to_one(self, binary_classification_data):
        X, y = binary_classification_data
        model = GradientBoostingClassifier(n_estimators=10, random_state=0).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    def test_train_score_is_decreasing_deviance(self, binary_classification_data):
        X, y = binary_classification_data
        model = GradientBoostingClassifier(n_estimators=50, random_state=0).fit(X, y)
        assert model.train_score_[-1] < model.train_score_[0]

    def test_feature_importances_sum_to_one(self, binary_classification_data):
        X, y = binary_classification_data
        model = GradientBoostingClassifier(n_estimators=10, random_state=0).fit(X, y)
        np.testing.assert_allclose(model.feature_importances_.sum(), 1.0, atol=1e-8)


class TestGBClassifierCorrectness:
    def test_accuracy_on_clean_data(self, binary_classification_data):
        X, y = binary_classification_data
        model = GradientBoostingClassifier(n_estimators=100, random_state=0).fit(X, y)
        assert model.score(X, y) >= 0.9

    def test_agrees_with_sklearn(self, binary_classification_data):
        from sklearn.ensemble import GradientBoostingClassifier as SKGBC

        X, y = binary_classification_data
        model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0
        ).fit(X, y)
        theirs = SKGBC(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(X, y)
        assert np.mean(model.predict(X) == theirs.predict(X)) >= 0.9


class TestGBClassifierEdgeCases:
    def test_predict_before_fit_raises(self):
        model = GradientBoostingClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(np.ones((5, 2)))

    def test_shape_mismatch_raises(self):
        model = GradientBoostingClassifier()
        with pytest.raises(ValueError, match="samples"):
            model.fit(np.ones((10, 2)), np.ones(5))

    def test_multiclass_raises(self, multiclass_classification_data):
        X, y = multiclass_classification_data
        model = GradientBoostingClassifier(n_estimators=5)
        with pytest.raises(ValueError, match="binary"):
            model.fit(X, y)
