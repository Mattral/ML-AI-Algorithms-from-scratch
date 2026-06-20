"""
Tests for mlscratch.supervised.random_forest
"""

import numpy as np
import pytest

from mlscratch.supervised.random_forest import RandomForestClassifier, RandomForestRegressor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_classification_data():
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=300,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        class_sep=2.0,
        random_state=42,
    )


@pytest.fixture
def multiclass_classification_data():
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=0,
    )


@pytest.fixture
def regression_data():
    from sklearn.datasets import make_regression

    return make_regression(n_samples=300, n_features=5, noise=8.0, random_state=0)


# ============================================================
# RandomForestClassifier
# ============================================================


class TestRandomForestClassifierBasic:
    def test_fit_returns_self(self, binary_classification_data):
        X, y = binary_classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=0)
        assert model.fit(X, y) is model

    def test_n_estimators_trees_grown(self, binary_classification_data):
        X, y = binary_classification_data
        model = RandomForestClassifier(n_estimators=17, random_state=0).fit(X, y)
        assert len(model.estimators_) == 17

    def test_predict_proba_sums_to_one(self, multiclass_classification_data):
        X, y = multiclass_classification_data
        model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (X.shape[0], 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    def test_feature_importances_sum_to_one(self, binary_classification_data):
        X, y = binary_classification_data
        model = RandomForestClassifier(n_estimators=15, random_state=0).fit(X, y)
        np.testing.assert_allclose(model.feature_importances_.sum(), 1.0, atol=1e-8)

    def test_oob_score_in_plausible_range(self, binary_classification_data):
        X, y = binary_classification_data
        model = RandomForestClassifier(n_estimators=50, random_state=0, oob_score=True).fit(X, y)
        assert 0.5 <= model.oob_score_ <= 1.0

    def test_oob_requires_bootstrap(self):
        with pytest.raises(ValueError, match="bootstrap"):
            RandomForestClassifier(oob_score=True, bootstrap=False)

    def test_max_features_int_and_float(self, binary_classification_data):
        X, y = binary_classification_data
        m_int = RandomForestClassifier(n_estimators=5, max_features=2, random_state=0).fit(X, y)
        m_float = RandomForestClassifier(n_estimators=5, max_features=0.5, random_state=0).fit(X, y)
        assert m_int.score(X, y) > 0.5
        assert m_float.score(X, y) > 0.5


class TestRandomForestClassifierCorrectness:
    def test_accuracy_on_clean_data(self, binary_classification_data):
        X, y = binary_classification_data
        model = RandomForestClassifier(n_estimators=25, random_state=42).fit(X, y)
        assert model.score(X, y) >= 0.95

    def test_agrees_with_sklearn(self, binary_classification_data):
        from sklearn.ensemble import RandomForestClassifier as SKRF

        X, y = binary_classification_data
        model = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
        theirs = SKRF(n_estimators=30, random_state=0).fit(X, y)
        assert np.mean(model.predict(X) == theirs.predict(X)) >= 0.85

    def test_more_trees_does_not_hurt_accuracy(self, binary_classification_data):
        X, y = binary_classification_data
        small = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
        large = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
        assert large.score(X, y) >= small.score(X, y) - 0.05


class TestRandomForestClassifierEdgeCases:
    def test_predict_before_fit_raises(self):
        model = RandomForestClassifier(n_estimators=10)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(np.ones((5, 3)))

    def test_invalid_input_raises(self):
        model = RandomForestClassifier(n_estimators=5)
        with pytest.raises(ValueError, match="samples"):
            model.fit(np.ones((10, 2)), np.ones(5))

    def test_invalid_n_estimators_raises(self):
        with pytest.raises(ValueError):
            RandomForestClassifier(n_estimators=0)

    def test_single_tree_forest_runs(self, binary_classification_data):
        X, y = binary_classification_data
        model = RandomForestClassifier(n_estimators=1, random_state=0).fit(X, y)
        assert model.score(X, y) > 0.5


# ============================================================
# RandomForestRegressor
# ============================================================


class TestRandomForestRegressorBasic:
    def test_fit_returns_self(self, regression_data):
        X, y = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=0)
        assert model.fit(X, y) is model

    def test_predict_shape(self, regression_data):
        X, y = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
        assert model.predict(X).shape == (X.shape[0],)

    def test_feature_importances_sum_to_one(self, regression_data):
        X, y = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
        np.testing.assert_allclose(model.feature_importances_.sum(), 1.0, atol=1e-8)

    def test_oob_score_in_plausible_range(self, regression_data):
        X, y = regression_data
        model = RandomForestRegressor(n_estimators=50, random_state=0, oob_score=True).fit(X, y)
        assert model.oob_score_ > 0.5


class TestRandomForestRegressorCorrectness:
    def test_r2_close_to_sklearn(self, regression_data):
        from sklearn.ensemble import RandomForestRegressor as SKRFR

        X, y = regression_data
        model = RandomForestRegressor(n_estimators=50, random_state=0).fit(X, y)
        theirs = SKRFR(n_estimators=50, random_state=0).fit(X, y)
        assert model.score(X, y) >= theirs.score(X, y) - 0.05

    def test_ensemble_outperforms_single_tree(self, regression_data):
        from mlscratch.supervised.decision_tree import DecisionTreeRegressor

        X, y = regression_data
        X_train, y_train = X[:200], y[:200]
        X_test, y_test = X[200:], y[200:]

        forest = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0).fit(
            X_train, y_train
        )
        tree = DecisionTreeRegressor(max_depth=5).fit(X_train, y_train)
        assert forest.score(X_test, y_test) >= tree.score(X_test, y_test) - 0.1


class TestRandomForestRegressorEdgeCases:
    def test_predict_before_fit_raises(self):
        model = RandomForestRegressor(n_estimators=10)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(np.ones((5, 3)))

    def test_invalid_input_raises(self):
        model = RandomForestRegressor(n_estimators=5)
        with pytest.raises(ValueError, match="samples"):
            model.fit(np.ones((10, 2)), np.ones(5))
