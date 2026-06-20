"""
Tests for mlscratch.supervised.decision_tree
"""

import numpy as np
import pytest

from mlscratch.supervised.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def iris_data():
    from sklearn.datasets import load_iris

    data = load_iris()
    return data.data, data.target


@pytest.fixture
def binary_classification_data():
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )


@pytest.fixture
def multiclass_classification_data():
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=240,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=7,
    )


@pytest.fixture
def regression_data():
    from sklearn.datasets import make_regression

    return make_regression(n_samples=200, n_features=4, noise=5.0, random_state=0)


# ============================================================
# DecisionTreeClassifier
# ============================================================


class TestDecisionTreeClassifierBasic:
    def test_fit_returns_self(self, binary_classification_data):
        X, y = binary_classification_data
        model = DecisionTreeClassifier(max_depth=3)
        assert model.fit(X, y) is model

    def test_classes_detected(self, multiclass_classification_data):
        X, y = multiclass_classification_data
        model = DecisionTreeClassifier(max_depth=4).fit(X, y)
        np.testing.assert_array_equal(model.classes_, [0, 1, 2])
        assert model.n_classes_ == 3
        assert model.n_features_in_ == X.shape[1]

    def test_predict_shape_and_labels(self, binary_classification_data):
        X, y = binary_classification_data
        model = DecisionTreeClassifier(max_depth=4).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)
        assert set(np.unique(preds)).issubset(set(np.unique(y)))

    def test_predict_proba_sums_to_one(self, multiclass_classification_data):
        X, y = multiclass_classification_data
        model = DecisionTreeClassifier(max_depth=4).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (X.shape[0], 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)
        assert np.all(proba >= 0.0)

    def test_predict_is_argmax_of_proba(self, multiclass_classification_data):
        X, y = multiclass_classification_data
        model = DecisionTreeClassifier(max_depth=4).fit(X, y)
        proba = model.predict_proba(X)
        expected = model.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(model.predict(X), expected)

    def test_feature_importances_sum_to_one(self, multiclass_classification_data):
        X, y = multiclass_classification_data
        model = DecisionTreeClassifier(max_depth=4).fit(X, y)
        assert model.feature_importances_.shape == (X.shape[1],)
        np.testing.assert_allclose(model.feature_importances_.sum(), 1.0, atol=1e-8)
        assert np.all(model.feature_importances_ >= 0.0)

    def test_apply_returns_one_leaf_per_row(self, binary_classification_data):
        X, y = binary_classification_data
        model = DecisionTreeClassifier(max_depth=3).fit(X, y)
        leaves = model.apply(X)
        assert len(leaves) == X.shape[0]
        assert all(leaf.is_leaf for leaf in leaves)

    def test_max_depth_one_is_a_stump(self, binary_classification_data):
        X, y = binary_classification_data
        model = DecisionTreeClassifier(max_depth=1).fit(X, y)
        assert not model.tree_.is_leaf
        assert model.tree_.left.is_leaf
        assert model.tree_.right.is_leaf

    def test_entropy_criterion_runs(self, binary_classification_data):
        X, y = binary_classification_data
        model = DecisionTreeClassifier(max_depth=4, criterion="entropy").fit(X, y)
        assert model.score(X, y) >= 0.9


class TestDecisionTreeClassifierCorrectness:
    def test_accuracy_on_iris(self, iris_data):
        """Decision tree should classify Iris with >=95% accuracy."""
        X, y = iris_data
        model = DecisionTreeClassifier(max_depth=4).fit(X, y)
        assert model.score(X, y) >= 0.95

    def test_agrees_with_sklearn_gini(self, binary_classification_data):
        from sklearn.tree import DecisionTreeClassifier as SKDecisionTree

        X, y = binary_classification_data
        model = DecisionTreeClassifier(max_depth=5).fit(X, y)
        theirs = SKDecisionTree(max_depth=5, random_state=42).fit(X, y)
        assert np.mean(model.predict(X) == theirs.predict(X)) >= 0.98

    def test_agrees_with_sklearn_entropy(self, multiclass_classification_data):
        from sklearn.tree import DecisionTreeClassifier as SKDecisionTree

        X, y = multiclass_classification_data
        model = DecisionTreeClassifier(max_depth=5, criterion="entropy").fit(X, y)
        theirs = SKDecisionTree(max_depth=5, criterion="entropy", random_state=0).fit(X, y)
        assert np.mean(model.predict(X) == theirs.predict(X)) >= 0.95

    def test_sample_weight_matches_sklearn(self, binary_classification_data):
        from sklearn.tree import DecisionTreeClassifier as SKDecisionTree

        X, y = binary_classification_data
        rng = np.random.default_rng(3)
        w = rng.uniform(0.2, 3.0, len(y))
        model = DecisionTreeClassifier(max_depth=4).fit(X, y, sample_weight=w)
        theirs = SKDecisionTree(max_depth=4, random_state=42).fit(X, y, sample_weight=w)
        assert np.mean(model.predict(X) == theirs.predict(X)) >= 0.95

    def test_unweighted_equals_uniform_weighted(self, binary_classification_data):
        X, y = binary_classification_data
        unweighted = DecisionTreeClassifier(max_depth=4).fit(X, y)
        weighted = DecisionTreeClassifier(max_depth=4).fit(X, y, sample_weight=np.ones(len(y)))
        np.testing.assert_array_equal(unweighted.predict(X), weighted.predict(X))


class TestDecisionTreeClassifierEdgeCases:
    def test_predict_before_fit_raises(self):
        model = DecisionTreeClassifier(max_depth=3)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(np.ones((5, 2)))

    def test_predict_proba_before_fit_raises(self):
        model = DecisionTreeClassifier(max_depth=3)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_proba(np.ones((5, 2)))

    def test_shape_mismatch_raises(self):
        model = DecisionTreeClassifier(max_depth=3)
        with pytest.raises(ValueError, match="samples"):
            model.fit(np.ones((10, 2)), np.ones(5))

    def test_invalid_criterion_raises(self):
        with pytest.raises(ValueError, match="criterion"):
            DecisionTreeClassifier(criterion="invalid")

    def test_invalid_min_samples_split_raises(self):
        with pytest.raises(ValueError):
            DecisionTreeClassifier(min_samples_split=1)

    def test_single_class_is_pure_leaf(self):
        X = np.random.default_rng(0).standard_normal((20, 3))
        y = np.zeros(20, dtype=int)
        model = DecisionTreeClassifier(max_depth=5).fit(X, y)
        assert model.tree_.is_leaf
        assert model.score(X, y) == 1.0

    def test_min_samples_leaf_respected(self, binary_classification_data):
        X, y = binary_classification_data
        model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=20).fit(X, y)
        leaves = model.apply(X)
        for leaf in {id(leaf) for leaf in leaves}:
            count = sum(1 for leaf_ in leaves if id(leaf_) == leaf)
            assert count >= 20


# ============================================================
# DecisionTreeRegressor
# ============================================================


class TestDecisionTreeRegressorBasic:
    def test_fit_returns_self(self, regression_data):
        X, y = regression_data
        model = DecisionTreeRegressor(max_depth=4)
        assert model.fit(X, y) is model

    def test_predict_shape(self, regression_data):
        X, y = regression_data
        model = DecisionTreeRegressor(max_depth=4).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_score_is_r2(self, regression_data):
        X, y = regression_data
        model = DecisionTreeRegressor(max_depth=6).fit(X, y)
        assert model.score(X, y) > 0.8

    def test_feature_importances_sum_to_one(self, regression_data):
        X, y = regression_data
        model = DecisionTreeRegressor(max_depth=4).fit(X, y)
        np.testing.assert_allclose(model.feature_importances_.sum(), 1.0, atol=1e-8)

    def test_deeper_tree_fits_training_data_better(self, regression_data):
        X, y = regression_data
        shallow = DecisionTreeRegressor(max_depth=2).fit(X, y)
        deep = DecisionTreeRegressor(max_depth=8).fit(X, y)
        assert deep.score(X, y) >= shallow.score(X, y)


class TestDecisionTreeRegressorCorrectness:
    def test_agrees_with_sklearn(self, regression_data):
        from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor

        X, y = regression_data
        model = DecisionTreeRegressor(max_depth=4).fit(X, y)
        theirs = SKDecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
        np.testing.assert_allclose(model.score(X, y), theirs.score(X, y), atol=1e-8)
        np.testing.assert_allclose(model.predict(X), theirs.predict(X), atol=1e-8)

    def test_sample_weight_matches_sklearn(self, regression_data):
        from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor

        X, y = regression_data
        rng = np.random.default_rng(2)
        w = rng.uniform(0.5, 2.0, len(y))
        model = DecisionTreeRegressor(max_depth=4).fit(X, y, sample_weight=w)
        theirs = SKDecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y, sample_weight=w)
        corr = np.corrcoef(model.predict(X), theirs.predict(X))[0, 1]
        assert corr > 0.95

    def test_perfect_fit_with_unbounded_depth(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((30, 2))
        y = rng.standard_normal(30)
        model = DecisionTreeRegressor(max_depth=None, min_samples_split=2).fit(X, y)
        np.testing.assert_allclose(model.predict(X), y, atol=1e-8)


class TestDecisionTreeRegressorEdgeCases:
    def test_predict_before_fit_raises(self):
        model = DecisionTreeRegressor(max_depth=3)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(np.ones((5, 2)))

    def test_shape_mismatch_raises(self):
        model = DecisionTreeRegressor(max_depth=3)
        with pytest.raises(ValueError, match="samples"):
            model.fit(np.ones((10, 2)), np.ones(5))

    def test_constant_target_gives_zero_variance_leaf(self):
        X = np.random.default_rng(0).standard_normal((20, 3))
        y = np.full(20, 7.0)
        model = DecisionTreeRegressor(max_depth=5).fit(X, y)
        assert model.tree_.is_leaf
        np.testing.assert_allclose(model.predict(X), 7.0)

    def test_apply_before_fit_raises(self):
        model = DecisionTreeRegressor()
        with pytest.raises(RuntimeError, match="fit"):
            model.apply(np.ones((3, 2)))
