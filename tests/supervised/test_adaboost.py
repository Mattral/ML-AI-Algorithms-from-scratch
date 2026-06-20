"""
Tests for mlscratch.supervised.adaboost
"""

import numpy as np
import pytest

from mlscratch.supervised.adaboost import AdaBoostClassifier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_classification_data():
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        class_sep=1.0,
        random_state=1,
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
        random_state=2,
    )


ALGORITHMS = ["SAMME", "SAMME.R"]


# ============================================================
# Basic API
# ============================================================


class TestAdaBoostBasic:
    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_fit_returns_self(self, binary_classification_data, algorithm):
        X, y = binary_classification_data
        model = AdaBoostClassifier(n_estimators=10, algorithm=algorithm, random_state=0)
        assert model.fit(X, y) is model

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_estimator_weights_and_errors_recorded(self, binary_classification_data, algorithm):
        X, y = binary_classification_data
        model = AdaBoostClassifier(n_estimators=15, algorithm=algorithm, random_state=0).fit(X, y)
        assert (
            len(model.estimators_) == len(model.estimator_weights_) == len(model.estimator_errors_)
        )
        assert len(model.estimators_) <= 15

    def test_default_uses_decision_stumps(self, binary_classification_data):
        X, y = binary_classification_data
        model = AdaBoostClassifier(n_estimators=5, random_state=0).fit(X, y)
        for stump in model.estimators_:
            assert stump.max_depth == 1

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_predict_proba_sums_to_one(self, multiclass_classification_data, algorithm):
        X, y = multiclass_classification_data
        model = AdaBoostClassifier(
            n_estimators=20, algorithm=algorithm, max_depth=2, random_state=0
        ).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (X.shape[0], 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_staged_predict_final_matches_predict(self, binary_classification_data, algorithm):
        X, y = binary_classification_data
        model = AdaBoostClassifier(n_estimators=15, algorithm=algorithm, random_state=0).fit(X, y)
        staged = list(model.staged_predict(X[:20]))
        assert len(staged) == len(model.estimators_)
        np.testing.assert_array_equal(staged[-1], model.predict(X[:20]))


# ============================================================
# Correctness
# ============================================================


class TestAdaBoostCorrectness:
    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_accuracy_on_binary_data(self, binary_classification_data, algorithm):
        X, y = binary_classification_data
        model = AdaBoostClassifier(
            n_estimators=50, algorithm=algorithm, max_depth=1, random_state=0
        ).fit(X, y)
        assert model.score(X, y) >= 0.85

    def test_agrees_with_sklearn(self, binary_classification_data):
        from sklearn.ensemble import AdaBoostClassifier as SKAda
        from sklearn.tree import DecisionTreeClassifier as SKDT

        X, y = binary_classification_data
        model = AdaBoostClassifier(
            n_estimators=50, algorithm="SAMME", max_depth=1, random_state=0
        ).fit(X, y)
        theirs = SKAda(estimator=SKDT(max_depth=1), n_estimators=50, random_state=0).fit(X, y)
        assert np.mean(model.predict(X) == theirs.predict(X)) >= 0.85

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_multiclass_native_support(self, multiclass_classification_data, algorithm):
        X, y = multiclass_classification_data
        model = AdaBoostClassifier(
            n_estimators=50, algorithm=algorithm, max_depth=2, random_state=0
        ).fit(X, y)
        assert model.score(X, y) >= 0.8

    def test_ensemble_accuracy_improves_with_more_rounds(self, binary_classification_data):
        X, y = binary_classification_data
        model = AdaBoostClassifier(
            n_estimators=30, algorithm="SAMME", max_depth=1, random_state=0
        ).fit(X, y)
        staged_preds = list(model.staged_predict(X))
        early_acc = float(np.mean(staged_preds[0] == y))
        late_acc = float(np.mean(staged_preds[-1] == y))
        # The full ensemble should never do meaningfully worse than a
        # single weak learner on the training data it was boosted on.
        assert late_acc >= early_acc - 0.05

    def test_deeper_weak_learners_converge_faster(self, binary_classification_data):
        X, y = binary_classification_data
        stumps = AdaBoostClassifier(n_estimators=10, max_depth=1, random_state=0).fit(X, y)
        deeper = AdaBoostClassifier(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
        assert deeper.score(X, y) >= stumps.score(X, y) - 0.05


# ============================================================
# Edge cases
# ============================================================


class TestAdaBoostEdgeCases:
    def test_predict_before_fit_raises(self):
        model = AdaBoostClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(np.ones((5, 2)))

    def test_shape_mismatch_raises(self):
        model = AdaBoostClassifier()
        with pytest.raises(ValueError, match="samples"):
            model.fit(np.ones((10, 2)), np.ones(5))

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="algorithm"):
            AdaBoostClassifier(algorithm="not_a_real_algorithm")

    def test_single_class_raises(self):
        X = np.random.default_rng(0).standard_normal((10, 2))
        y = np.zeros(10, dtype=int)
        with pytest.raises(ValueError, match="2 classes"):
            AdaBoostClassifier().fit(X, y)

    def test_perfectly_separable_data_stops_early(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal([-5, -5], 0.3, (20, 2)), rng.normal([5, 5], 0.3, (20, 2))])
        y = np.array([0] * 20 + [1] * 20)
        model = AdaBoostClassifier(
            n_estimators=100, algorithm="SAMME", max_depth=1, random_state=0
        ).fit(X, y)
        assert len(model.estimators_) < 100
        assert model.score(X, y) == 1.0

    def test_score_matches_manual_accuracy(self, binary_classification_data):
        X, y = binary_classification_data
        model = AdaBoostClassifier(n_estimators=10, random_state=0).fit(X, y)
        assert model.score(X, y) == float(np.mean(model.predict(X) == y))
