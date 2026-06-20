"""
Tests for mlscratch.supervised.svm
"""

import numpy as np
import pytest

from mlscratch.supervised.svm import SVC

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linearly_separable_data():
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=150,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        class_sep=2.5,
        random_state=42,
    )


@pytest.fixture
def moons_data():
    from sklearn.datasets import make_moons

    return make_moons(n_samples=150, noise=0.15, random_state=0)


@pytest.fixture
def multiclass_data():
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=180,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=7,
    )


# ============================================================
# Basic API
# ============================================================


class TestSVCBasic:
    def test_fit_returns_self(self, linearly_separable_data):
        X, y = linearly_separable_data
        model = SVC(kernel="linear", max_iter=10, random_state=0)
        assert model.fit(X, y) is model

    def test_support_vectors_are_subset_of_training_data(self, linearly_separable_data):
        X, y = linearly_separable_data
        model = SVC(kernel="linear", max_iter=10, random_state=0).fit(X, y)
        assert 0 < model.n_support_ <= X.shape[0]
        assert model.support_vectors_.shape == (model.n_support_, X.shape[1])
        np.testing.assert_array_equal(model.support_vectors_, X[model.support_])

    def test_decision_function_shape_binary(self, linearly_separable_data):
        X, y = linearly_separable_data
        model = SVC(kernel="rbf", max_iter=10, random_state=0).fit(X, y)
        scores = model.decision_function(X)
        assert scores.shape == (X.shape[0],)

    def test_decision_function_shape_multiclass(self, multiclass_data):
        X, y = multiclass_data
        model = SVC(kernel="rbf", max_iter=5, random_state=0).fit(X, y)
        scores = model.decision_function(X)
        assert scores.shape == (X.shape[0], 3)

    def test_predict_returns_known_labels(self, linearly_separable_data):
        X, y = linearly_separable_data
        model = SVC(kernel="rbf", max_iter=10, random_state=0).fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset(set(np.unique(y)))

    def test_multiclass_flag_set_correctly(self, linearly_separable_data, multiclass_data):
        Xb, yb = linearly_separable_data
        Xm, ym = multiclass_data
        assert not SVC(max_iter=5).fit(Xb, yb).multiclass_
        assert SVC(max_iter=5).fit(Xm, ym).multiclass_

    def test_custom_callable_kernel(self, linearly_separable_data):
        X, y = linearly_separable_data

        def my_linear(A, B):
            return A @ B.T

        model = SVC(kernel=my_linear, max_iter=10, random_state=0).fit(X, y)
        assert model.score(X, y) >= 0.8


# ============================================================
# Correctness
# ============================================================


class TestSVCCorrectness:
    def test_linear_kernel_separates_clean_data(self, linearly_separable_data):
        X, y = linearly_separable_data
        model = SVC(C=1.0, kernel="linear", max_iter=20, random_state=0).fit(X, y)
        assert model.score(X, y) >= 0.95

    def test_rbf_agrees_with_sklearn_on_moons(self, moons_data):
        from sklearn.svm import SVC as SKSVC

        X, y = moons_data
        model = SVC(C=1.0, kernel="rbf", gamma="scale", max_iter=20, random_state=0).fit(X, y)
        theirs = SKSVC(C=1.0, kernel="rbf", gamma="scale").fit(X, y)
        assert np.mean(model.predict(X) == theirs.predict(X)) >= 0.9

    def test_rbf_outperforms_linear_on_nonlinear_data(self, moons_data):
        X, y = moons_data
        linear = SVC(kernel="linear", max_iter=20, random_state=0).fit(X, y)
        rbf = SVC(kernel="rbf", gamma="scale", max_iter=20, random_state=0).fit(X, y)
        assert rbf.score(X, y) >= linear.score(X, y)

    def test_multiclass_one_vs_rest_reasonable_accuracy(self, multiclass_data):
        X, y = multiclass_data
        model = SVC(kernel="rbf", gamma="scale", max_iter=10, random_state=0).fit(X, y)
        assert model.score(X, y) >= 0.85

    def test_larger_c_reduces_margin_violations(self, moons_data):
        X, y = moons_data
        loose = SVC(C=0.01, kernel="rbf", max_iter=15, random_state=0).fit(X, y)
        tight = SVC(C=100.0, kernel="rbf", max_iter=15, random_state=0).fit(X, y)
        assert tight.score(X, y) >= loose.score(X, y) - 0.05


# ============================================================
# Edge cases
# ============================================================


class TestSVCEdgeCases:
    def test_predict_before_fit_raises(self):
        model = SVC()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(np.ones((5, 2)))

    def test_decision_function_before_fit_raises(self):
        model = SVC()
        with pytest.raises(RuntimeError, match="fit"):
            model.decision_function(np.ones((5, 2)))

    def test_shape_mismatch_raises(self):
        model = SVC()
        with pytest.raises(ValueError, match="samples"):
            model.fit(np.ones((10, 2)), np.ones(5))

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValueError, match="kernel"):
            SVC(kernel="not_a_kernel")

    def test_non_positive_c_raises(self):
        with pytest.raises(ValueError, match="C"):
            SVC(C=0.0)

    def test_single_class_raises(self):
        X = np.random.default_rng(0).standard_normal((10, 2))
        y = np.zeros(10, dtype=int)
        with pytest.raises(ValueError, match="2 classes"):
            SVC().fit(X, y)

    def test_score_matches_manual_accuracy(self, linearly_separable_data):
        X, y = linearly_separable_data
        model = SVC(kernel="linear", max_iter=10, random_state=0).fit(X, y)
        assert model.score(X, y) == float(np.mean(model.predict(X) == y))
