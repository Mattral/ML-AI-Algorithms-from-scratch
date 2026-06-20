"""
Tests for mlscratch.preprocessing.scalers
"""

import numpy as np
import pytest

from mlscratch.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler


@pytest.fixture
def feature_matrix():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=[10, -5, 100], scale=[2, 1, 50], size=(80, 3))
    return X


# ============================================================
# StandardScaler
# ============================================================


class TestStandardScaler:
    def test_fit_transform_zero_mean_unit_var(self, feature_matrix):
        Xt = StandardScaler().fit_transform(feature_matrix)
        np.testing.assert_allclose(Xt.mean(axis=0), 0.0, atol=1e-8)
        np.testing.assert_allclose(Xt.std(axis=0), 1.0, atol=1e-8)

    def test_agrees_with_sklearn(self, feature_matrix):
        from sklearn.preprocessing import StandardScaler as SKStandardScaler

        ours = StandardScaler().fit_transform(feature_matrix)
        theirs = SKStandardScaler().fit_transform(feature_matrix)
        np.testing.assert_allclose(ours, theirs, atol=1e-8)

    def test_inverse_transform_recovers_original(self, feature_matrix):
        scaler = StandardScaler().fit(feature_matrix)
        Xt = scaler.transform(feature_matrix)
        np.testing.assert_allclose(scaler.inverse_transform(Xt), feature_matrix, atol=1e-8)

    def test_constant_column_does_not_produce_nan(self):
        X = np.column_stack([np.ones(10), np.arange(10, dtype=float)])
        Xt = StandardScaler().fit_transform(X)
        assert np.all(np.isfinite(Xt))
        np.testing.assert_allclose(Xt[:, 0], 0.0)

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            StandardScaler().transform(np.ones((3, 2)))

    def test_with_mean_false(self, feature_matrix):
        scaler = StandardScaler(with_mean=False).fit(feature_matrix)
        np.testing.assert_array_equal(scaler.mean_, np.zeros(3))

    def test_wrong_n_features_raises(self, feature_matrix):
        scaler = StandardScaler().fit(feature_matrix)
        with pytest.raises(ValueError, match="features"):
            scaler.transform(np.ones((5, 2)))


# ============================================================
# MinMaxScaler
# ============================================================


class TestMinMaxScaler:
    def test_default_range_is_zero_one(self, feature_matrix):
        Xt = MinMaxScaler().fit_transform(feature_matrix)
        np.testing.assert_allclose(Xt.min(axis=0), 0.0, atol=1e-8)
        np.testing.assert_allclose(Xt.max(axis=0), 1.0, atol=1e-8)

    def test_custom_range(self, feature_matrix):
        Xt = MinMaxScaler(feature_range=(-1, 1)).fit_transform(feature_matrix)
        np.testing.assert_allclose(Xt.min(axis=0), -1.0, atol=1e-8)
        np.testing.assert_allclose(Xt.max(axis=0), 1.0, atol=1e-8)

    def test_agrees_with_sklearn(self, feature_matrix):
        from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler

        ours = MinMaxScaler().fit_transform(feature_matrix)
        theirs = SKMinMaxScaler().fit_transform(feature_matrix)
        np.testing.assert_allclose(ours, theirs, atol=1e-8)

    def test_inverse_transform_recovers_original(self, feature_matrix):
        scaler = MinMaxScaler().fit(feature_matrix)
        Xt = scaler.transform(feature_matrix)
        np.testing.assert_allclose(scaler.inverse_transform(Xt), feature_matrix, atol=1e-8)

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="feature_range"):
            MinMaxScaler(feature_range=(1.0, 0.0))

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            MinMaxScaler().transform(np.ones((3, 2)))


# ============================================================
# RobustScaler
# ============================================================


class TestRobustScaler:
    def test_agrees_with_sklearn(self, feature_matrix):
        from sklearn.preprocessing import RobustScaler as SKRobustScaler

        ours = RobustScaler().fit_transform(feature_matrix)
        theirs = SKRobustScaler().fit_transform(feature_matrix)
        np.testing.assert_allclose(ours, theirs, atol=1e-8)

    def test_robust_to_outliers_vs_standard_scaler(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (100, 1))
        X[0, 0] = 1000.0  # extreme outlier

        robust = RobustScaler().fit_transform(X)
        standard = StandardScaler().fit_transform(X)
        # the bulk of (non-outlier) robust-scaled points should be closer
        # to a sane range than the standard-scaled ones, which get crushed
        # toward zero by the outlier-inflated std
        assert np.median(np.abs(robust[1:])) > np.median(np.abs(standard[1:]))

    def test_invalid_quantile_range_raises(self):
        with pytest.raises(ValueError, match="quantile_range"):
            RobustScaler(quantile_range=(75.0, 25.0))

    def test_inverse_transform_recovers_original(self, feature_matrix):
        scaler = RobustScaler().fit(feature_matrix)
        Xt = scaler.transform(feature_matrix)
        np.testing.assert_allclose(scaler.inverse_transform(Xt), feature_matrix, atol=1e-8)


# ============================================================
# Normalizer
# ============================================================


class TestNormalizer:
    def test_l2_rows_have_unit_norm(self, feature_matrix):
        Xt = Normalizer(norm="l2").fit_transform(feature_matrix)
        norms = np.sqrt(np.sum(Xt**2, axis=1))
        np.testing.assert_allclose(norms, 1.0, atol=1e-8)

    def test_l1_rows_have_unit_norm(self, feature_matrix):
        Xt = Normalizer(norm="l1").fit_transform(feature_matrix)
        norms = np.sum(np.abs(Xt), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-8)

    def test_agrees_with_sklearn(self, feature_matrix):
        from sklearn.preprocessing import Normalizer as SKNormalizer

        ours = Normalizer(norm="l2").fit_transform(feature_matrix)
        theirs = SKNormalizer(norm="l2").fit_transform(feature_matrix)
        np.testing.assert_allclose(ours, theirs, atol=1e-8)

    def test_zero_row_does_not_produce_nan(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        Xt = Normalizer().fit_transform(X)
        assert np.all(np.isfinite(Xt))

    def test_invalid_norm_raises(self):
        with pytest.raises(ValueError, match="norm"):
            Normalizer(norm="l3")
