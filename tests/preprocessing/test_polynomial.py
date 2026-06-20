"""
Tests for mlscratch.preprocessing.polynomial
"""

import numpy as np
import pytest

from mlscratch.preprocessing import PolynomialFeatures


class TestPolynomialFeatures:
    def test_degree_2_matches_sklearn(self):
        from sklearn.preprocessing import PolynomialFeatures as SKPolynomialFeatures

        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 2))
        ours = PolynomialFeatures(degree=2).fit_transform(X)
        theirs = SKPolynomialFeatures(degree=2).fit_transform(X)
        np.testing.assert_allclose(ours, theirs)

    def test_degree_3_matches_sklearn(self):
        from sklearn.preprocessing import PolynomialFeatures as SKPolynomialFeatures

        rng = np.random.default_rng(1)
        X = rng.normal(size=(15, 3))
        ours = PolynomialFeatures(degree=3).fit_transform(X)
        theirs = SKPolynomialFeatures(degree=3).fit_transform(X)
        np.testing.assert_allclose(ours, theirs)

    def test_no_bias_drops_constant_column(self):
        X = np.array([[2.0, 3.0]])
        Xt = PolynomialFeatures(degree=1, include_bias=False).fit_transform(X)
        np.testing.assert_allclose(Xt, [[2.0, 3.0]])

    def test_bias_column_is_all_ones(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        Xt = PolynomialFeatures(degree=1, include_bias=True).fit_transform(X)
        np.testing.assert_allclose(Xt[:, 0], 1.0)

    def test_interaction_only_excludes_pure_powers(self):
        from sklearn.preprocessing import PolynomialFeatures as SKPolynomialFeatures

        X = np.array([[1.0, 2.0, 3.0]])
        ours = PolynomialFeatures(degree=2, interaction_only=True).fit_transform(X)
        theirs = SKPolynomialFeatures(degree=2, interaction_only=True).fit_transform(X)
        np.testing.assert_allclose(ours, theirs)

    def test_known_expansion_values(self):
        X = np.array([[2.0, 3.0]])
        Xt = PolynomialFeatures(degree=2).fit_transform(X)
        # [1, a, b, a^2, ab, b^2] = [1, 2, 3, 4, 6, 9]
        np.testing.assert_allclose(Xt, [[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]])

    def test_get_feature_names(self):
        X = np.array([[1.0, 2.0]])
        pf = PolynomialFeatures(degree=2).fit(X)
        names = pf.get_feature_names(["a", "b"])
        assert names == ["1", "a", "b", "a^2", "a b", "b^2"]

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            PolynomialFeatures().transform(np.ones((3, 2)))

    def test_wrong_n_features_raises(self):
        pf = PolynomialFeatures(degree=2).fit(np.ones((5, 2)))
        with pytest.raises(ValueError, match="features"):
            pf.transform(np.ones((5, 3)))

    def test_invalid_degree_raises(self):
        with pytest.raises(ValueError, match="degree"):
            PolynomialFeatures(degree=0)

    def test_linear_model_fits_quadratic_data_after_expansion(self):
        from mlscratch.supervised.linear_models import LinearRegression

        x = np.linspace(-3, 3, 50).reshape(-1, 1)
        y = 2.0 * x.ravel() ** 2 + 1.0 * x.ravel() + 5.0

        X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
        model = LinearRegression().fit(X_poly, y)
        preds = model.predict(X_poly)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        assert r2 > 0.99
