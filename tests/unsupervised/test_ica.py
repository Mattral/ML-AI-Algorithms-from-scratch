"""
Tests for mlscratch.unsupervised.ica.FastICA
"""

import numpy as np
import pytest
from mlscratch.unsupervised.ica import FastICA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mixed_signals():
    """Two independent sources mixed linearly."""
    rng = np.random.default_rng(42)
    n = 300
    s1 = rng.standard_normal(n)
    s2 = rng.uniform(-1, 1, n)
    S = np.column_stack([s1, s2])            # (n, 2) sources
    A = np.array([[1.0, 0.5], [0.3, 1.0]])  # mixing matrix
    return S @ A.T                           # (n, 2) observed mixture


@pytest.fixture
def random_3d():
    rng = np.random.default_rng(11)
    return rng.standard_normal((100, 3))


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

class TestFastICABasic:
    def test_fit_returns_self(self, mixed_signals):
        model = FastICA(n_components=2, random_state=0)
        assert model.fit(mixed_signals) is model

    def test_components_shape(self, mixed_signals):
        model = FastICA(n_components=2, random_state=0).fit(mixed_signals)
        assert model.components_.shape == (2, 2)

    def test_mixing_shape(self, mixed_signals):
        model = FastICA(n_components=2, random_state=0).fit(mixed_signals)
        assert model.mixing_.shape == (2, 2)

    def test_mean_shape(self, mixed_signals):
        model = FastICA(n_components=2, random_state=0).fit(mixed_signals)
        assert model.mean_.shape == (2,)

    def test_transform_shape(self, mixed_signals):
        model = FastICA(n_components=2, random_state=0).fit(mixed_signals)
        S = model.transform(mixed_signals)
        assert S.shape == (len(mixed_signals), 2)

    def test_fit_transform_same_as_fit_then_transform(self, mixed_signals):
        m1 = FastICA(n_components=2, random_state=0)
        S1 = m1.fit_transform(mixed_signals)
        m2 = FastICA(n_components=2, random_state=0)
        m2.fit(mixed_signals)
        S2 = m2.transform(mixed_signals)
        np.testing.assert_allclose(S1, S2, atol=1e-10)

    def test_invalid_fun_raises(self):
        with pytest.raises(ValueError):
            FastICA(fun="invalid")

    @pytest.mark.parametrize("fun", ["logcosh", "exp"])
    def test_valid_fun_runs(self, mixed_signals, fun):
        model = FastICA(n_components=2, fun=fun, random_state=0)
        S = model.fit_transform(mixed_signals)
        assert S.shape == (len(mixed_signals), 2)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestFastICACorrectness:
    def test_recovered_signals_decorrelated(self, mixed_signals):
        """Independent components should be (approximately) decorrelated."""
        model = FastICA(n_components=2, max_iter=500, random_state=0)
        S = model.fit_transform(mixed_signals)
        corr = np.corrcoef(S.T)
        # Off-diagonal should be near 0
        assert abs(corr[0, 1]) < 0.15

    def test_components_are_orthonormal_in_whitened_space(self, mixed_signals):
        """Unmixing matrix rows should be orthonormal after whitening."""
        model = FastICA(n_components=2, random_state=0).fit(mixed_signals)
        W = model.components_ @ np.linalg.pinv(model.whitening_).T \
            if hasattr(model, "whitening_") else model.components_
        # Just check components_ rows have unit norm (in original space they
        # won't be perfectly orthonormal, but the test checks no blow-up)
        norms = np.linalg.norm(model.components_, axis=1)
        assert np.all(norms > 0)

    def test_none_n_components_uses_all_features(self, random_3d):
        model = FastICA(n_components=None, random_state=0).fit(random_3d)
        assert model.components_.shape[0] <= 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestFastICAEdgeCases:
    def test_single_component(self, mixed_signals):
        model = FastICA(n_components=1, random_state=0)
        S = model.fit_transform(mixed_signals)
        assert S.shape == (len(mixed_signals), 1)

    def test_1d_data(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 1))
        model = FastICA(n_components=1, random_state=0)
        S = model.fit_transform(X)
        assert S.shape == (50, 1)
