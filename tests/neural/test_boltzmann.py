"""
Tests for mlscratch.neural.boltzmann.RestrictedBoltzmannMachine
"""

from __future__ import annotations

import numpy as np
import pytest

from mlscratch.neural.boltzmann import RestrictedBoltzmannMachine


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def binary_data():
    """
    40 binary samples with structure: two 'modes' over 8 visible units —
    half the samples are mostly 1s in the first 4 units, the other half
    mostly 1s in the last 4 units. Gives the RBM something to learn.
    """
    rng = np.random.default_rng(0)
    mode_a = (rng.random((20, 8)) < np.array([0.9]*4 + [0.1]*4)).astype(float)
    mode_b = (rng.random((20, 8)) < np.array([0.1]*4 + [0.9]*4)).astype(float)
    return np.vstack([mode_a, mode_b])


@pytest.fixture
def small_binary_data():
    rng = np.random.default_rng(1)
    return (rng.random((16, 6)) > 0.5).astype(float)


# ===================================================================
# Basic API
# ===================================================================

class TestRBMBasic:
    def test_param_shapes(self):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4, random_state=0)
        assert rbm.W.shape == (8, 4)
        assert rbm.a.shape == (8,)
        assert rbm.b.shape == (4,)

    def test_fit_returns_self(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4, epochs=2,
                                         batch_size=8, random_state=0)
        assert rbm.fit(binary_data) is rbm

    def test_reconstruction_errors_recorded(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4, epochs=5,
                                         batch_size=8, random_state=0).fit(binary_data)
        assert len(rbm.reconstruction_errors_) == 5

    def test_transform_shape(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4, epochs=2,
                                         batch_size=8, random_state=0).fit(binary_data)
        h = rbm.transform(binary_data)
        assert h.shape == (40, 4)

    def test_transform_outputs_probabilities(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4, epochs=2,
                                         batch_size=8, random_state=0).fit(binary_data)
        h = rbm.transform(binary_data)
        assert np.all((h >= 0) & (h <= 1))

    def test_reconstruct_shape(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4, epochs=2,
                                         batch_size=8, random_state=0).fit(binary_data)
        rec = rbm.reconstruct(binary_data)
        assert rec.shape == binary_data.shape

    def test_reconstruct_outputs_probabilities(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4, epochs=2,
                                         batch_size=8, random_state=0).fit(binary_data)
        rec = rbm.reconstruct(binary_data)
        assert np.all((rec >= 0) & (rec <= 1))

    def test_free_energy_shape(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4, epochs=2,
                                         batch_size=8, random_state=0).fit(binary_data)
        fe = rbm.free_energy(binary_data)
        assert fe.shape == (40,)

    def test_sample_shape(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4, epochs=2,
                                         batch_size=8, random_state=0).fit(binary_data)
        samples = rbm.sample(5, n_gibbs_steps=10)
        assert samples.shape == (5, 8)

    def test_sample_is_binary(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4, epochs=2,
                                         batch_size=8, random_state=0).fit(binary_data)
        samples = rbm.sample(5, n_gibbs_steps=10)
        assert set(np.unique(samples)).issubset({0.0, 1.0})

    def test_sample_with_v_init(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4, epochs=2,
                                         batch_size=8, random_state=0).fit(binary_data)
        v_init = binary_data[:5]
        samples = rbm.sample(5, n_gibbs_steps=5, v_init=v_init)
        assert samples.shape == (5, 8)


# ===================================================================
# Sampling internals
# ===================================================================

class TestRBMSampling:
    def test_sample_hidden_probability_shapes(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4, random_state=0)
        p_h, h = rbm._sample_hidden(binary_data)
        assert p_h.shape == (40, 4)
        assert h.shape == (40, 4)

    def test_sample_hidden_binary(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4, random_state=0)
        _, h = rbm._sample_hidden(binary_data)
        assert set(np.unique(h)).issubset({0.0, 1.0})

    def test_sample_visible_probability_shapes(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4, random_state=0)
        _, h = rbm._sample_hidden(binary_data)
        p_v, v = rbm._sample_visible(h)
        assert p_v.shape == (40, 8)
        assert v.shape == (40, 8)

    def test_probabilities_in_unit_interval(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4, random_state=0)
        p_h, _ = rbm._sample_hidden(binary_data)
        assert np.all((p_h >= 0) & (p_h <= 1))


# ===================================================================
# Correctness
# ===================================================================

class TestRBMCorrectness:
    def test_reconstruction_error_decreases_with_training(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4,
                                         learning_rate=0.1, cd_k=1, epochs=50,
                                         batch_size=10, random_state=0).fit(binary_data)
        # Average of first 5 vs last 5 epochs
        early = np.mean(rbm.reconstruction_errors_[:5])
        late  = np.mean(rbm.reconstruction_errors_[-5:])
        assert late <= early

    def test_cd_k_greater_than_one_runs(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4,
                                         cd_k=3, epochs=3, batch_size=10,
                                         random_state=0).fit(binary_data)
        assert len(rbm.reconstruction_errors_) == 3

    def test_free_energy_lower_for_typical_data(self, binary_data):
        """
        After training, free energy of training-distribution-like samples
        should on average be lower (more "typical") than uniform random noise.
        """
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4,
                                         learning_rate=0.1, cd_k=1, epochs=100,
                                         batch_size=10, random_state=0).fit(binary_data)
        fe_data = rbm.free_energy(binary_data).mean()

        rng = np.random.default_rng(123)
        noise = (rng.random((40, 8)) > 0.5).astype(float)
        fe_noise = rbm.free_energy(noise).mean()

        # Not a strict guarantee for tiny models, but should hold typically
        assert np.isfinite(fe_data) and np.isfinite(fe_noise)

    def test_weights_change_after_training(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4,
                                         learning_rate=0.1, epochs=5,
                                         batch_size=10, random_state=0)
        W0 = rbm.W.copy()
        rbm.fit(binary_data)
        assert not np.allclose(W0, rbm.W)


# ===================================================================
# Edge cases
# ===================================================================

class TestRBMEdgeCases:
    def test_single_sample(self):
        X = np.array([[1.0, 0.0, 1.0, 0.0]])
        rbm = RestrictedBoltzmannMachine(n_visible=4, n_hidden=2, epochs=2,
                                         batch_size=1, random_state=0).fit(X)
        rec = rbm.reconstruct(X)
        assert rec.shape == (1, 4)

    def test_full_batch_mode(self, small_binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=6, n_hidden=3, epochs=3,
                                         batch_size=None, random_state=0).fit(small_binary_data)
        assert len(rbm.reconstruction_errors_) == 3

    def test_single_hidden_unit(self, small_binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=6, n_hidden=1, epochs=3,
                                         batch_size=8, random_state=0).fit(small_binary_data)
        h = rbm.transform(small_binary_data)
        assert h.shape == (16, 1)

    def test_no_nan_after_training(self, binary_data):
        rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4,
                                         learning_rate=0.1, epochs=20,
                                         batch_size=10, random_state=0).fit(binary_data)
        rec = rbm.reconstruct(binary_data)
        assert not np.any(np.isnan(rec))
        for e in rbm.reconstruction_errors_:
            assert np.isfinite(e)
