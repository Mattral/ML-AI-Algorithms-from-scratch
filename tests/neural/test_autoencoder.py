"""
Tests for mlscratch.neural.autoencoder
Covers: Autoencoder, DenoisingAutoencoder, VariationalAutoencoder
"""

from __future__ import annotations

import numpy as np
import pytest

from mlscratch.neural.autoencoder import (
    Autoencoder,
    DenoisingAutoencoder,
    VariationalAutoencoder,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def data():
    """40 samples, 8 features, low-rank structure (compressible)."""
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((40, 3))
    projection = rng.standard_normal((3, 8))
    return latent @ projection + rng.normal(0, 0.05, (40, 8))


@pytest.fixture
def small_data():
    rng = np.random.default_rng(1)
    return rng.standard_normal((20, 4))


# ===================================================================
# Autoencoder — Basic API
# ===================================================================

class TestAutoencoderBasic:
    def test_fit_returns_self(self, data):
        ae = Autoencoder(input_size=8, hidden_sizes=[6, 4], epochs=3,
                         batch_size=8, random_state=0)
        assert ae.fit(data) is ae

    def test_encoder_weight_count(self, data):
        ae = Autoencoder(input_size=8, hidden_sizes=[6, 4, 2], epochs=1,
                         batch_size=8, random_state=0).fit(data)
        assert len(ae._enc_W) == 3

    def test_encode_shape(self, data):
        ae = Autoencoder(input_size=8, hidden_sizes=[6, 3], epochs=3,
                         batch_size=8, random_state=0).fit(data)
        code = ae.encode(data)
        assert code.shape == (40, 3)

    def test_decode_shape(self, data):
        ae = Autoencoder(input_size=8, hidden_sizes=[6, 3], epochs=3,
                         batch_size=8, random_state=0).fit(data)
        code = ae.encode(data)
        rec = ae.decode(code)
        assert rec.shape == (40, 8)

    def test_reconstruct_shape(self, data):
        ae = Autoencoder(input_size=8, hidden_sizes=[6, 3], epochs=3,
                         batch_size=8, random_state=0).fit(data)
        rec = ae.reconstruct(data)
        assert rec.shape == data.shape

    def test_reconstruction_error_shape(self, data):
        ae = Autoencoder(input_size=8, hidden_sizes=[6, 3], epochs=3,
                         batch_size=8, random_state=0).fit(data)
        err = ae.reconstruction_error(data)
        assert err.shape == (40,)

    def test_reconstruction_error_non_negative(self, data):
        ae = Autoencoder(input_size=8, hidden_sizes=[6, 3], epochs=3,
                         batch_size=8, random_state=0).fit(data)
        err = ae.reconstruction_error(data)
        assert np.all(err >= 0)

    def test_losses_recorded(self, data):
        ae = Autoencoder(input_size=8, hidden_sizes=[6, 3], epochs=5,
                         batch_size=8, random_state=0).fit(data)
        assert len(ae.losses_) == 5

    def test_decode_output_non_negative(self, data):
        """ReLU decoder output is always >= 0."""
        ae = Autoencoder(input_size=8, hidden_sizes=[6, 3], epochs=3,
                         batch_size=8, random_state=0).fit(data)
        rec = ae.reconstruct(data)
        assert np.all(rec >= 0)


# ===================================================================
# Autoencoder — Correctness
# ===================================================================

class TestAutoencoderCorrectness:
    def test_loss_decreases_with_training(self, data):
        ae = Autoencoder(input_size=8, hidden_sizes=[6, 3], epochs=30,
                         batch_size=8, learning_rate=0.01, random_state=0).fit(data)
        assert ae.losses_[-1] < ae.losses_[0]

    def test_reconstruction_improves_with_training(self, data):
        ae_short = Autoencoder(input_size=8, hidden_sizes=[6, 3], epochs=2,
                               batch_size=8, learning_rate=0.01, random_state=0).fit(data)
        ae_long  = Autoencoder(input_size=8, hidden_sizes=[6, 3], epochs=50,
                               batch_size=8, learning_rate=0.01, random_state=0).fit(data)
        err_short = ae_short.reconstruction_error(data).mean()
        err_long  = ae_long.reconstruction_error(data).mean()
        assert err_long <= err_short

    def test_anomaly_has_higher_reconstruction_error(self, data):
        """A point far outside the training distribution should have
        higher reconstruction error than in-distribution points."""
        ae = Autoencoder(input_size=8, hidden_sizes=[6, 3], epochs=40,
                         batch_size=8, learning_rate=0.01, random_state=0).fit(data)
        normal_err = ae.reconstruction_error(data).mean()

        anomaly = np.ones((1, 8)) * 100.0
        anomaly_err = ae.reconstruction_error(anomaly)[0]
        assert anomaly_err > normal_err


# ===================================================================
# Autoencoder — Edge cases
# ===================================================================

class TestAutoencoderEdgeCases:
    def test_single_sample(self):
        X = np.random.default_rng(0).standard_normal((1, 6))
        ae = Autoencoder(input_size=6, hidden_sizes=[4, 2], epochs=2,
                         batch_size=1, random_state=0).fit(X)
        rec = ae.reconstruct(X)
        assert rec.shape == (1, 6)

    def test_single_hidden_layer(self, small_data):
        ae = Autoencoder(input_size=4, hidden_sizes=[2], epochs=3,
                         batch_size=8, random_state=0).fit(small_data)
        code = ae.encode(small_data)
        assert code.shape == (20, 2)

    def test_full_batch_mode(self, small_data):
        ae = Autoencoder(input_size=4, hidden_sizes=[2], epochs=3,
                         batch_size=None, random_state=0).fit(small_data)
        assert len(ae.losses_) == 3


# ===================================================================
# DenoisingAutoencoder
# ===================================================================

class TestDenoisingAutoencoder:
    def test_invalid_noise_type_raises(self):
        with pytest.raises(ValueError):
            DenoisingAutoencoder(input_size=8, noise_type="invalid")

    def test_gaussian_noise_runs(self, data):
        dae = DenoisingAutoencoder(
            input_size=8, hidden_sizes=[6, 3], noise_type="gaussian",
            noise_level=0.1, epochs=5, batch_size=8, random_state=0
        ).fit(data)
        rec = dae.reconstruct(data)
        assert rec.shape == data.shape

    def test_dropout_noise_runs(self, data):
        dae = DenoisingAutoencoder(
            input_size=8, hidden_sizes=[6, 3], noise_type="dropout",
            noise_level=0.2, epochs=5, batch_size=8, random_state=0
        ).fit(data)
        rec = dae.reconstruct(data)
        assert rec.shape == data.shape

    def test_corrupt_gaussian_adds_noise(self, data):
        dae = DenoisingAutoencoder(
            input_size=8, noise_type="gaussian", noise_level=0.5, random_state=0
        )
        corrupted = dae._corrupt(data)
        assert not np.array_equal(corrupted, data)

    def test_corrupt_dropout_zeros_entries(self, data):
        dae = DenoisingAutoencoder(
            input_size=8, noise_type="dropout", noise_level=0.9, random_state=0
        )
        corrupted = dae._corrupt(data)
        # With 90% drop rate, many entries should be exactly 0
        assert (corrupted == 0).mean() > 0.5

    def test_loss_decreases(self, data):
        dae = DenoisingAutoencoder(
            input_size=8, hidden_sizes=[6, 3], noise_type="gaussian",
            noise_level=0.05, epochs=30, batch_size=8,
            learning_rate=0.01, random_state=0
        ).fit(data)
        assert dae.losses_[-1] < dae.losses_[0]

    def test_inherits_from_autoencoder(self, data):
        dae = DenoisingAutoencoder(input_size=8, hidden_sizes=[6, 3],
                                   epochs=2, batch_size=8, random_state=0).fit(data)
        assert isinstance(dae, Autoencoder)
        # Inherited methods work
        err = dae.reconstruction_error(data)
        assert err.shape == (40,)


# ===================================================================
# VariationalAutoencoder — Basic API
# ===================================================================

class TestVAEBasic:
    def test_fit_returns_self(self, data):
        vae = VariationalAutoencoder(input_size=8, hidden_size=16, latent_dim=3,
                                     epochs=2, batch_size=8, random_state=0)
        assert vae.fit(data) is vae

    def test_encode_shapes(self, data):
        vae = VariationalAutoencoder(input_size=8, hidden_size=16, latent_dim=3,
                                     epochs=2, batch_size=8, random_state=0).fit(data)
        mu, log_var = vae.encode(data)
        assert mu.shape == (40, 3)
        assert log_var.shape == (40, 3)

    def test_decode_shape(self, data):
        vae = VariationalAutoencoder(input_size=8, hidden_size=16, latent_dim=3,
                                     epochs=2, batch_size=8, random_state=0).fit(data)
        z = np.random.default_rng(0).standard_normal((5, 3))
        rec = vae.decode(z)
        assert rec.shape == (5, 8)

    def test_reconstruct_shape(self, data):
        vae = VariationalAutoencoder(input_size=8, hidden_size=16, latent_dim=3,
                                     epochs=2, batch_size=8, random_state=0).fit(data)
        rec = vae.reconstruct(data)
        assert rec.shape == data.shape

    def test_sample_shape(self, data):
        vae = VariationalAutoencoder(input_size=8, hidden_size=16, latent_dim=3,
                                     epochs=2, batch_size=8, random_state=0).fit(data)
        samples = vae.sample(10)
        assert samples.shape == (10, 8)

    def test_losses_recorded(self, data):
        vae = VariationalAutoencoder(input_size=8, hidden_size=16, latent_dim=3,
                                     epochs=5, batch_size=8, random_state=0).fit(data)
        assert len(vae.losses_) == 5

    def test_log_var_clamped(self, data):
        """log_var must be clipped to [-10, 10] for numerical stability."""
        vae = VariationalAutoencoder(input_size=8, hidden_size=16, latent_dim=3,
                                     epochs=2, batch_size=8, random_state=0).fit(data)
        _, log_var = vae.encode(data)
        assert np.all(log_var >= -10) and np.all(log_var <= 10)


# ===================================================================
# VariationalAutoencoder — Correctness
# ===================================================================

class TestVAECorrectness:
    def test_loss_decreases(self, data):
        vae = VariationalAutoencoder(input_size=8, hidden_size=16, latent_dim=3,
                                     epochs=30, batch_size=8, beta=0.1,
                                     learning_rate=0.01, random_state=0).fit(data)
        assert vae.losses_[-1] < vae.losses_[0]

    def test_reconstruction_uses_mean_not_sample(self, data):
        """reconstruct() should be deterministic (uses μ, not a sample)."""
        vae = VariationalAutoencoder(input_size=8, hidden_size=16, latent_dim=3,
                                     epochs=5, batch_size=8, random_state=0).fit(data)
        rec1 = vae.reconstruct(data)
        rec2 = vae.reconstruct(data)
        np.testing.assert_allclose(rec1, rec2)

    def test_samples_differ(self, data):
        """sample() draws fresh z each call → outputs should differ."""
        vae = VariationalAutoencoder(input_size=8, hidden_size=16, latent_dim=3,
                                     epochs=5, batch_size=8, random_state=0).fit(data)
        s1 = vae.sample(5)
        s2 = vae.sample(5)
        assert not np.allclose(s1, s2)

    def test_beta_zero_reduces_to_plain_autoencoder_loss(self, data):
        """With beta=0, only reconstruction loss should matter (KL ignored)."""
        vae = VariationalAutoencoder(input_size=8, hidden_size=16, latent_dim=3,
                                     beta=0.0, epochs=20, batch_size=8,
                                     learning_rate=0.01, random_state=0).fit(data)
        assert vae.losses_[-1] < vae.losses_[0]

    def test_higher_beta_increases_kl_pressure(self, data):
        """With very high beta, μ should be pulled toward 0 (prior mean)."""
        vae_low  = VariationalAutoencoder(input_size=8, hidden_size=16, latent_dim=3,
                                          beta=0.01, epochs=30, batch_size=8,
                                          learning_rate=0.01, random_state=0).fit(data)
        vae_high = VariationalAutoencoder(input_size=8, hidden_size=16, latent_dim=3,
                                          beta=10.0, epochs=30, batch_size=8,
                                          learning_rate=0.01, random_state=0).fit(data)
        mu_low,  _ = vae_low.encode(data)
        mu_high, _ = vae_high.encode(data)
        assert np.abs(mu_high).mean() <= np.abs(mu_low).mean() + 1e-6


# ===================================================================
# VariationalAutoencoder — Edge cases
# ===================================================================

class TestVAEEdgeCases:
    def test_latent_dim_one(self, data):
        vae = VariationalAutoencoder(input_size=8, hidden_size=8, latent_dim=1,
                                     epochs=2, batch_size=8, random_state=0).fit(data)
        mu, log_var = vae.encode(data)
        assert mu.shape == (40, 1)

    def test_single_sample(self):
        X = np.random.default_rng(0).standard_normal((1, 6))
        vae = VariationalAutoencoder(input_size=6, hidden_size=8, latent_dim=2,
                                     epochs=2, batch_size=1, random_state=0).fit(X)
        rec = vae.reconstruct(X)
        assert rec.shape == (1, 6)

    def test_no_nan_after_training(self, data):
        vae = VariationalAutoencoder(input_size=8, hidden_size=16, latent_dim=3,
                                     epochs=20, batch_size=8, random_state=0).fit(data)
        rec = vae.reconstruct(data)
        assert not np.any(np.isnan(rec))
        for l in vae.losses_:
            assert np.isfinite(l)
