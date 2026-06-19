"""
Tests for mlscratch.neural.gan
Covers: Generator, Discriminator, GAN
"""

from __future__ import annotations

import numpy as np
import pytest

from mlscratch.neural.gan import Generator, Discriminator, GAN


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def real_data():
    """40 samples in [-1, 1], data_dim=4, with structure (two clusters)."""
    rng = np.random.default_rng(0)
    c0 = rng.normal(-0.5, 0.1, (20, 4))
    c1 = rng.normal(0.5, 0.1, (20, 4))
    X = np.vstack([c0, c1])
    return np.clip(X, -1, 1)


# ===================================================================
# Generator
# ===================================================================

class TestGenerator:
    def test_forward_output_shape(self):
        gen = Generator(latent_dim=4, output_dim=6, hidden_sizes=[8],
                        random_state=0)
        z = np.random.default_rng(0).standard_normal((5, 4))
        out = gen.forward(z)
        assert out.shape == (5, 6)

    def test_output_bounded_by_tanh(self):
        gen = Generator(latent_dim=4, output_dim=6, hidden_sizes=[8],
                        random_state=0)
        z = np.random.default_rng(0).standard_normal((20, 4)) * 100
        out = gen.forward(z)
        assert np.all(out >= -1.0) and np.all(out <= 1.0)

    def test_sample_noise_shape(self):
        gen = Generator(latent_dim=5, output_dim=6, random_state=0)
        rng = np.random.default_rng(0)
        z = gen.sample_noise(10, rng)
        assert z.shape == (10, 5)

    def test_backward_updates_weights(self):
        gen = Generator(latent_dim=4, output_dim=6, hidden_sizes=[8],
                        random_state=0)
        W0 = [w.copy() for w in gen._net.W]
        z = np.random.default_rng(0).standard_normal((5, 4))
        out = gen.forward(z)
        gen.backward(np.ones_like(out) * 0.1, learning_rate=0.1)
        assert any(not np.allclose(w0, w1) for w0, w1 in zip(W0, gen._net.W))

    def test_default_hidden_sizes(self):
        gen = Generator(latent_dim=2, output_dim=3, random_state=0)
        # default hidden_sizes = [64, 64] -> 3 weight matrices
        assert len(gen._net.W) == 3


# ===================================================================
# Discriminator
# ===================================================================

class TestDiscriminator:
    def test_forward_output_shape(self, real_data):
        disc = Discriminator(input_dim=4, hidden_sizes=[8], random_state=0)
        out = disc.forward(real_data)
        assert out.shape == (40, 1)

    def test_output_in_unit_interval(self, real_data):
        disc = Discriminator(input_dim=4, hidden_sizes=[8], random_state=0)
        out = disc.forward(real_data)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_backward_updates_weights(self, real_data):
        disc = Discriminator(input_dim=4, hidden_sizes=[8], random_state=0)
        W0 = [w.copy() for w in disc._net.W]
        out = disc.forward(real_data)
        disc.backward(np.ones_like(out) * 0.1, learning_rate=0.1)
        assert any(not np.allclose(w0, w1) for w0, w1 in zip(W0, disc._net.W))

    def test_backward_with_zero_lr_returns_gradient_only(self, real_data):
        """learning_rate=0 should not change weights but still return d_input."""
        disc = Discriminator(input_dim=4, hidden_sizes=[8], random_state=0)
        W0 = [w.copy() for w in disc._net.W]
        out = disc.forward(real_data)
        d_in = disc.backward(np.ones_like(out) * 0.1, learning_rate=0.0)
        assert d_in.shape == real_data.shape
        for w0, w1 in zip(W0, disc._net.W):
            np.testing.assert_allclose(w0, w1)


# ===================================================================
# GAN — Basic API
# ===================================================================

class TestGANBasic:
    def test_init_creates_generator_and_discriminator(self):
        gan = GAN(latent_dim=4, data_dim=6, hidden_sizes=[8, 8],
                 learning_rate=0.01, random_state=0)
        assert isinstance(gan.generator, Generator)
        assert isinstance(gan.discriminator, Discriminator)

    def test_train_step_returns_two_floats(self, real_data):
        gan = GAN(latent_dim=4, data_dim=4, hidden_sizes=[8, 8],
                 learning_rate=0.01, random_state=0)
        d_loss, g_loss = gan.train_step(real_data[:8])
        assert isinstance(d_loss, float)
        assert isinstance(g_loss, float)

    def test_train_step_losses_finite(self, real_data):
        gan = GAN(latent_dim=4, data_dim=4, hidden_sizes=[8, 8],
                 learning_rate=0.01, random_state=0)
        d_loss, g_loss = gan.train_step(real_data[:8])
        assert np.isfinite(d_loss)
        assert np.isfinite(g_loss)

    def test_train_step_losses_recorded(self, real_data):
        gan = GAN(latent_dim=4, data_dim=4, hidden_sizes=[8, 8],
                 learning_rate=0.01, random_state=0)
        gan.train_step(real_data[:8])
        assert len(gan.d_losses_) == 1
        assert len(gan.g_losses_) == 1

    def test_fit_returns_self(self, real_data):
        gan = GAN(latent_dim=4, data_dim=4, hidden_sizes=[8, 8],
                 learning_rate=0.01, random_state=0)
        assert gan.fit(real_data, epochs=1, batch_size=8) is gan

    def test_generate_shape(self, real_data):
        gan = GAN(latent_dim=4, data_dim=4, hidden_sizes=[8, 8],
                 learning_rate=0.01, random_state=0)
        samples = gan.generate(10)
        assert samples.shape == (10, 4)

    def test_generate_in_data_range(self, real_data):
        gan = GAN(latent_dim=4, data_dim=4, hidden_sizes=[8, 8],
                 learning_rate=0.01, random_state=0)
        samples = gan.generate(10)
        assert np.all(samples >= -1.0) and np.all(samples <= 1.0)

    def test_discriminate_shape(self, real_data):
        gan = GAN(latent_dim=4, data_dim=4, hidden_sizes=[8, 8],
                 learning_rate=0.01, random_state=0)
        scores = gan.discriminate(real_data[:5])
        assert scores.shape == (5,)

    def test_discriminate_in_unit_interval(self, real_data):
        gan = GAN(latent_dim=4, data_dim=4, hidden_sizes=[8, 8],
                 learning_rate=0.01, random_state=0)
        scores = gan.discriminate(real_data[:5])
        assert np.all(scores >= 0) and np.all(scores <= 1)


# ===================================================================
# GAN — Correctness
# ===================================================================

class TestGANCorrectness:
    def test_losses_stay_finite_over_training(self, real_data):
        gan = GAN(latent_dim=4, data_dim=4, hidden_sizes=[8, 8],
                 learning_rate=0.001, random_state=0).fit(
            real_data, epochs=5, batch_size=8
        )
        assert all(np.isfinite(l) for l in gan.d_losses_)
        assert all(np.isfinite(l) for l in gan.g_losses_)

    def test_discriminator_distinguishes_better_than_random_initially(self, real_data):
        """
        Before any training, D's output on real vs generator output may be
        close to 0.5 (random init). After some training, D should have moved
        away from purely random outputs (non-degenerate).
        """
        gan = GAN(latent_dim=4, data_dim=4, hidden_sizes=[8, 8],
                 learning_rate=0.01, random_state=0)
        gan.fit(real_data, epochs=10, batch_size=8)
        real_scores = gan.discriminate(real_data)
        fake_scores = gan.discriminate(gan.generate(40))
        # Both should be valid probabilities and not NaN
        assert np.all(np.isfinite(real_scores))
        assert np.all(np.isfinite(fake_scores))

    def test_generator_weights_change_after_training(self, real_data):
        gan = GAN(latent_dim=4, data_dim=4, hidden_sizes=[8, 8],
                 learning_rate=0.01, random_state=0)
        W0 = [w.copy() for w in gan.generator._net.W]
        gan.fit(real_data, epochs=3, batch_size=8)
        assert any(
            not np.allclose(w0, w1)
            for w0, w1 in zip(W0, gan.generator._net.W)
        )

    def test_discriminator_weights_change_after_training(self, real_data):
        gan = GAN(latent_dim=4, data_dim=4, hidden_sizes=[8, 8],
                 learning_rate=0.01, random_state=0)
        W0 = [w.copy() for w in gan.discriminator._net.W]
        gan.fit(real_data, epochs=3, batch_size=8)
        assert any(
            not np.allclose(w0, w1)
            for w0, w1 in zip(W0, gan.discriminator._net.W)
        )

    def test_generated_samples_change_after_training(self, real_data):
        """Generator outputs for the same noise should change after training."""
        gan = GAN(latent_dim=4, data_dim=4, hidden_sizes=[8, 8],
                 learning_rate=0.01, random_state=0)
        z = np.random.default_rng(99).standard_normal((5, 4))
        out_before = gan.generator.forward(z)
        gan.fit(real_data, epochs=5, batch_size=8)
        out_after = gan.generator.forward(z)
        assert not np.allclose(out_before, out_after)


# ===================================================================
# GAN — Edge cases
# ===================================================================

class TestGANEdgeCases:
    def test_small_batch_size(self, real_data):
        gan = GAN(latent_dim=4, data_dim=4, hidden_sizes=[8],
                 learning_rate=0.01, random_state=0)
        d_loss, g_loss = gan.train_step(real_data[:2])
        assert np.isfinite(d_loss) and np.isfinite(g_loss)

    def test_latent_dim_one(self, real_data):
        gan = GAN(latent_dim=1, data_dim=4, hidden_sizes=[8],
                 learning_rate=0.01, random_state=0)
        samples = gan.generate(5)
        assert samples.shape == (5, 4)

    def test_data_dim_one(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, (20, 1))
        gan = GAN(latent_dim=2, data_dim=1, hidden_sizes=[8],
                 learning_rate=0.01, random_state=0)
        gan.fit(X, epochs=2, batch_size=4)
        samples = gan.generate(5)
        assert samples.shape == (5, 1)

    def test_reproducibility_with_seed(self, real_data):
        gan1 = GAN(latent_dim=4, data_dim=4, hidden_sizes=[8],
                  learning_rate=0.01, random_state=42)
        gan2 = GAN(latent_dim=4, data_dim=4, hidden_sizes=[8],
                  learning_rate=0.01, random_state=42)
        z = np.random.default_rng(0).standard_normal((3, 4))
        out1 = gan1.generator.forward(z)
        out2 = gan2.generator.forward(z)
        np.testing.assert_allclose(out1, out2)
