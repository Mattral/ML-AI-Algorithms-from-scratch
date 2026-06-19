"""
Generative Adversarial Network (GAN)
======================================
Two networks trained adversarially (Goodfellow et al., 2014):

    Generator     G: z → x̂        maps noise to fake samples
    Discriminator D: x → [0,1]     estimates P(x is real)

Minimax objective
------------------
    min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]

In practice the generator is trained with the non-saturating loss:
    max_G  E_z[log D(G(z))]

Training loop (per batch)
---------------------------
1. Sample real batch x ~ data, noise z ~ N(0,I)
2. Generate fakes:  x̂ = G(z)
3. Update D to maximise log D(x) + log(1 - D(x̂))
4. Update G to maximise log D(x̂)        (non-saturating)

Both G and D are simple MLPs (Linear → ReLU/Tanh/Sigmoid),
implemented with manual forward/backward passes.

Reference
----------
Goodfellow et al. (2014). Generative Adversarial Networks. NeurIPS.

Only numpy is used.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# Activations
# ============================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


# ============================================================
# Simple MLP block (used for both G and D)
# ============================================================

class _MLPBlock:
    """A minimal MLP with manual forward/backward for GAN sub-networks."""

    def __init__(
        self,
        layer_sizes: list[int],
        output_activation: str,
        rng: np.random.Generator,
    ) -> None:
        self.layer_sizes = layer_sizes
        self.output_activation = output_activation

        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.W.append(rng.normal(0, scale, (layer_sizes[i], layer_sizes[i + 1])))
            self.b.append(np.zeros(layer_sizes[i + 1]))

        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = x
        zs, acts = [], [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            zs.append(z)
            if i < len(self.W) - 1:
                a = _relu(z)
            else:
                if self.output_activation == "sigmoid":
                    a = _sigmoid(z)
                elif self.output_activation == "tanh":
                    a = _tanh(z)
                else:
                    a = z
            acts.append(a)
        self._cache = {"zs": zs, "acts": acts}
        return a

    def backward(self, d_out: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backprop d_out through the network; returns gradient w.r.t. input."""
        zs, acts = self._cache["zs"], self._cache["acts"]
        n = len(d_out)

        # Output layer activation gradient
        if self.output_activation == "sigmoid":
            delta = d_out * acts[-1] * (1 - acts[-1])
        elif self.output_activation == "tanh":
            delta = d_out * _tanh_grad(zs[-1])
        else:
            delta = d_out

        for i in reversed(range(len(self.W))):
            dW = acts[i].T @ delta / n
            db = delta.mean(axis=0)

            self.W[i] -= learning_rate * dW
            self.b[i] -= learning_rate * db

            if i > 0:
                delta = (delta @ self.W[i].T) * _relu_grad(zs[i - 1])
            else:
                d_input = delta @ self.W[i].T

        return d_input


# ============================================================
# Generator / Discriminator
# ============================================================

class Generator:
    """
    Generator network: noise → fake data.

    Architecture: Linear → ReLU → ... → Linear → Tanh
    (tanh assumes data is scaled to [-1, 1]).

    Parameters
    ----------
    latent_dim : int
    output_dim : int
    hidden_sizes : list[int]
    random_state : int or None
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_sizes: list[int] | None = None,
        random_state: int | None = None,
    ) -> None:
        hidden_sizes = hidden_sizes or [64, 64]
        rng = np.random.default_rng(random_state)
        self.latent_dim = latent_dim
        self._net = _MLPBlock(
            [latent_dim] + hidden_sizes + [output_dim],
            output_activation="tanh",
            rng=rng,
        )

    def forward(self, z: np.ndarray) -> np.ndarray:
        """z : (B, latent_dim) → (B, output_dim)"""
        return self._net.forward(z)

    def backward(self, d_out: np.ndarray, learning_rate: float) -> np.ndarray:
        return self._net.backward(d_out, learning_rate)

    def sample_noise(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample n latent vectors ~ N(0,I)."""
        return rng.standard_normal((n, self.latent_dim))


class Discriminator:
    """
    Discriminator network: data → P(real).

    Architecture: Linear → ReLU → ... → Linear → Sigmoid

    Parameters
    ----------
    input_dim : int
    hidden_sizes : list[int]
    random_state : int or None
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int] | None = None,
        random_state: int | None = None,
    ) -> None:
        hidden_sizes = hidden_sizes or [64, 64]
        rng = np.random.default_rng(random_state)
        self._net = _MLPBlock(
            [input_dim] + hidden_sizes + [1],
            output_activation="sigmoid",
            rng=rng,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x : (B, input_dim) → (B, 1) probability of being real."""
        return self._net.forward(x)

    def backward(self, d_out: np.ndarray, learning_rate: float) -> np.ndarray:
        return self._net.backward(d_out, learning_rate)


# ============================================================
# GAN — training orchestration
# ============================================================

class GAN:
    """
    Generative Adversarial Network — orchestrates G and D training.

    Parameters
    ----------
    latent_dim : int
    data_dim : int
    hidden_sizes : list[int]
    learning_rate : float
    random_state : int or None
    """

    def __init__(
        self,
        latent_dim: int,
        data_dim: int,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 1e-3,
        random_state: int | None = None,
    ) -> None:
        self.latent_dim    = latent_dim
        self.data_dim      = data_dim
        self.learning_rate = learning_rate
        self._rng          = np.random.default_rng(random_state)

        self.generator     = Generator(latent_dim, data_dim, hidden_sizes, random_state)
        self.discriminator = Discriminator(data_dim, hidden_sizes,
                                            (random_state or 0) + 1)

        self.d_losses_: list[float] = []
        self.g_losses_: list[float] = []

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def train_step(self, real_batch: np.ndarray) -> tuple[float, float]:
        """
        Run one D-update followed by one G-update.

        Parameters
        ----------
        real_batch : ndarray of shape (batch_size, data_dim)
            Real data scaled to [-1, 1] (to match the generator's tanh output).

        Returns
        -------
        (d_loss, g_loss) : tuple[float, float]
        """
        eps = 1e-8
        B   = len(real_batch)

        # ── 1. Discriminator update ─────────────────────────────────
        z       = self.generator.sample_noise(B, self._rng)
        fake    = self.generator.forward(z)

        d_real  = self.discriminator.forward(real_batch)
        d_fake  = self.discriminator.forward(fake)

        d_loss  = float(-np.mean(np.log(d_real + eps) + np.log(1 - d_fake + eps)))

        # Gradients for D: maximise log(d_real) + log(1 - d_fake)
        # ⇒ minimise -log(d_real) - log(1 - d_fake)
        grad_real = -(1.0 / (d_real + eps)) / B
        grad_fake =  (1.0 / (1 - d_fake + eps)) / B

        # Backprop through D for real batch (don't propagate into G)
        self.discriminator.forward(real_batch)   # refresh cache
        self.discriminator.backward(grad_real, self.learning_rate)

        # Backprop through D for fake batch (don't propagate into G here)
        self.discriminator.forward(fake)          # refresh cache
        self.discriminator.backward(grad_fake, self.learning_rate)

        # ── 2. Generator update (non-saturating loss) ────────────────
        z2       = self.generator.sample_noise(B, self._rng)
        fake2    = self.generator.forward(z2)
        d_fake2  = self.discriminator.forward(fake2)

        g_loss   = float(-np.mean(np.log(d_fake2 + eps)))

        # dG_loss/d(d_fake2) = -1/d_fake2
        grad_g_out = -(1.0 / (d_fake2 + eps)) / B
        # Backprop through D (no weight update) to get gradient w.r.t. fake2
        d_input_to_D = self.discriminator.backward(grad_g_out, learning_rate=0.0)
        # Backprop that gradient through G (updates G's weights)
        self.generator.backward(d_input_to_D, self.learning_rate)

        self.d_losses_.append(d_loss)
        self.g_losses_.append(g_loss)
        return d_loss, g_loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> "GAN":
        """
        Train the GAN on dataset X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, data_dim)
            Should be scaled to [-1, 1].
        epochs : int
        batch_size : int

        Returns
        -------
        self
        """
        n = len(X)

        for _ in range(epochs):
            idx = self._rng.permutation(n)
            for start in range(0, n, batch_size):
                mb = idx[start:start + batch_size]
                self.train_step(X[mb])

        return self

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate n_samples fake samples.

        Returns
        -------
        ndarray of shape (n_samples, data_dim), values in [-1, 1]
        """
        z = self.generator.sample_noise(n_samples, self._rng)
        return self.generator.forward(z)

    def discriminate(self, X: np.ndarray) -> np.ndarray:
        """Return D(X) — probability each sample is real."""
        return self.discriminator.forward(X).ravel()
