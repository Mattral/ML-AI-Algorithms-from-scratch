"""
Autoencoder Variants
=====================
Unsupervised representation-learning networks that compress input data
through a bottleneck and reconstruct it:

    Encoder:  X  →  h = f(WₑX + bₑ)          (latent code)
    Decoder:  h  →  X̂ = g(WdX + bd)           (reconstruction)
    Loss:     MSE(X, X̂)

Three variants are implemented
--------------------------------
Autoencoder
    Vanilla tied-weight autoencoder.  Encoder and decoder share transposed
    weights for parameter efficiency and implicit regularisation.

DenoisingAutoencoder
    Corrupts inputs with additive Gaussian or Bernoulli dropout noise
    before encoding, forcing the network to learn robust features.
    Otherwise identical API to ``Autoencoder``.

VariationalAutoencoder
    Learns a *distribution* over the latent space rather than a point
    estimate.  The encoder outputs (μ, log σ²) and samples via the
    reparameterisation trick:
        z = μ + σ ε,  ε ~ N(0,I)
    Loss:  Reconstruction (BCE or MSE) + KL[N(μ,σ²) ‖ N(0,I)]
        KL = −½ Σ(1 + log σ² − μ² − σ²)

References
----------
Hinton & Salakhutdinov (2006). Reducing the dimensionality of data with
neural networks. Science, 313(5786), 504-507.

Vincent et al. (2008). Extracting and composing robust features with
denoising autoencoders. ICML.

Kingma & Welling (2013). Auto-encoding variational Bayes. ICLR 2014.

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


# ============================================================
# Vanilla Autoencoder
# ============================================================

class Autoencoder:
    """
    Vanilla Autoencoder with tied encoder / decoder weights.

    Architecture:  X → [Linear → ReLU] × n_hidden_layers → code
                   code → [Linear → ReLU] × n_hidden_layers → X̂

    Parameters
    ----------
    input_size : int
    hidden_sizes : list[int]
        Sizes of hidden layers in the encoder.  The decoder mirrors them
        in reverse.  The last element is the code (bottleneck) dimension.
    learning_rate : float
    epochs : int
    batch_size : int or None
    random_state : int or None
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int | None = 64,
        random_state: int | None = None,
    ) -> None:
        self.input_size    = input_size
        self.hidden_sizes  = hidden_sizes or [64, 32]
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.batch_size    = batch_size
        self._rng          = np.random.default_rng(random_state)

        # Built during fit
        self._enc_W: list[np.ndarray] = []
        self._enc_b: list[np.ndarray] = []
        self._dec_b: list[np.ndarray] = []
        self.losses_: list[float]     = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        sizes = [self.input_size] + list(self.hidden_sizes)
        self._enc_W = []
        self._enc_b = []
        self._dec_b = []
        for i in range(len(sizes) - 1):
            fan_in  = sizes[i]
            fan_out = sizes[i + 1]
            scale   = np.sqrt(2.0 / fan_in)
            self._enc_W.append(self._rng.normal(0, scale, (fan_in, fan_out)))
            self._enc_b.append(np.zeros(fan_out))
            self._dec_b.append(np.zeros(fan_in))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _encode(self, X: np.ndarray) -> tuple[list, list]:
        """Return (pre_acts, activations) for encoder."""
        pre_acts, acts = [], [X]
        a = X
        for W, b in zip(self._enc_W, self._enc_b):
            z = a @ W + b
            pre_acts.append(z)
            a = _relu(z)
            acts.append(a)
        return pre_acts, acts

    def _decode(self, code: np.ndarray) -> tuple[list, list]:
        """Return (pre_acts, activations) for decoder (tied weights)."""
        pre_acts, acts = [], [code]
        a = code
        for W, b in zip(reversed(self._enc_W), reversed(self._dec_b)):
            z = a @ W.T + b
            pre_acts.append(z)
            a = _relu(z)
            acts.append(a)
        return pre_acts, acts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "Autoencoder":
        """
        Train autoencoder to reconstruct X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, input_size)

        Returns
        -------
        self
        """
        self._build()
        n_samples = len(X)
        bs        = self.batch_size or n_samples
        self.losses_ = []

        for _ in range(self.epochs):
            idx       = self._rng.permutation(n_samples)
            epoch_loss = 0.0

            for start in range(0, n_samples, bs):
                mb = idx[start:start + bs]
                Xb = X[mb]

                # Forward
                enc_pre, enc_acts = self._encode(Xb)
                code              = enc_acts[-1]
                dec_pre, dec_acts = self._decode(code)
                X_hat             = dec_acts[-1]

                # Loss
                loss       = float(np.mean((X_hat - Xb) ** 2))
                epoch_loss += loss

                # Backward — decoder first
                delta = 2.0 * (X_hat - Xb) / len(mb)
                for i, (dec_z, dec_a_prev) in enumerate(
                    zip(reversed(dec_pre), reversed(dec_acts[:-1]))
                ):
                    delta = delta * _relu_grad(dec_z)
                    layer_idx = i
                    W      = self._enc_W[layer_idx]
                    dW     = delta.T @ dec_a_prev               # == W.shape
                    db_dec = delta.mean(axis=0)
                    self._enc_W[layer_idx] -= self.learning_rate * dW
                    self._dec_b[layer_idx] -= self.learning_rate * db_dec
                    delta = delta @ W                            # propagate back

                # Continue through encoder
                for i in reversed(range(len(self._enc_W))):
                    delta = delta * _relu_grad(enc_pre[i])
                    db_enc = delta.mean(axis=0)
                    self._enc_b[i] -= self.learning_rate * db_enc
                    if i > 0:
                        delta = delta @ self._enc_W[i].T

            self.losses_.append(epoch_loss / max(1, n_samples // bs))

        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Compress X to latent codes."""
        _, acts = self._encode(X)
        return acts[-1]

    def decode(self, code: np.ndarray) -> np.ndarray:
        """Reconstruct from latent codes."""
        _, acts = self._decode(code)
        return acts[-1]

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Encode then decode X."""
        return self.decode(self.encode(X))

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Per-sample MSE reconstruction error.  Useful for anomaly detection:
        high error ≈ anomalous sample.

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        X_hat = self.reconstruct(X)
        return np.mean((X - X_hat) ** 2, axis=1)


# ============================================================
# Denoising Autoencoder
# ============================================================

class DenoisingAutoencoder(Autoencoder):
    """
    Denoising Autoencoder — adds noise to inputs before encoding.

    Parameters
    ----------
    noise_type : str
        ``'gaussian'`` — adds N(0, noise_level²) noise.
        ``'dropout'``  — randomly zeros out inputs with probability noise_level.
    noise_level : float
        Std dev for Gaussian noise, or drop probability for dropout noise.
    All other parameters : see ``Autoencoder``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        noise_type: str = "gaussian",
        noise_level: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int | None = 64,
        random_state: int | None = None,
    ) -> None:
        super().__init__(input_size, hidden_sizes, learning_rate,
                         epochs, batch_size, random_state)
        if noise_type not in {"gaussian", "dropout"}:
            raise ValueError("noise_type must be 'gaussian' or 'dropout'.")
        self.noise_type  = noise_type
        self.noise_level = noise_level

    def _corrupt(self, X: np.ndarray) -> np.ndarray:
        if self.noise_type == "gaussian":
            return X + self._rng.normal(0, self.noise_level, X.shape)
        # dropout
        mask = self._rng.random(X.shape) > self.noise_level
        return X * mask

    def fit(self, X: np.ndarray) -> "DenoisingAutoencoder":
        """Train on corrupted inputs, reconstruct clean targets."""
        self._build()
        n_samples = len(X)
        bs        = self.batch_size or n_samples
        self.losses_ = []

        for _ in range(self.epochs):
            idx       = self._rng.permutation(n_samples)
            epoch_loss = 0.0

            for start in range(0, n_samples, bs):
                mb       = idx[start:start + bs]
                Xb_clean = X[mb]
                Xb_noisy = self._corrupt(Xb_clean)

                enc_pre, enc_acts = self._encode(Xb_noisy)
                code              = enc_acts[-1]
                dec_pre, dec_acts = self._decode(code)
                X_hat             = dec_acts[-1]

                loss       = float(np.mean((X_hat - Xb_clean) ** 2))
                epoch_loss += loss

                # Identical backward to Autoencoder but against clean target
                delta = 2.0 * (X_hat - Xb_clean) / len(mb)
                for i, (dec_z, dec_a_prev) in enumerate(
                    zip(reversed(dec_pre), reversed(dec_acts[:-1]))
                ):
                    delta     = delta * _relu_grad(dec_z)
                    layer_idx = i
                    W         = self._enc_W[layer_idx]
                    dW        = delta.T @ dec_a_prev            # == W.shape
                    db_dec    = delta.mean(axis=0)
                    self._enc_W[layer_idx] -= self.learning_rate * dW
                    self._dec_b[layer_idx] -= self.learning_rate * db_dec
                    delta = delta @ W

                for i in reversed(range(len(self._enc_W))):
                    delta = delta * _relu_grad(enc_pre[i])
                    db_enc = delta.mean(axis=0)
                    self._enc_b[i] -= self.learning_rate * db_enc
                    if i > 0:
                        delta = delta @ self._enc_W[i].T

            self.losses_.append(epoch_loss / max(1, n_samples // bs))

        return self


# ============================================================
# Variational Autoencoder
# ============================================================

class VariationalAutoencoder:
    """
    Variational Autoencoder (VAE).

    Encoder outputs μ and log σ² for a Gaussian latent distribution.
    Samples via reparameterisation: z = μ + σ ε, ε ~ N(0,I).
    Decoder reconstructs X̂ from z.
    Loss = Reconstruction (MSE) + β * KL[N(μ,σ²) ‖ N(0,I)]

    Parameters
    ----------
    input_size : int
    hidden_size : int
        Size of the single hidden layer in both encoder and decoder.
    latent_dim : int
        Dimensionality of the latent space.
    beta : float
        KL weight (β-VAE: β > 1 encourages disentanglement).
    learning_rate : float
    epochs : int
    batch_size : int or None
    random_state : int or None
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        latent_dim: int = 8,
        beta: float = 1.0,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int | None = 64,
        random_state: int | None = None,
    ) -> None:
        self.input_size    = input_size
        self.hidden_size   = hidden_size
        self.latent_dim    = latent_dim
        self.beta          = beta
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.batch_size    = batch_size
        self._rng          = np.random.default_rng(random_state)

        self._init_params()
        self.losses_: list[float] = []

    # ------------------------------------------------------------------
    # Parameter initialisation
    # ------------------------------------------------------------------

    def _init_params(self) -> None:
        D, H, L = self.input_size, self.hidden_size, self.latent_dim
        s = lambda fi, fo: np.sqrt(2.0 / fi)

        # Encoder:  input → hidden
        self._We1 = self._rng.normal(0, s(D, H), (D, H))
        self._be1 = np.zeros(H)
        # Encoder:  hidden → μ
        self._Wmu = self._rng.normal(0, s(H, L), (H, L))
        self._bmu = np.zeros(L)
        # Encoder:  hidden → log σ²
        self._Wlv = self._rng.normal(0, s(H, L), (H, L))
        self._blv = np.zeros(L)

        # Decoder:  z → hidden
        self._Wd1 = self._rng.normal(0, s(L, H), (L, H))
        self._bd1 = np.zeros(H)
        # Decoder:  hidden → X̂
        self._Wd2 = self._rng.normal(0, s(H, D), (H, D))
        self._bd2 = np.zeros(D)

    # ------------------------------------------------------------------
    # Encoder / Decoder
    # ------------------------------------------------------------------

    def _encode(self, X: np.ndarray) -> tuple:
        h_enc = _relu(X @ self._We1 + self._be1)
        mu    = h_enc @ self._Wmu + self._bmu
        log_var = np.clip(h_enc @ self._Wlv + self._blv, -10, 10)
        return h_enc, mu, log_var

    def _reparameterise(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        eps = self._rng.standard_normal(mu.shape)
        return mu + np.exp(0.5 * log_var) * eps

    def _decode(self, z: np.ndarray) -> tuple:
        h_dec = _relu(z @ self._Wd1 + self._bd1)
        X_hat = h_dec @ self._Wd2 + self._bd2
        return h_dec, X_hat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "VariationalAutoencoder":
        """
        Train VAE on X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, input_size)
        """
        n_samples = len(X)
        bs        = self.batch_size or n_samples
        lr        = self.learning_rate
        self.losses_ = []

        for _ in range(self.epochs):
            idx        = self._rng.permutation(n_samples)
            epoch_loss = 0.0

            for start in range(0, n_samples, bs):
                mb   = idx[start:start + bs]
                Xb   = X[mb]
                n    = len(mb)

                # ── Forward ─────────────────────────────────────────
                h_enc, mu, log_var = self._encode(Xb)
                z                  = self._reparameterise(mu, log_var)
                h_dec, X_hat       = self._decode(z)

                # ── Loss ─────────────────────────────────────────────
                rec_loss = float(np.mean((X_hat - Xb) ** 2))
                kl_loss  = float(-0.5 * np.mean(
                    1 + log_var - mu ** 2 - np.exp(log_var)
                ))
                loss = rec_loss + self.beta * kl_loss
                epoch_loss += loss

                # ── Backward decoder ─────────────────────────────────
                d_Xhat  = 2.0 * (X_hat - Xb) / n
                d_Wd2   = h_dec.T @ d_Xhat
                d_bd2   = d_Xhat.mean(axis=0)
                d_hdec  = d_Xhat @ self._Wd2.T * _relu_grad(z @ self._Wd1 + self._bd1)
                d_Wd1   = z.T @ d_hdec
                d_bd1   = d_hdec.mean(axis=0)
                d_z     = d_hdec @ self._Wd1.T    # gradient w.r.t. z

                # ── Backward through reparameterise ──────────────────
                sigma   = np.exp(0.5 * log_var)
                d_mu    = d_z + self.beta * mu / n
                d_lv    = (d_z * sigma * 0.5
                           + self.beta * 0.5 * (np.exp(log_var) - 1) / n)

                # ── Backward encoder ─────────────────────────────────
                d_Wmu   = h_enc.T @ d_mu
                d_bmu   = d_mu.mean(axis=0)
                d_Wlv   = h_enc.T @ d_lv
                d_blv   = d_lv.mean(axis=0)
                d_henc  = (d_mu @ self._Wmu.T + d_lv @ self._Wlv.T) * \
                          _relu_grad(Xb @ self._We1 + self._be1)
                d_We1   = Xb.T @ d_henc
                d_be1   = d_henc.mean(axis=0)

                # ── Gradient clipping (prevents overflow) ─────────────
                _clip = 5.0
                for _g in [d_We1, d_be1, d_Wmu, d_bmu, d_Wlv, d_blv,
                            d_Wd1, d_bd1, d_Wd2, d_bd2]:
                    np.clip(_g, -_clip, _clip, out=_g)

                # ── Gradient descent ─────────────────────────────────
                self._We1  -= lr * d_We1
                self._be1  -= lr * d_be1
                self._Wmu  -= lr * d_Wmu
                self._bmu  -= lr * d_bmu
                self._Wlv  -= lr * d_Wlv
                self._blv  -= lr * d_blv
                self._Wd1  -= lr * d_Wd1
                self._bd1  -= lr * d_bd1
                self._Wd2  -= lr * d_Wd2
                self._bd2  -= lr * d_bd2

            self.losses_.append(epoch_loss / max(1, n_samples // bs))

        return self

    def encode(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (μ, log σ²) for each sample."""
        _, mu, log_var = self._encode(X)
        return mu, log_var

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent codes to reconstruction."""
        _, X_hat = self._decode(z)
        return X_hat

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Encode (use μ, no sampling) and decode."""
        _, mu, _ = self._encode(X)
        _, X_hat = self._decode(mu)
        return X_hat

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate new samples by sampling z ~ N(0,I) and decoding.

        Returns
        -------
        ndarray of shape (n_samples, input_size)
        """
        z = self._rng.standard_normal((n_samples, self.latent_dim))
        return self.decode(z)
