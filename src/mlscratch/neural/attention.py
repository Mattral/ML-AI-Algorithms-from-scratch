"""
Attention Mechanisms and Transformer
======================================
Building blocks of the Transformer architecture (Vaswani et al., 2017).

ScaledDotProductAttention
--------------------------
The core attention operation:

    Attention(Q, K, V) = softmax(QK^T / √d_k) V

MultiHeadAttention
-------------------
Splits Q, K, V into ``n_heads`` parallel attention computations,
concatenates results, and projects back to ``d_model``:

    head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    MHA(Q,K,V) = Concat(head_1, ..., head_h) W^O

PositionalEncoding
-------------------
Injects order information using sinusoids of varying frequency:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

LayerNorm
---------
Normalises across the feature dimension:

    LN(x) = γ (x - μ) / √(σ² + ε) + β

FeedForward
-----------
Two-layer MLP with ReLU, applied position-wise:

    FFN(x) = ReLU(x W1 + b1) W2 + b2

TransformerEncoderLayer / TransformerEncoder
----------------------------------------------
Standard encoder block: MHA → Add&Norm → FFN → Add&Norm, stacked
``n_layers`` times.

References
----------
Vaswani et al. (2017). Attention is all you need. NeurIPS.

Only numpy is used.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# Helpers
# ============================================================

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


# ============================================================
# Scaled Dot-Product Attention
# ============================================================

class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention (stateless — no learnable parameters).

    Attention(Q, K, V) = softmax(QK^T / √d_k + mask) V
    """

    def __call__(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        Q : (..., seq_len_q, d_k)
        K : (..., seq_len_k, d_k)
        V : (..., seq_len_k, d_v)
        mask : (..., seq_len_q, seq_len_k) or None
            Positions with mask == 0 are set to -inf before softmax
            (used for causal / padding masks).

        Returns
        -------
        output : (..., seq_len_q, d_v)
        attn_weights : (..., seq_len_q, seq_len_k)
        """
        d_k = Q.shape[-1]
        scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)

        if mask is not None:
            # Squeeze extra leading dims so mask broadcasts correctly against scores
            m = mask
            while m.ndim > scores.ndim:
                m = m.squeeze(0)
            scores = np.where(m == 0, -1e9, scores)

        attn_weights = _softmax(scores, axis=-1)
        output = attn_weights @ V
        return output, attn_weights


# ============================================================
# Multi-Head Attention
# ============================================================

class MultiHeadAttention:
    """
    Multi-Head Attention with learnable projection matrices.

    Parameters
    ----------
    d_model : int
        Input/output feature dimension.
    n_heads : int
        Number of attention heads.  Must evenly divide d_model.
    random_state : int or None
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        random_state: int | None = None,
    ) -> None:
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        rng   = np.random.default_rng(random_state)
        scale = np.sqrt(2.0 / d_model)
        self.W_q = rng.normal(0, scale, (d_model, d_model))
        self.W_k = rng.normal(0, scale, (d_model, d_model))
        self.W_v = rng.normal(0, scale, (d_model, d_model))
        self.W_o = rng.normal(0, scale, (d_model, d_model))

        self._attn = ScaledDotProductAttention()
        self.last_attn_weights_: np.ndarray | None = None

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, T, d_model) → (B, n_heads, T, d_k)"""
        B, T, _ = x.shape
        x = x.reshape(B, T, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, n_heads, T, d_k) → (B, T, d_model)"""
        B, H, T, d_k = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(B, T, H * d_k)

    def forward(
        self,
        x: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Self-attention (Q=K=V=x).

        Parameters
        ----------
        x : (B, T, d_model)
        mask : (B, 1, T, T) or None

        Returns
        -------
        out : (B, T, d_model)
        """
        Q = self._split_heads(x @ self.W_q)
        K = self._split_heads(x @ self.W_k)
        V = self._split_heads(x @ self.W_v)

        attn_out, attn_weights = self._attn(Q, K, V, mask)
        self.last_attn_weights_ = attn_weights

        combined = self._combine_heads(attn_out)
        return combined @ self.W_o


# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding:
    """
    Sinusoidal positional encoding (no learnable parameters).

    Parameters
    ----------
    d_model : int
    max_len : int
        Maximum supported sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        self.d_model = d_model
        self.max_len = max_len
        self.pe = self._build(d_model, max_len)

    @staticmethod
    def _build(d_model: int, max_len: int) -> np.ndarray:
        position = np.arange(max_len)[:, np.newaxis]                    # (max_len, 1)
        div_term = np.exp(
            np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )                                                                # (d_model/2,)
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to x.

        Parameters
        ----------
        x : (B, T, d_model)  or  (T, d_model)

        Returns
        -------
        same shape as x
        """
        T = x.shape[-2]
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len={self.max_len}.")
        return x + self.pe[:T]


# ============================================================
# LayerNorm
# ============================================================

class LayerNorm:
    """
    Layer Normalisation over the last dimension.

    Parameters
    ----------
    d_model : int
    eps : float
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        self.gamma = np.ones(d_model)
        self.beta  = np.zeros(d_model)
        self.eps   = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x : (..., d_model)"""
        mean = x.mean(axis=-1, keepdims=True)
        var  = x.var(axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


# ============================================================
# Feed-Forward Network
# ============================================================

class FeedForward:
    """
    Position-wise feed-forward network: Linear → ReLU → Linear.

    Parameters
    ----------
    d_model : int
    d_ff : int
        Hidden layer size (typically 4 × d_model).
    random_state : int or None
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = 256,
        random_state: int | None = None,
    ) -> None:
        rng = np.random.default_rng(random_state)
        self.W1 = rng.normal(0, np.sqrt(2.0 / d_model), (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.normal(0, np.sqrt(2.0 / d_ff), (d_ff, d_model))
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x : (..., d_model) → (..., d_model)"""
        h = _relu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


# ============================================================
# Transformer Encoder Layer
# ============================================================

class TransformerEncoderLayer:
    """
    A single Transformer encoder layer:

        x  = LayerNorm(x + MultiHeadAttention(x))
        x  = LayerNorm(x + FeedForward(x))

    Parameters
    ----------
    d_model : int
    n_heads : int
    d_ff : int
    random_state : int or None
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = 256,
        random_state: int | None = None,
    ) -> None:
        self.attn  = MultiHeadAttention(d_model, n_heads, random_state)
        self.ffn   = FeedForward(d_model, d_ff, random_state)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """x : (B, T, d_model) → (B, T, d_model)"""
        attn_out = self.attn.forward(x, mask)
        x = self.norm1.forward(x + attn_out)

        ffn_out = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_out)
        return x


# ============================================================
# Transformer Encoder (stack of layers)
# ============================================================

class TransformerEncoder:
    """
    Stack of TransformerEncoderLayer with input embedding + positional
    encoding.

    Parameters
    ----------
    vocab_size : int
        Size of the input vocabulary (for the embedding lookup).
    d_model : int
    n_heads : int
    n_layers : int
    d_ff : int
    max_len : int
    random_state : int or None
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int = 2,
        d_ff: int = 256,
        max_len: int = 512,
        random_state: int | None = None,
    ) -> None:
        rng = np.random.default_rng(random_state)
        self.d_model    = d_model
        self.embedding  = rng.normal(0, 0.02, (vocab_size, d_model))
        self.pos_enc    = PositionalEncoding(d_model, max_len)

        self.layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff,
                                    (random_state or 0) + i)
            for i in range(n_layers)
        ]

    def forward(
        self,
        token_ids: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        token_ids : (B, T) integer token indices
        mask : (B, 1, T, T) or None

        Returns
        -------
        out : (B, T, d_model)
        """
        x = self.embedding[token_ids]                     # (B, T, d_model)
        x = x * np.sqrt(self.d_model)                      # scale embeddings
        x = self.pos_enc.forward(x)

        for layer in self.layers:
            x = layer.forward(x, mask)

        return x

    @staticmethod
    def causal_mask(seq_len: int) -> np.ndarray:
        """
        Build a causal (look-ahead) mask of shape (1, 1, T, T)
        where position i can attend to positions <= i.
        """
        mask = np.tril(np.ones((seq_len, seq_len)))
        return mask[np.newaxis, np.newaxis, :, :]
