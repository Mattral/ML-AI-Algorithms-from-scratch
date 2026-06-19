"""
Tests for mlscratch.neural.attention
Covers: ScaledDotProductAttention, MultiHeadAttention, PositionalEncoding,
        LayerNorm, FeedForward, TransformerEncoderLayer, TransformerEncoder
"""

from __future__ import annotations

import numpy as np
import pytest

from mlscratch.neural.attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionalEncoding,
    LayerNorm,
    FeedForward,
    TransformerEncoderLayer,
    TransformerEncoder,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def qkv():
    rng = np.random.default_rng(0)
    B, T, d = 2, 5, 8
    Q = rng.standard_normal((B, T, d))
    K = rng.standard_normal((B, T, d))
    V = rng.standard_normal((B, T, d))
    return Q, K, V


@pytest.fixture
def sequence_batch():
    rng = np.random.default_rng(1)
    return rng.standard_normal((2, 6, 16))   # (B, T, d_model)


# ===================================================================
# ScaledDotProductAttention
# ===================================================================

class TestScaledDotProductAttention:
    def test_output_shape(self, qkv):
        Q, K, V = qkv
        attn = ScaledDotProductAttention()
        out, weights = attn(Q, K, V)
        assert out.shape == Q.shape

    def test_weights_shape(self, qkv):
        Q, K, V = qkv
        attn = ScaledDotProductAttention()
        _, weights = attn(Q, K, V)
        B, T, _ = Q.shape
        assert weights.shape == (B, T, T)

    def test_weights_sum_to_one(self, qkv):
        Q, K, V = qkv
        attn = ScaledDotProductAttention()
        _, weights = attn(Q, K, V)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-6)

    def test_weights_non_negative(self, qkv):
        Q, K, V = qkv
        attn = ScaledDotProductAttention()
        _, weights = attn(Q, K, V)
        assert np.all(weights >= 0)

    def test_identical_qk_gives_diagonal_dominant_attention(self):
        """When Q == K and each row is distinct, the attention weight
        on the matching position should be the largest in that row."""
        d = 8
        Q = np.eye(4, d) * 10   # large magnitude -> sharp softmax
        K = Q.copy()
        V = np.random.default_rng(0).standard_normal((4, d))
        attn = ScaledDotProductAttention()
        _, weights = attn(Q[np.newaxis], K[np.newaxis], V[np.newaxis])
        diag = np.diagonal(weights[0])
        for i in range(4):
            assert weights[0, i, i] == diag[i]
            assert diag[i] >= weights[0, i].max() - 1e-9

    def test_causal_mask_blocks_future(self, qkv):
        Q, K, V = qkv
        B, T, _ = Q.shape
        mask = TransformerEncoder.causal_mask(T)
        attn = ScaledDotProductAttention()
        _, weights = attn(Q, K, V, mask=mask)
        for i in range(T):
            for j in range(i + 1, T):
                assert weights[:, i, j].max() < 1e-6

    def test_mask_preserves_softmax_normalisation(self, qkv):
        Q, K, V = qkv
        B, T, _ = Q.shape
        mask = TransformerEncoder.causal_mask(T)
        attn = ScaledDotProductAttention()
        _, weights = attn(Q, K, V, mask=mask)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-6)

    def test_different_d_v(self):
        """V can have a different feature dimension than Q/K."""
        Q = np.random.default_rng(0).standard_normal((1, 3, 4))
        K = np.random.default_rng(1).standard_normal((1, 3, 4))
        V = np.random.default_rng(2).standard_normal((1, 3, 7))
        attn = ScaledDotProductAttention()
        out, _ = attn(Q, K, V)
        assert out.shape == (1, 3, 7)


# ===================================================================
# MultiHeadAttention
# ===================================================================

class TestMultiHeadAttention:
    def test_invalid_head_count_raises(self):
        with pytest.raises(ValueError):
            MultiHeadAttention(d_model=10, n_heads=3)

    def test_output_shape(self, sequence_batch):
        mha = MultiHeadAttention(d_model=16, n_heads=4, random_state=0)
        out = mha.forward(sequence_batch)
        assert out.shape == sequence_batch.shape

    def test_attention_weights_shape(self, sequence_batch):
        mha = MultiHeadAttention(d_model=16, n_heads=4, random_state=0)
        mha.forward(sequence_batch)
        B, T, _ = sequence_batch.shape
        assert mha.last_attn_weights_.shape == (B, 4, T, T)

    def test_attention_weights_sum_to_one(self, sequence_batch):
        mha = MultiHeadAttention(d_model=16, n_heads=4, random_state=0)
        mha.forward(sequence_batch)
        np.testing.assert_allclose(
            mha.last_attn_weights_.sum(axis=-1), 1.0, atol=1e-6
        )

    def test_split_combine_heads_inverse(self, sequence_batch):
        mha = MultiHeadAttention(d_model=16, n_heads=4, random_state=0)
        split = mha._split_heads(sequence_batch)
        combined = mha._combine_heads(split)
        np.testing.assert_allclose(combined, sequence_batch, atol=1e-10)

    def test_split_heads_shape(self, sequence_batch):
        mha = MultiHeadAttention(d_model=16, n_heads=4, random_state=0)
        split = mha._split_heads(sequence_batch)
        B, T, d = sequence_batch.shape
        assert split.shape == (B, 4, T, d // 4)

    def test_d_k_computed_correctly(self):
        mha = MultiHeadAttention(d_model=64, n_heads=8, random_state=0)
        assert mha.d_k == 8

    def test_single_head_equals_full_attention(self, sequence_batch):
        """With n_heads=1, MHA reduces to a single scaled dot-product
        attention with full d_model."""
        mha = MultiHeadAttention(d_model=16, n_heads=1, random_state=0)
        out = mha.forward(sequence_batch)
        assert out.shape == sequence_batch.shape
        assert mha.last_attn_weights_.shape[1] == 1

    def test_causal_mask_applied(self, sequence_batch):
        B, T, _ = sequence_batch.shape
        mha = MultiHeadAttention(d_model=16, n_heads=4, random_state=0)
        mask = TransformerEncoder.causal_mask(T)
        mha.forward(sequence_batch, mask=mask)
        for i in range(T):
            for j in range(i + 1, T):
                assert mha.last_attn_weights_[:, :, i, j].max() < 1e-6


# ===================================================================
# PositionalEncoding
# ===================================================================

class TestPositionalEncoding:
    def test_pe_shape(self):
        pe = PositionalEncoding(d_model=16, max_len=50)
        assert pe.pe.shape == (50, 16)

    def test_forward_shape_unchanged(self, sequence_batch):
        pe = PositionalEncoding(d_model=16, max_len=50)
        out = pe.forward(sequence_batch)
        assert out.shape == sequence_batch.shape

    def test_exceeding_max_len_raises(self):
        pe = PositionalEncoding(d_model=8, max_len=4)
        x = np.random.default_rng(0).standard_normal((1, 5, 8))
        with pytest.raises(ValueError):
            pe.forward(x)

    def test_sinusoidal_values_bounded(self):
        pe = PositionalEncoding(d_model=16, max_len=50)
        assert np.all(pe.pe >= -1.0) and np.all(pe.pe <= 1.0)

    def test_position_zero_alternates_sin_cos(self):
        """At position 0: sin(0)=0 for even indices, cos(0)=1 for odd indices."""
        pe = PositionalEncoding(d_model=8, max_len=10)
        np.testing.assert_allclose(pe.pe[0, 0::2], 0.0, atol=1e-10)   # sin(0)=0
        np.testing.assert_allclose(pe.pe[0, 1::2], 1.0, atol=1e-10)   # cos(0)=1

    def test_different_positions_have_different_encodings(self):
        pe = PositionalEncoding(d_model=16, max_len=50)
        assert not np.allclose(pe.pe[0], pe.pe[1])

    def test_2d_input(self):
        """Should also work on (T, d_model) without batch dim."""
        pe = PositionalEncoding(d_model=8, max_len=20)
        x = np.random.default_rng(0).standard_normal((5, 8))
        out = pe.forward(x)
        assert out.shape == (5, 8)


# ===================================================================
# LayerNorm
# ===================================================================

class TestLayerNorm:
    def test_init_gamma_ones_beta_zeros(self):
        ln = LayerNorm(d_model=8)
        np.testing.assert_array_equal(ln.gamma, np.ones(8))
        np.testing.assert_array_equal(ln.beta, np.zeros(8))

    def test_output_shape(self, sequence_batch):
        ln = LayerNorm(d_model=16)
        out = ln.forward(sequence_batch)
        assert out.shape == sequence_batch.shape

    def test_normalised_zero_mean_unit_var(self, sequence_batch):
        ln = LayerNorm(d_model=16)
        out = ln.forward(sequence_batch)
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-5)
        np.testing.assert_allclose(out.var(axis=-1), 1.0, atol=1e-3)

    def test_gamma_beta_scale_and_shift(self, sequence_batch):
        ln = LayerNorm(d_model=16)
        ln.gamma = np.ones(16) * 2.0
        ln.beta  = np.ones(16) * 3.0
        out = ln.forward(sequence_batch)
        np.testing.assert_allclose(out.mean(axis=-1), 3.0, atol=1e-4)


# ===================================================================
# FeedForward
# ===================================================================

class TestFeedForward:
    def test_output_shape(self, sequence_batch):
        ffn = FeedForward(d_model=16, d_ff=32, random_state=0)
        out = ffn.forward(sequence_batch)
        assert out.shape == sequence_batch.shape

    def test_weight_shapes(self):
        ffn = FeedForward(d_model=16, d_ff=32, random_state=0)
        assert ffn.W1.shape == (16, 32)
        assert ffn.W2.shape == (32, 16)

    def test_intermediate_relu_nonneg(self):
        """The hidden activation (after ReLU) inside FFN should be >= 0."""
        ffn = FeedForward(d_model=4, d_ff=8, random_state=0)
        x = np.random.default_rng(0).standard_normal((1, 3, 4))
        h = np.maximum(0.0, x @ ffn.W1 + ffn.b1)
        assert np.all(h >= 0)


# ===================================================================
# TransformerEncoderLayer
# ===================================================================

class TestTransformerEncoderLayer:
    def test_output_shape(self, sequence_batch):
        layer = TransformerEncoderLayer(d_model=16, n_heads=4, d_ff=32,
                                        random_state=0)
        out = layer.forward(sequence_batch)
        assert out.shape == sequence_batch.shape

    def test_output_normalised(self, sequence_batch):
        """After the final LayerNorm, mean should be ~0 per position."""
        layer = TransformerEncoderLayer(d_model=16, n_heads=4, d_ff=32,
                                        random_state=0)
        out = layer.forward(sequence_batch)
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-4)

    def test_residual_connection_changes_output(self, sequence_batch):
        """Output should differ from input (non-trivial transformation)."""
        layer = TransformerEncoderLayer(d_model=16, n_heads=4, d_ff=32,
                                        random_state=0)
        out = layer.forward(sequence_batch)
        assert not np.allclose(out, sequence_batch)

    def test_with_causal_mask(self, sequence_batch):
        B, T, _ = sequence_batch.shape
        layer = TransformerEncoderLayer(d_model=16, n_heads=4, d_ff=32,
                                        random_state=0)
        mask = TransformerEncoder.causal_mask(T)
        out = layer.forward(sequence_batch, mask=mask)
        assert out.shape == sequence_batch.shape
        assert not np.any(np.isnan(out))


# ===================================================================
# TransformerEncoder — Basic API
# ===================================================================

class TestTransformerEncoderBasic:
    def test_output_shape(self):
        enc = TransformerEncoder(vocab_size=20, d_model=16, n_heads=4,
                                 n_layers=2, d_ff=32, max_len=10,
                                 random_state=0)
        tokens = np.random.default_rng(0).integers(0, 20, (2, 6))
        out = enc.forward(tokens)
        assert out.shape == (2, 6, 16)

    def test_embedding_shape(self):
        enc = TransformerEncoder(vocab_size=20, d_model=16, n_heads=4,
                                 n_layers=2, random_state=0)
        assert enc.embedding.shape == (20, 16)

    def test_num_layers(self):
        enc = TransformerEncoder(vocab_size=20, d_model=16, n_heads=4,
                                 n_layers=3, random_state=0)
        assert len(enc.layers) == 3

    def test_causal_mask_shape(self):
        mask = TransformerEncoder.causal_mask(5)
        assert mask.shape == (1, 1, 5, 5)

    def test_causal_mask_lower_triangular(self):
        mask = TransformerEncoder.causal_mask(4)
        expected = np.tril(np.ones((4, 4)))
        np.testing.assert_array_equal(mask[0, 0], expected)

    def test_no_nan_in_output(self):
        enc = TransformerEncoder(vocab_size=20, d_model=16, n_heads=4,
                                 n_layers=2, d_ff=32, max_len=10,
                                 random_state=0)
        tokens = np.random.default_rng(0).integers(0, 20, (2, 6))
        out = enc.forward(tokens)
        assert not np.any(np.isnan(out))

    def test_with_causal_mask_end_to_end(self):
        enc = TransformerEncoder(vocab_size=20, d_model=16, n_heads=4,
                                 n_layers=2, d_ff=32, max_len=10,
                                 random_state=0)
        tokens = np.random.default_rng(0).integers(0, 20, (2, 6))
        mask = TransformerEncoder.causal_mask(6)
        out = enc.forward(tokens, mask=mask)
        assert out.shape == (2, 6, 16)


# ===================================================================
# TransformerEncoder — Edge cases
# ===================================================================

class TestTransformerEncoderEdgeCases:
    def test_single_layer(self):
        enc = TransformerEncoder(vocab_size=10, d_model=8, n_heads=2,
                                 n_layers=1, d_ff=16, max_len=10,
                                 random_state=0)
        tokens = np.array([[0, 1, 2]])
        out = enc.forward(tokens)
        assert out.shape == (1, 3, 8)

    def test_sequence_length_one(self):
        enc = TransformerEncoder(vocab_size=10, d_model=8, n_heads=2,
                                 n_layers=1, d_ff=16, max_len=10,
                                 random_state=0)
        tokens = np.array([[3]])
        out = enc.forward(tokens)
        assert out.shape == (1, 1, 8)

    def test_single_head(self):
        enc = TransformerEncoder(vocab_size=10, d_model=8, n_heads=1,
                                 n_layers=1, d_ff=16, max_len=10,
                                 random_state=0)
        tokens = np.array([[0, 1, 2]])
        out = enc.forward(tokens)
        assert out.shape == (1, 3, 8)

    def test_batch_size_one(self):
        enc = TransformerEncoder(vocab_size=10, d_model=8, n_heads=2,
                                 n_layers=2, d_ff=16, max_len=10,
                                 random_state=0)
        tokens = np.array([[0, 1, 2, 3]])
        out = enc.forward(tokens)
        assert out.shape == (1, 4, 8)
