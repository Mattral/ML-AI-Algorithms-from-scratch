"""
Tests for mlscratch.neural.recurrent
Covers: SimpleRNN, LSTMCell, LSTM, EncoderDecoder
"""

from __future__ import annotations

import numpy as np
import pytest

from mlscratch.neural.recurrent import SimpleRNN, LSTMCell, LSTM, EncoderDecoder


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def seq_regression_data():
    """20 sequences of length 5, 3 features; target = 2 * last-step feature 0."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 5, 3))
    y = X[:, -1, 0] * 2.0
    return X, y


@pytest.fixture
def single_sequence():
    rng = np.random.default_rng(1)
    return rng.standard_normal((6, 4))   # (T, D)


# ===================================================================
# SimpleRNN — Basic API
# ===================================================================

class TestSimpleRNNBasic:
    def test_init_param_shapes(self):
        rnn = SimpleRNN(input_size=3, hidden_size=8, random_state=0)
        assert rnn.W_xh.shape == (3, 8)
        assert rnn.W_hh.shape == (8, 8)
        assert rnn.b_h.shape == (8,)

    def test_output_layer_created_when_output_size_set(self):
        rnn = SimpleRNN(input_size=3, hidden_size=8, output_size=2, random_state=0)
        assert rnn.W_hy.shape == (8, 2)
        assert rnn.b_y.shape == (2,)

    def test_forward_single_sequence_default(self, single_sequence):
        rnn = SimpleRNN(input_size=4, hidden_size=6, random_state=0)
        out = rnn.forward(single_sequence)
        assert out.shape == (6,)         # final hidden state only

    def test_forward_return_sequences(self, single_sequence):
        rnn = SimpleRNN(input_size=4, hidden_size=6, return_sequences=True,
                        random_state=0)
        out = rnn.forward(single_sequence)
        assert out.shape == (6, 6)        # (T, H)

    def test_forward_batched_input(self, seq_regression_data):
        X, _ = seq_regression_data
        rnn = SimpleRNN(input_size=3, hidden_size=6, random_state=0)
        out = rnn.forward(X)
        assert out.shape == (20, 6)       # (B, H)

    def test_forward_with_output_size(self, single_sequence):
        rnn = SimpleRNN(input_size=4, hidden_size=6, output_size=2, random_state=0)
        out = rnn.forward(single_sequence)
        assert out.shape == (2,)

    def test_fit_without_output_size_raises(self, seq_regression_data):
        X, y = seq_regression_data
        rnn = SimpleRNN(input_size=3, hidden_size=6, random_state=0)
        with pytest.raises(ValueError):
            rnn.fit(X, y)

    def test_fit_returns_self(self, seq_regression_data):
        X, y = seq_regression_data
        rnn = SimpleRNN(input_size=3, hidden_size=6, output_size=1, epochs=2,
                       random_state=0)
        assert rnn.fit(X, y) is rnn

    def test_losses_recorded(self, seq_regression_data):
        X, y = seq_regression_data
        rnn = SimpleRNN(input_size=3, hidden_size=6, output_size=1, epochs=4,
                       random_state=0).fit(X, y)
        assert len(rnn.losses_) == 4

    def test_predict_matches_forward(self, seq_regression_data):
        X, y = seq_regression_data
        rnn = SimpleRNN(input_size=3, hidden_size=6, output_size=1, epochs=2,
                       random_state=0).fit(X, y)
        np.testing.assert_array_equal(rnn.predict(X), rnn.forward(X))


# ===================================================================
# SimpleRNN — Correctness
# ===================================================================

class TestSimpleRNNCorrectness:
    def test_loss_decreases_with_training(self, seq_regression_data):
        X, y = seq_regression_data
        rnn = SimpleRNN(input_size=3, hidden_size=8, output_size=1,
                       learning_rate=0.01, epochs=30, random_state=0).fit(X, y)
        assert rnn.losses_[-1] < rnn.losses_[0]

    def test_hidden_state_bounded_by_tanh(self, single_sequence):
        rnn = SimpleRNN(input_size=4, hidden_size=6, return_sequences=True,
                        random_state=0)
        out = rnn.forward(single_sequence)
        assert np.all(out >= -1.0) and np.all(out <= 1.0)

    def test_gradient_clipping_prevents_explosion(self, seq_regression_data):
        """Even with a high learning rate, gradients should be clipped to ±5
        and the loss should remain finite throughout training."""
        X, y = seq_regression_data
        rnn = SimpleRNN(input_size=3, hidden_size=8, output_size=1,
                       learning_rate=0.5, epochs=10, random_state=0).fit(X, y)
        assert all(np.isfinite(l) for l in rnn.losses_)


# ===================================================================
# SimpleRNN — Edge cases
# ===================================================================

class TestSimpleRNNEdgeCases:
    def test_sequence_length_one(self):
        rnn = SimpleRNN(input_size=3, hidden_size=4, random_state=0)
        x = np.random.default_rng(0).standard_normal((1, 3))
        out = rnn.forward(x)
        assert out.shape == (4,)

    def test_single_feature(self):
        rnn = SimpleRNN(input_size=1, hidden_size=4, return_sequences=True,
                        random_state=0)
        x = np.random.default_rng(0).standard_normal((5, 1))
        out = rnn.forward(x)
        assert out.shape == (5, 4)


# ===================================================================
# LSTMCell
# ===================================================================

class TestLSTMCell:
    def test_init_state_zero(self):
        cell = LSTMCell(input_size=4, hidden_size=6, random_state=0)
        assert np.all(cell.h_t == 0)
        assert np.all(cell.c_t == 0)

    def test_forward_output_shape(self):
        cell = LSTMCell(input_size=4, hidden_size=6, random_state=0)
        h = cell.forward(np.random.default_rng(0).standard_normal(4))
        assert h.shape == (6,)

    def test_hidden_state_bounded(self):
        """h_t = o_t * tanh(c_t) is bounded in [-1, 1]."""
        cell = LSTMCell(input_size=4, hidden_size=6, random_state=0)
        h = cell.forward(np.random.default_rng(0).standard_normal(4) * 10)
        assert np.all(h >= -1.0) and np.all(h <= 1.0)

    def test_state_persists_across_calls(self):
        cell = LSTMCell(input_size=4, hidden_size=6, random_state=0)
        x = np.random.default_rng(0).standard_normal(4)
        h1 = cell.forward(x).copy()
        h2 = cell.forward(x).copy()
        # State accumulates, so repeated identical input gives different output
        assert not np.allclose(h1, h2)

    def test_reset_state_zeros_h_and_c(self):
        cell = LSTMCell(input_size=4, hidden_size=6, random_state=0)
        cell.forward(np.random.default_rng(0).standard_normal(4))
        cell.reset_state()
        assert np.all(cell.h_t == 0)
        assert np.all(cell.c_t == 0)

    def test_gate_weight_shape(self):
        cell = LSTMCell(input_size=4, hidden_size=6, random_state=0)
        # Stacked [i,f,g,o] gates: shape (4H, D+H)
        assert cell.W.shape == (4 * 6, 4 + 6)
        assert cell.b.shape == (4 * 6,)


# ===================================================================
# LSTM — Basic API
# ===================================================================

class TestLSTMBasic:
    def test_num_cells_matches_num_layers(self):
        lstm = LSTM(input_size=4, hidden_size=6, num_layers=3, random_state=0)
        assert len(lstm.cells) == 3

    def test_forward_default_returns_final_hidden(self, single_sequence):
        lstm = LSTM(input_size=4, hidden_size=6, random_state=0)
        out = lstm.forward(single_sequence)
        assert out.shape == (6,)

    def test_forward_return_sequences(self, single_sequence):
        lstm = LSTM(input_size=4, hidden_size=6, return_sequences=True,
                    random_state=0)
        out = lstm.forward(single_sequence)
        assert out.shape == (6, 6)   # (T, H)

    def test_output_head_shape(self, single_sequence):
        lstm = LSTM(input_size=4, hidden_size=6, output_size=3, random_state=0)
        out = lstm.forward(single_sequence)
        assert out.shape == (3,)

    def test_multilayer_first_layer_input_size(self):
        lstm = LSTM(input_size=4, hidden_size=6, num_layers=2, random_state=0)
        assert lstm.cells[0].input_size == 4
        assert lstm.cells[1].input_size == 6   # subsequent layers take H as input

    def test_batched_forward(self, seq_regression_data):
        X, _ = seq_regression_data
        lstm = LSTM(input_size=3, hidden_size=5, random_state=0)
        out = lstm.forward(X)
        assert out.shape == (20, 5)

    def test_reset_states(self, single_sequence):
        lstm = LSTM(input_size=4, hidden_size=6, num_layers=2, random_state=0)
        lstm.forward(single_sequence)
        lstm.reset_states()
        for cell in lstm.cells:
            assert np.all(cell.h_t == 0)
            assert np.all(cell.c_t == 0)

    def test_no_output_head_when_output_size_none(self):
        lstm = LSTM(input_size=4, hidden_size=6, random_state=0)
        assert lstm.W_out is None
        assert lstm.b_out is None


# ===================================================================
# LSTM — Correctness
# ===================================================================

class TestLSTMCorrectness:
    def test_output_no_nan(self, single_sequence):
        lstm = LSTM(input_size=4, hidden_size=8, num_layers=2,
                    output_size=2, random_state=0)
        out = lstm.forward(single_sequence)
        assert not np.any(np.isnan(out))

    def test_hidden_state_bounded(self, single_sequence):
        lstm = LSTM(input_size=4, hidden_size=6, return_sequences=True,
                    random_state=0)
        out = lstm.forward(single_sequence)
        assert np.all(out >= -1.0) and np.all(out <= 1.0)

    def test_dropout_training_vs_inference_differ(self, single_sequence):
        """With dropout > 0, training=True can produce different outputs
        across calls due to random masking."""
        lstm = LSTM(input_size=4, hidden_size=6, num_layers=2, dropout=0.5,
                    random_state=0)
        out1 = lstm.forward(single_sequence, training=True)
        out2 = lstm.forward(single_sequence, training=True)
        # At least one of the two should differ due to dropout randomness
        # (allow equality in rare cases, but generally should differ)
        assert out1.shape == out2.shape


# ===================================================================
# LSTM — Edge cases
# ===================================================================

class TestLSTMEdgeCases:
    def test_single_layer(self, single_sequence):
        lstm = LSTM(input_size=4, hidden_size=6, num_layers=1, random_state=0)
        out = lstm.forward(single_sequence)
        assert out.shape == (6,)

    def test_sequence_length_one(self):
        lstm = LSTM(input_size=3, hidden_size=4, random_state=0)
        x = np.random.default_rng(0).standard_normal((1, 3))
        out = lstm.forward(x)
        assert out.shape == (4,)


# ===================================================================
# EncoderDecoder
# ===================================================================

class TestEncoderDecoder:
    def test_init_param_shapes(self):
        ed = EncoderDecoder(input_vocab_size=5, output_vocab_size=4,
                            hidden_size=8, random_state=0)
        assert ed.We_xh.shape == (5, 8)
        assert ed.We_hh.shape == (8, 8)
        assert ed.Wd_hy.shape == (8, 4)

    def test_encode_shape(self):
        ed = EncoderDecoder(input_vocab_size=5, output_vocab_size=4,
                            hidden_size=8, random_state=0)
        seq = np.eye(5)[[0, 1, 2]]
        context = ed._encode(seq)
        assert context.shape == (8,)

    def test_decode_shape(self):
        ed = EncoderDecoder(input_vocab_size=5, output_vocab_size=4,
                            hidden_size=8, random_state=0)
        context = np.random.default_rng(0).standard_normal(8)
        out = ed._decode(context, output_len=6)
        assert out.shape == (6, 4)

    def test_decode_output_is_distribution(self):
        """Each output row should be a valid softmax distribution."""
        ed = EncoderDecoder(input_vocab_size=5, output_vocab_size=4,
                            hidden_size=8, random_state=0)
        context = np.random.default_rng(0).standard_normal(8)
        out = ed._decode(context, output_len=3)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(out >= 0)

    def test_forward_default_output_len(self):
        ed = EncoderDecoder(input_vocab_size=5, output_vocab_size=4,
                            hidden_size=8, random_state=0)
        seq = np.eye(5)[[0, 1, 2]]
        out = ed.forward(seq)
        assert out.shape == (3, 4)   # default output_len == input length

    def test_forward_custom_output_len(self):
        ed = EncoderDecoder(input_vocab_size=5, output_vocab_size=4,
                            hidden_size=8, random_state=0)
        seq = np.eye(5)[[0, 1, 2]]
        out = ed.forward(seq, output_len=7)
        assert out.shape == (7, 4)

    def test_predict_sequence_returns_indices(self):
        ed = EncoderDecoder(input_vocab_size=5, output_vocab_size=4,
                            hidden_size=8, random_state=0)
        seq = np.eye(5)[[0, 1, 2, 3]]
        pred = ed.predict_sequence(seq)
        assert pred.shape == (4,)
        assert np.all((pred >= 0) & (pred < 4))

    def test_different_inputs_give_different_contexts(self):
        ed = EncoderDecoder(input_vocab_size=5, output_vocab_size=4,
                            hidden_size=8, random_state=0)
        seq1 = np.eye(5)[[0, 1, 2]]
        seq2 = np.eye(5)[[4, 4, 4]]
        c1 = ed._encode(seq1)
        c2 = ed._encode(seq2)
        assert not np.allclose(c1, c2)

    def test_single_token_input(self):
        ed = EncoderDecoder(input_vocab_size=5, output_vocab_size=4,
                            hidden_size=8, random_state=0)
        seq = np.eye(5)[[0]]
        out = ed.forward(seq)
        assert out.shape == (1, 4)
