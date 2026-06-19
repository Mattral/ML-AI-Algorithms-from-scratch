"""
Recurrent Neural Networks
==========================
Sequential-data architectures that maintain a hidden state across timesteps.

SimpleRNN
---------
Elman recurrent network:
    h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)
    y_t = W_hy h_t + b_y          (output layer, optional)

LSTMCell / LSTM
---------------
Long Short-Term Memory (Hochreiter & Schmidhuber, 1997).
Four gates operating on the concatenated [x_t; h_{t-1}]:

    i_t = σ(W_i [x_t; h_{t-1}] + b_i)     input gate
    f_t = σ(W_f [x_t; h_{t-1}] + b_f)     forget gate
    g_t = tanh(W_g [x_t; h_{t-1}] + b_g)  cell gate (candidate)
    o_t = σ(W_o [x_t; h_{t-1}] + b_o)     output gate
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    h_t = o_t ⊙ tanh(c_t)

EncoderDecoder
--------------
Sequence-to-sequence architecture with an RNN encoder that compresses
an input sequence to a context vector, and an RNN decoder that
unrolls to produce the output sequence.

References
----------
Elman, J. (1990). Finding structure in time. Cognitive Science, 14(2), 179-211.
Hochreiter & Schmidhuber (1997). Long short-term memory. Neural Computation.
Sutskever et al. (2014). Sequence to sequence learning with neural networks. NeurIPS.

Only numpy is used.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# Helpers
# ============================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ============================================================
# SimpleRNN
# ============================================================

class SimpleRNN:
    """
    Simple Elman RNN.

    Supports sequence classification (uses final hidden state),
    sequence regression, and returning all hidden states.

    Parameters
    ----------
    input_size : int
    hidden_size : int
    output_size : int or None
        If None, the network is a feature extractor (returns hidden states).
    return_sequences : bool
        If True, return hidden state at every timestep.
        If False (default), return only the final hidden state.
    learning_rate : float
    epochs : int
    random_state : int or None
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        return_sequences: bool = False,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        random_state: int | None = None,
    ) -> None:
        self.input_size      = input_size
        self.hidden_size     = hidden_size
        self.output_size     = output_size
        self.return_sequences = return_sequences
        self.learning_rate   = learning_rate
        self.epochs          = epochs
        self._rng            = np.random.default_rng(random_state)

        self._init_params()
        self.losses_: list[float] = []

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_params(self) -> None:
        D, H = self.input_size, self.hidden_size
        s_xh = np.sqrt(2.0 / D)
        s_hh = np.sqrt(2.0 / H)

        self.W_xh = self._rng.normal(0, s_xh, (D, H))
        self.W_hh = self._rng.normal(0, s_hh, (H, H))
        self.b_h  = np.zeros(H)

        if self.output_size is not None:
            self.W_hy = self._rng.normal(0, np.sqrt(2.0 / H), (H, self.output_size))
            self.b_y  = np.zeros(self.output_size)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the RNN.

        Parameters
        ----------
        X : ndarray of shape (seq_len, input_size) or
                             (batch, seq_len, input_size)

        Returns
        -------
        ndarray — hidden states (and optionally output projections)
        """
        batched = X.ndim == 3
        if not batched:
            X = X[np.newaxis, :]          # (1, T, D)

        B, T, D = X.shape
        H = self.hidden_size
        h = np.zeros((B, H))
        hidden_states = []

        for t in range(T):
            h = np.tanh(X[:, t, :] @ self.W_xh + h @ self.W_hh + self.b_h)
            hidden_states.append(h.copy())

        hidden_states = np.stack(hidden_states, axis=1)   # (B, T, H)

        if self.return_sequences:
            out = hidden_states
        else:
            out = hidden_states[:, -1, :]                 # (B, H)

        if self.output_size is not None:
            out = out @ self.W_hy + self.b_y

        return out[0] if not batched else out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleRNN":
        """
        Train the RNN on sequences X with targets y.

        Parameters
        ----------
        X : ndarray (n_samples, seq_len, input_size)
        y : ndarray (n_samples,) or (n_samples, output_size)

        Returns
        -------
        self
        """
        if self.output_size is None:
            raise ValueError("output_size must be set to use fit().")
        n = len(X)
        lr = self.learning_rate
        self.losses_ = []

        for epoch in range(self.epochs):
            idx  = self._rng.permutation(n)
            loss = 0.0

            for i in idx:
                # Single-sample BPTT (simplified; no truncation)
                xi    = X[i]                    # (T, D)
                yi    = y[i:i+1] if y.ndim == 1 else y[i:i+1]
                T_len = xi.shape[0]
                H     = self.hidden_size

                # Forward
                hs = np.zeros((T_len + 1, H))
                for t in range(T_len):
                    hs[t + 1] = np.tanh(
                        xi[t:t+1] @ self.W_xh + hs[t:t+1] @ self.W_hh + self.b_h
                    )

                out   = hs[-1:] @ self.W_hy + self.b_y
                error = out - yi.reshape(1, -1)
                loss += float(np.mean(error ** 2))

                # Backward through output layer
                d_out  = 2.0 * error
                dW_hy  = hs[-1:].T @ d_out
                db_y   = d_out.squeeze()

                # BPTT
                dh_next  = d_out @ self.W_hy.T
                dW_xh    = np.zeros_like(self.W_xh)
                dW_hh    = np.zeros_like(self.W_hh)
                db_h     = np.zeros(H)

                for t in reversed(range(T_len)):
                    dtanh = dh_next * (1.0 - hs[t + 1] ** 2)
                    dW_xh += xi[t:t+1].T @ dtanh
                    dW_hh += hs[t:t+1].T @ dtanh
                    db_h  += dtanh.squeeze()
                    dh_next = dtanh @ self.W_hh.T

                # Clip gradients
                for grad in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
                    np.clip(grad, -5, 5, out=grad)

                self.W_xh  -= lr * dW_xh
                self.W_hh  -= lr * dW_hh
                self.b_h   -= lr * db_h
                self.W_hy  -= lr * dW_hy
                self.b_y   -= lr * db_y

            self.losses_.append(loss / n)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run forward pass on X."""
        return self.forward(X)


# ============================================================
# LSTMCell
# ============================================================

class LSTMCell:
    """
    A single LSTM cell — stateful, processes one timestep at a time.

    Parameters
    ----------
    input_size : int
    hidden_size : int
    random_state : int or None
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        random_state: int | None = None,
    ) -> None:
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self._rng        = np.random.default_rng(random_state)

        H, D = hidden_size, input_size
        scale = np.sqrt(2.0 / (D + H))
        # Single stacked weight matrix for efficiency: [i, f, g, o]
        self.W   = self._rng.normal(0, scale, (4 * H, D + H))
        self.b   = np.zeros(4 * H)

        self.reset_state()

    def reset_state(self) -> None:
        """Reset hidden and cell state to zeros."""
        H        = self.hidden_size
        self.h_t = np.zeros((1, H))
        self.c_t = np.zeros((1, H))

    def forward(self, x_t: np.ndarray) -> np.ndarray:
        """
        Process one timestep.

        Parameters
        ----------
        x_t : ndarray of shape (input_size,) or (1, input_size)

        Returns
        -------
        h_t : ndarray of shape (hidden_size,)
        """
        x_t = np.atleast_2d(x_t)                           # (1, D)
        xh  = np.concatenate([x_t, self.h_t], axis=1)     # (1, D+H)
        gates = xh @ self.W.T + self.b                      # (1, 4H)

        H = self.hidden_size
        i_t = _sigmoid(gates[:, :H])
        f_t = _sigmoid(gates[:, H:2*H])
        g_t = np.tanh(gates[:, 2*H:3*H])
        o_t = _sigmoid(gates[:, 3*H:])

        self.c_t = f_t * self.c_t + i_t * g_t
        self.h_t = o_t * np.tanh(self.c_t)

        return self.h_t.squeeze()


# ============================================================
# LSTM (multi-layer, with optional linear output head)
# ============================================================

class LSTM:
    """
    Multi-layer LSTM for sequence modelling.

    Parameters
    ----------
    input_size : int
    hidden_size : int
    num_layers : int
        Number of stacked LSTM layers.
    output_size : int or None
        If set, a linear projection layer is added on top of the final
        hidden state.
    return_sequences : bool
        Return all hidden states (True) or just the final one (False).
    dropout : float
        Dropout probability applied between LSTM layers (0 = no dropout).
    random_state : int or None
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        output_size: int | None = None,
        return_sequences: bool = False,
        dropout: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        self.input_size      = input_size
        self.hidden_size     = hidden_size
        self.num_layers      = num_layers
        self.output_size     = output_size
        self.return_sequences = return_sequences
        self.dropout         = dropout
        self._rng            = np.random.default_rng(random_state)

        # Build one cell per layer
        layer_input = input_size
        self.cells: list[LSTMCell] = []
        for i in range(num_layers):
            seed = (random_state or 0) + i
            self.cells.append(LSTMCell(layer_input, hidden_size, seed))
            layer_input = hidden_size

        # Optional linear output head
        if output_size is not None:
            scale = np.sqrt(2.0 / hidden_size)
            self.W_out = self._rng.normal(0, scale, (hidden_size, output_size))
            self.b_out = np.zeros(output_size)
        else:
            self.W_out = None
            self.b_out = None

    def reset_states(self) -> None:
        """Reset all cell hidden and cell states."""
        for cell in self.cells:
            cell.reset_state()

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass through the stacked LSTM.

        Parameters
        ----------
        X : ndarray of shape (seq_len, input_size) or
                             (batch, seq_len, input_size)
        training : bool
            If True and dropout > 0, apply dropout between layers.

        Returns
        -------
        ndarray — shape depends on return_sequences and output_size
        """
        batched = X.ndim == 3
        if batched:
            # Process each sequence in batch independently
            results = [self._forward_single(X[b], training) for b in range(X.shape[0])]
            return np.stack(results)
        return self._forward_single(X, training)

    def _forward_single(self, X: np.ndarray, training: bool) -> np.ndarray:
        """Forward pass for a single (unbatched) sequence (T, D)."""
        T = len(X)
        H = self.hidden_size

        # Reset states for fresh inference
        self.reset_states()

        all_outputs = []
        current_input = X   # (T, D)

        for layer_idx, cell in enumerate(self.cells):
            layer_outputs = []
            for t in range(T):
                h_t = cell.forward(current_input[t])
                layer_outputs.append(h_t.copy())
            layer_outputs = np.stack(layer_outputs)    # (T, H)

            # Dropout between layers (not on last layer)
            if (training and self.dropout > 0
                    and layer_idx < self.num_layers - 1):
                mask = (self._rng.random(layer_outputs.shape) > self.dropout).astype(float)
                layer_outputs = layer_outputs * mask / (1.0 - self.dropout + 1e-8)

            current_input = layer_outputs
            all_outputs.append(layer_outputs)

        final_hidden = all_outputs[-1]   # (T, H) from last layer

        if self.return_sequences:
            out = final_hidden
        else:
            out = final_hidden[-1]       # (H,)

        if self.W_out is not None:
            out = out @ self.W_out + self.b_out

        return out


# ============================================================
# Encoder-Decoder (Seq2Seq)
# ============================================================

class EncoderDecoder:
    """
    Sequence-to-sequence Encoder-Decoder with RNN encoder and decoder.

    The encoder reads the input sequence and produces a context vector
    (final hidden state).  The decoder is initialised with this context
    and unrolls to generate the output sequence.

    Parameters
    ----------
    input_vocab_size : int
        Vocabulary size of the input sequence (one-hot encoded).
    output_vocab_size : int
        Vocabulary size of the output sequence.
    hidden_size : int
        Hidden state size for both encoder and decoder.
    random_state : int or None
    """

    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        hidden_size: int,
        random_state: int | None = None,
    ) -> None:
        self.input_vocab_size  = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.hidden_size       = hidden_size
        self._rng              = np.random.default_rng(random_state)

        self._init_params()

    def _init_params(self) -> None:
        D_in  = self.input_vocab_size
        D_out = self.output_vocab_size
        H     = self.hidden_size
        s = lambda fi: np.sqrt(2.0 / fi)

        # Encoder
        self.We_xh = self._rng.normal(0, s(D_in), (D_in, H))
        self.We_hh = self._rng.normal(0, s(H),    (H, H))
        self.be_h  = np.zeros(H)

        # Decoder
        self.Wd_xh = self._rng.normal(0, s(H),    (H, H))
        self.Wd_hh = self._rng.normal(0, s(H),    (H, H))
        self.bd_h  = np.zeros(H)

        # Decoder output projection
        self.Wd_hy = self._rng.normal(0, s(H),    (H, D_out))
        self.bd_y  = np.zeros(D_out)

    def _encode(self, X_one_hot: np.ndarray) -> np.ndarray:
        """
        Encode input sequence into a context vector.

        Parameters
        ----------
        X_one_hot : ndarray (seq_len, input_vocab_size)

        Returns
        -------
        context : ndarray (hidden_size,) — final hidden state
        """
        H = self.hidden_size
        h = np.zeros(H)
        for t in range(len(X_one_hot)):
            h = np.tanh(
                X_one_hot[t] @ self.We_xh + h @ self.We_hh + self.be_h
            )
        return h

    def _decode(self, context: np.ndarray, output_len: int) -> np.ndarray:
        """
        Decode context vector into an output sequence.

        Parameters
        ----------
        context : ndarray (hidden_size,)
        output_len : int

        Returns
        -------
        outputs : ndarray (output_len, output_vocab_size)
        """
        H = self.hidden_size
        h = np.zeros(H)
        outputs = []

        for t in range(output_len):
            h = np.tanh(
                context @ self.Wd_xh + h @ self.Wd_hh + self.bd_h
            )
            y_t = _softmax((h @ self.Wd_hy + self.bd_y).reshape(1, -1)).squeeze()
            outputs.append(y_t)

        return np.stack(outputs)

    def forward(
        self,
        input_sequence: np.ndarray,
        output_len: int | None = None,
    ) -> np.ndarray:
        """
        Encode input_sequence and decode to output_len tokens.

        Parameters
        ----------
        input_sequence : ndarray (seq_len, input_vocab_size)
            One-hot encoded input.
        output_len : int or None
            Target sequence length.  Defaults to len(input_sequence).

        Returns
        -------
        outputs : ndarray (output_len, output_vocab_size)
        """
        if output_len is None:
            output_len = len(input_sequence)
        context = self._encode(input_sequence)
        return self._decode(context, output_len)

    def predict_sequence(self, input_sequence: np.ndarray) -> np.ndarray:
        """
        Predict token indices for an input sequence.

        Returns
        -------
        ndarray of shape (output_len,) — integer token indices
        """
        outputs = self.forward(input_sequence)
        return np.argmax(outputs, axis=1)
