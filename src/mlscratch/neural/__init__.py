"""
mlscratch.neural
=================
From-scratch implementations of neural network architectures.
Pure NumPy — no PyTorch, no TensorFlow.

Perceptrons
-----------
SingleLayerPerceptron   – binary classification or regression
MultiLayerPerceptron    – feedforward network, classification or regression

Autoencoders
------------
Autoencoder              – tied-weight vanilla autoencoder
DenoisingAutoencoder     – trained on corrupted inputs (Gaussian / dropout noise)
VariationalAutoencoder   – Gaussian latent space, reparameterisation trick

Recurrent Networks
-------------------
SimpleRNN        – Elman RNN, classification/regression/feature-extractor
LSTMCell         – single-timestep LSTM cell
LSTM             – multi-layer LSTM, optional linear output head
EncoderDecoder   – seq2seq RNN encoder-decoder

Convolutional Networks
------------------------
Conv2D, MaxPool2D, AvgPool2D, BatchNorm2D, Flatten, Dense  – CNN building blocks
SimpleCNN        – pre-wired conv → pool → conv → pool → dense → softmax

Attention / Transformer
--------------------------
ScaledDotProductAttention
MultiHeadAttention
PositionalEncoding
LayerNorm
FeedForward
TransformerEncoderLayer
TransformerEncoder

Generative Models
-------------------
Generator, Discriminator, GAN  – adversarial generative network

Associative Memory
--------------------
HopfieldNetwork   – discrete bipolar associative memory

Energy-Based Models
----------------------
RestrictedBoltzmannMachine  – RBM trained with Contrastive Divergence

Radial Basis Function Networks
---------------------------------
RBFNetwork   – Gaussian RBF hidden layer + closed-form linear output

Complex-Valued Networks
---------------------------
ComplexDense       – complex-valued fully-connected layer
ComplexValuedNN    – multi-layer complex-valued feedforward network

Note
----
Bayesian Neural Networks live in ``mlscratch.bayesian.bayesian_nn``
(``BayesianNeuralNetwork``) since they are fundamentally a Bayesian
inference method applied to a network architecture.
"""

from .perceptron import SingleLayerPerceptron, MultiLayerPerceptron      # noqa: F401
from .autoencoder import (                                                # noqa: F401
    Autoencoder,
    DenoisingAutoencoder,
    VariationalAutoencoder,
)
from .recurrent import SimpleRNN, LSTMCell, LSTM, EncoderDecoder          # noqa: F401
from .cnn import (                                                         # noqa: F401
    Conv2D,
    MaxPool2D,
    AvgPool2D,
    BatchNorm2D,
    Flatten,
    Dense,
    SimpleCNN,
)
from .attention import (                                                   # noqa: F401
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionalEncoding,
    LayerNorm,
    FeedForward,
    TransformerEncoderLayer,
    TransformerEncoder,
)
from .gan import Generator, Discriminator, GAN                            # noqa: F401
from .hopfield import HopfieldNetwork                                     # noqa: F401
from .boltzmann import RestrictedBoltzmannMachine                         # noqa: F401
from .rbf_network import RBFNetwork                                       # noqa: F401
from .cvnn import ComplexDense, ComplexValuedNN                           # noqa: F401

__all__ = [
    # Perceptrons
    "SingleLayerPerceptron", "MultiLayerPerceptron",
    # Autoencoders
    "Autoencoder", "DenoisingAutoencoder", "VariationalAutoencoder",
    # Recurrent
    "SimpleRNN", "LSTMCell", "LSTM", "EncoderDecoder",
    # CNN
    "Conv2D", "MaxPool2D", "AvgPool2D", "BatchNorm2D", "Flatten", "Dense", "SimpleCNN",
    # Attention / Transformer
    "ScaledDotProductAttention", "MultiHeadAttention", "PositionalEncoding",
    "LayerNorm", "FeedForward", "TransformerEncoderLayer", "TransformerEncoder",
    # GAN
    "Generator", "Discriminator", "GAN",
    # Associative memory
    "HopfieldNetwork",
    # Energy-based
    "RestrictedBoltzmannMachine",
    # RBF
    "RBFNetwork",
    # Complex-valued
    "ComplexDense", "ComplexValuedNN",
]
