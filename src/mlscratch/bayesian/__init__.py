"""
mlscratch.bayesian
==================
From-scratch implementations of Bayesian learning algorithms.
Drop these files alongside existing code in src/mlscratch/bayesian/.

Algorithms
----------
GaussianNB                  – Gaussian Naive Bayes
MultinomialNB               – Multinomial Naive Bayes
BernoulliNB                 – Bernoulli Naive Bayes
BayesianLinearRegression    – Conjugate Gaussian prior regression
GaussianProcessRegressor    – GP Regression (RBF, Matern52, Linear, Periodic)
RBFKernel                   – RBF / Squared-Exponential kernel
Matern52Kernel              – Matern 5/2 kernel
LinearKernel                – Linear kernel
PeriodicKernel              – Periodic kernel
HiddenMarkovModel           – Discrete HMM (forward-backward, Viterbi, Baum-Welch)
BayesianNeuralNetwork       – BNN via mean-field variational inference
BayesianNetwork             – Discrete DAG (variable elimination, sampling)
KalmanFilter                – Linear Kalman Filter + RTS Smoother
"""

from .naive_bayes import GaussianNB, MultinomialNB, BernoulliNB  # noqa: F401
from .bayesian_linear_regression import BayesianLinearRegression  # noqa: F401
from .gaussian_process import (                                    # noqa: F401
    GaussianProcessRegressor,
    RBFKernel,
    Matern52Kernel,
    LinearKernel,
    PeriodicKernel,
)
from .hmm import HiddenMarkovModel                                 # noqa: F401
from .bayesian_nn import BayesianNeuralNetwork                     # noqa: F401
from .bayesian_network import BayesianNetwork                      # noqa: F401
from .kalman_filter import KalmanFilter                            # noqa: F401

__all__ = [
    "GaussianNB",
    "MultinomialNB",
    "BernoulliNB",
    "BayesianLinearRegression",
    "GaussianProcessRegressor",
    "RBFKernel",
    "Matern52Kernel",
    "LinearKernel",
    "PeriodicKernel",
    "HiddenMarkovModel",
    "BayesianNeuralNetwork",
    "BayesianNetwork",
    "KalmanFilter",
]
