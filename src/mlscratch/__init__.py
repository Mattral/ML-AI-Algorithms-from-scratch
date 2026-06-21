"""
mlscratch
=========
Pure-NumPy from-scratch implementations of ML / AI / RL / Bayesian algorithms.
No PyTorch. No TensorFlow. No scikit-learn. Just numpy and the maths.

Sub-packages
------------
mlscratch.supervised        Supervised learning algorithms
mlscratch.unsupervised      Unsupervised learning algorithms
mlscratch.bayesian          Bayesian methods
mlscratch.reinforcement     Reinforcement learning algorithms
mlscratch.neural            Neural network architectures
mlscratch.metrics           Classification & regression evaluation metrics
mlscratch.preprocessing     Scalers, encoders, polynomial features, train_test_split

Quick-start
-----------
>>> from mlscratch.unsupervised import KMeans
>>> from mlscratch.supervised import LinearRegression, RandomForestClassifier
>>> from mlscratch.bayesian import GaussianNB
>>> from mlscratch.reinforcement import QLearning
>>> from mlscratch.neural import MultiLayerPerceptron
>>> from mlscratch.metrics import accuracy_score
>>> from mlscratch.preprocessing import StandardScaler, train_test_split

Install
-------
    pip install scratchkit              # core (numpy only); import name is "mlscratch"
    pip install "scratchkit[dev]"       # + pytest, ruff, black, mypy
    pip install "scratchkit[docs]"      # + mkdocs
    pip install "scratchkit[all]"       # everything

Links
-----
GitHub      : https://github.com/Mattral/ML-AI-Algorithms-from-scratch
Issues      : https://github.com/Mattral/ML-AI-Algorithms-from-scratch/issues
Changelog   : https://github.com/Mattral/ML-AI-Algorithms-from-scratch/blob/main/CHANGELOG.md
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # The importable package is "mlscratch", but the PyPI distribution it
    # ships under is "scratchkit" (the name "mlscratch" was already taken
    # by an unrelated project) — look up metadata under the distribution
    # name, the same way `import bs4` resolves "beautifulsoup4" metadata.
    __version__: str = version("scratchkit")
except PackageNotFoundError:
    # Package is not installed (e.g. running directly from source tree)
    __version__ = "0.0.0+dev"

__author__  = "Mattral"
__license__ = "Apache-2.0"

__all__ = ["__version__", "__author__", "__license__"]
