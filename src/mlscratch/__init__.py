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
mlscratch.neural            Neural network architectures  (coming soon)

Quick-start
-----------
>>> from mlscratch.unsupervised import KMeans
>>> from mlscratch.supervised import LinearRegression
>>> from mlscratch.bayesian import GaussianNB
>>> from mlscratch.reinforcement import QLearning

Install
-------
    pip install mlscratch              # core (numpy only)
    pip install "mlscratch[dev]"       # + pytest, ruff, black, mypy
    pip install "mlscratch[docs]"      # + mkdocs
    pip install "mlscratch[all]"       # everything

Links
-----
GitHub      : https://github.com/Mattral/ML-AI-Algorithms-from-scratch
Issues      : https://github.com/Mattral/ML-AI-Algorithms-from-scratch/issues
Changelog   : https://github.com/Mattral/ML-AI-Algorithms-from-scratch/blob/main/CHANGELOG.md
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("mlscratch")
except PackageNotFoundError:
    # Package is not installed (e.g. running directly from source tree)
    __version__ = "0.0.0+dev"

__author__  = "Mattral"
__license__ = "Apache-2.0"

__all__ = ["__version__", "__author__", "__license__"]
