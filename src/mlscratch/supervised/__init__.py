"""
mlscratch.supervised
====================
Supervised learning algorithms (in progress).
Currently implemented: LinearRegression, RidgeRegression, LassoRegression,
ElasticNet, LogisticRegression, KNeighboursClassifier, KNeighboursRegressor.

Coming: DecisionTree, RandomForest, SVM, GradientBoosting, AdaBoost.
"""
from .linear_models import (                     # noqa: F401
    LinearRegression, RidgeRegression,
    LassoRegression, ElasticNet, LogisticRegression,
)
from .knn import KNeighboursClassifier, KNeighboursRegressor  # noqa: F401

__all__ = [
    "LinearRegression", "RidgeRegression", "LassoRegression",
    "ElasticNet", "LogisticRegression",
    "KNeighboursClassifier", "KNeighboursRegressor",
]
