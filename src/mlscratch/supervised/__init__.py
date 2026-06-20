"""
mlscratch.supervised
====================
Supervised learning algorithms, implemented from scratch in pure numpy.

Linear models
-------------
LinearRegression, RidgeRegression, LassoRegression, ElasticNet,
LogisticRegression

Instance-based
--------------
KNeighboursClassifier, KNeighboursRegressor

Tree-based
----------
DecisionTreeClassifier, DecisionTreeRegressor

Ensembles
---------
RandomForestClassifier, RandomForestRegressor
GradientBoostingClassifier, GradientBoostingRegressor
AdaBoostClassifier

Kernel methods
--------------
SVC — kernel Support Vector Classifier (linear / poly / rbf / sigmoid),
trained via Sequential Minimal Optimization.
"""

from .adaboost import AdaBoostClassifier  # noqa: F401
from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: F401
from .gradient_boosting import (  # noqa: F401
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from .knn import (  # noqa: F401
    KNeighborsClassifier,
    KNeighborsRegressor,
    KNeighboursClassifier,
    KNeighboursRegressor,
)
from .linear_models import (  # noqa: F401
    ElasticNet,
    LassoRegression,
    LinearRegression,
    LogisticRegression,
    RidgeRegression,
)
from .random_forest import RandomForestClassifier, RandomForestRegressor  # noqa: F401
from .svm import SVC  # noqa: F401

__all__ = [
    # Linear models
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "ElasticNet",
    "LogisticRegression",
    # Instance-based
    "KNeighboursClassifier",
    "KNeighboursRegressor",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    # Tree-based
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    # Ensembles
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "AdaBoostClassifier",
    # Kernel methods
    "SVC",
]
