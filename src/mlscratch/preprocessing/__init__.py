"""
mlscratch.preprocessing
=========================
Feature scaling, categorical encoding, polynomial feature expansion,
and train/test splitting utilities — pure numpy, sklearn-familiar API.

Scalers
-------
StandardScaler, MinMaxScaler, RobustScaler, Normalizer

Encoders
--------
LabelEncoder, OneHotEncoder

Feature expansion
------------------
PolynomialFeatures

Splitting
---------
train_test_split
"""

from .encoders import LabelEncoder, OneHotEncoder  # noqa: F401
from .model_selection import train_test_split  # noqa: F401
from .polynomial import PolynomialFeatures  # noqa: F401
from .scalers import MinMaxScaler, Normalizer, RobustScaler, StandardScaler  # noqa: F401

__all__ = [
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "Normalizer",
    "LabelEncoder",
    "OneHotEncoder",
    "PolynomialFeatures",
    "train_test_split",
]
