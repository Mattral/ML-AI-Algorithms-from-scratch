"""
mlscratch.metrics
==================
Evaluation metrics for classifiers and regressors, implemented from
scratch in pure numpy — no scikit-learn dependency at runtime.

Classification
--------------
accuracy_score, precision_score, recall_score, f1_score,
precision_recall_fscore_support, confusion_matrix, classification_report,
roc_curve, roc_auc_score, log_loss

Regression
----------
mean_squared_error, root_mean_squared_error, mean_absolute_error,
mean_absolute_percentage_error, r2_score, explained_variance_score
"""

from .classification import (  # noqa: F401
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from .regression import (  # noqa: F401
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

__all__ = [
    # Classification
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "precision_recall_fscore_support",
    "confusion_matrix",
    "classification_report",
    "roc_curve",
    "roc_auc_score",
    "log_loss",
    # Regression
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "r2_score",
    "explained_variance_score",
]
