"""
Naive Bayes Classifiers
========================
Three variants of the Naive Bayes family, all built on Bayes' theorem with
the "naive" conditional-independence assumption among features.

    P(y | x) ∝ P(y) * ∏_i P(x_i | y)

Variants
--------
GaussianNB       – continuous features modelled as Gaussians
MultinomialNB    – integer/count features (e.g. word counts)
BernoulliNB      – binary features (0/1)

All log-probabilities are used internally to avoid underflow.
Only numpy and Python stdlib are used.
"""

import numpy as np


# ============================================================
# Base
# ============================================================

class _BaseNB:
    """Shared prediction logic for all NB variants."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([
            self.classes_[np.argmax(self._joint_log_likelihood(x))]
            for x in X
        ])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_probs = np.array([self._joint_log_likelihood(x) for x in X])
        # Numerically stable softmax per row
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def _joint_log_likelihood(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ============================================================
# Gaussian Naive Bayes
# ============================================================

class GaussianNB(_BaseNB):
    """
    Gaussian Naive Bayes for continuous features.

    Likelihood: P(x_i | y) = N(x_i; mu_{iy}, sigma_{iy}^2)

    Parameters
    ----------
    var_smoothing : float
        Small value added to variance for numerical stability.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_prior_ = None    # log P(y)
        self.theta_ = None          # means:    (n_classes, n_features)
        self.sigma_ = None          # variances:(n_classes, n_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        n_samples = len(y)

        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)

        for k, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.theta_[k] = Xc.mean(axis=0)
            self.sigma_[k] = Xc.var(axis=0) + self.var_smoothing
            self.class_prior_[k] = len(Xc) / n_samples

        return self

    def _joint_log_likelihood(self, x: np.ndarray) -> np.ndarray:
        jll = np.log(self.class_prior_).copy()
        for k in range(len(self.classes_)):
            log_pdf = -0.5 * np.sum(
                np.log(2 * np.pi * self.sigma_[k])
                + ((x - self.theta_[k]) ** 2) / self.sigma_[k]
            )
            jll[k] += log_pdf
        return jll


# ============================================================
# Multinomial Naive Bayes
# ============================================================

class MultinomialNB(_BaseNB):
    """
    Multinomial Naive Bayes for count/frequency features.

    Likelihood: P(x_i | y) ∝ theta_{iy}^{x_i}

    Parameters
    ----------
    alpha : float
        Laplace / Lidstone smoothing parameter (default 1.0).
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_prior_ = None     # log P(y)
        self.feature_log_prob_ = None  # log P(x_i | y): (n_classes, n_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNB":
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples = len(y)

        self.class_prior_ = np.zeros(n_classes)
        feature_counts = np.zeros((n_classes, X.shape[1]))

        for k, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.class_prior_[k] = len(Xc) / n_samples
            feature_counts[k] = Xc.sum(axis=0)

        # Smoothed log probabilities
        smoothed = feature_counts + self.alpha
        self.feature_log_prob_ = np.log(smoothed) - np.log(
            smoothed.sum(axis=1, keepdims=True)
        )
        return self

    def _joint_log_likelihood(self, x: np.ndarray) -> np.ndarray:
        return np.log(self.class_prior_) + self.feature_log_prob_ @ x


# ============================================================
# Bernoulli Naive Bayes
# ============================================================

class BernoulliNB(_BaseNB):
    """
    Bernoulli Naive Bayes for binary features.

    Likelihood: P(x_i | y) = p_{iy}^{x_i} * (1-p_{iy})^{1-x_i}

    Parameters
    ----------
    alpha : float
        Laplace smoothing (default 1.0).
    binarize : float or None
        Threshold to binarize continuous inputs.  If None, assume already
        binary.
    """

    def __init__(self, alpha: float = 1.0, binarize: float | None = 0.0):
        self.alpha = alpha
        self.binarize = binarize
        self.classes_ = None
        self.class_prior_ = None
        self.feature_log_prob_ = None    # log P(x_i=1 | y)
        self.feature_log_prob_neg_ = None  # log P(x_i=0 | y)

    def _binarize(self, X: np.ndarray) -> np.ndarray:
        if self.binarize is not None:
            return (X > self.binarize).astype(float)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BernoulliNB":
        X = self._binarize(X)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples = len(y)

        self.class_prior_ = np.zeros(n_classes)
        pos_count = np.zeros((n_classes, X.shape[1]))

        for k, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.class_prior_[k] = len(Xc) / n_samples
            pos_count[k] = Xc.sum(axis=0)

        n_per_class = np.array([(y == c).sum() for c in self.classes_])
        smoothed_pos = pos_count + self.alpha
        smoothed_total = n_per_class[:, np.newaxis] + 2 * self.alpha

        self.feature_log_prob_ = np.log(smoothed_pos / smoothed_total)
        self.feature_log_prob_neg_ = np.log(
            1.0 - smoothed_pos / smoothed_total
        )
        return self

    def _joint_log_likelihood(self, x: np.ndarray) -> np.ndarray:
        x = self._binarize(x)
        jll = np.log(self.class_prior_).copy()
        for k in range(len(self.classes_)):
            jll[k] += np.sum(
                x * self.feature_log_prob_[k]
                + (1 - x) * self.feature_log_prob_neg_[k]
            )
        return jll
