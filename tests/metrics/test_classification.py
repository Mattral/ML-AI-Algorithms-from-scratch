"""
Tests for mlscratch.metrics.classification
"""

import numpy as np
import pytest

from mlscratch.metrics import (
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_labels():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    return y_true, y_pred


@pytest.fixture
def multiclass_labels():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 3, 200)
    y_pred = rng.integers(0, 3, 200)
    return y_true, y_pred


# ============================================================
# accuracy_score
# ============================================================


class TestAccuracyScore:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 1, 0, 1])
        assert accuracy_score(y, y) == 1.0

    def test_matches_manual_computation(self, binary_labels):
        y_true, y_pred = binary_labels
        assert accuracy_score(y_true, y_pred) == np.mean(y_true == y_pred)

    def test_agrees_with_sklearn(self, multiclass_labels):
        from sklearn.metrics import accuracy_score as sk_accuracy

        y_true, y_pred = multiclass_labels
        assert accuracy_score(y_true, y_pred) == pytest.approx(sk_accuracy(y_true, y_pred))

    def test_weighted_accuracy(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        w = np.array([1.0, 1.0, 1.0, 3.0])
        # correct: idx 0 (w=1), idx 2 (w=1), idx 3 (w=3) -> 5/6
        assert accuracy_score(y_true, y_pred, sample_weight=w) == pytest.approx(5.0 / 6.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="samples"):
            accuracy_score(np.ones(5), np.ones(3))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            accuracy_score(np.array([]), np.array([]))


# ============================================================
# confusion_matrix
# ============================================================


class TestConfusionMatrix:
    def test_matches_sklearn_binary(self, binary_labels):
        from sklearn.metrics import confusion_matrix as sk_cm

        y_true, y_pred = binary_labels
        np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), sk_cm(y_true, y_pred))

    def test_matches_sklearn_multiclass(self, multiclass_labels):
        from sklearn.metrics import confusion_matrix as sk_cm

        y_true, y_pred = multiclass_labels
        np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), sk_cm(y_true, y_pred))

    def test_diagonal_is_correct_count(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        cm = confusion_matrix(y_true, y_pred)
        assert cm.sum() == 5
        assert np.trace(cm) == 3  # 3 correct predictions


# ============================================================
# precision / recall / f1
# ============================================================


class TestPrecisionRecallF1:
    def test_binary_agrees_with_sklearn(self, binary_labels):
        from sklearn.metrics import f1_score as sk_f1
        from sklearn.metrics import precision_score as sk_p
        from sklearn.metrics import recall_score as sk_r

        y_true, y_pred = binary_labels
        assert precision_score(y_true, y_pred) == pytest.approx(sk_p(y_true, y_pred))
        assert recall_score(y_true, y_pred) == pytest.approx(sk_r(y_true, y_pred))
        assert f1_score(y_true, y_pred) == pytest.approx(sk_f1(y_true, y_pred))

    @pytest.mark.parametrize("average", ["macro", "micro", "weighted"])
    def test_multiclass_averages_agree_with_sklearn(self, multiclass_labels, average):
        from sklearn.metrics import precision_score as sk_p

        y_true, y_pred = multiclass_labels
        assert precision_score(y_true, y_pred, average=average) == pytest.approx(
            sk_p(y_true, y_pred, average=average)
        )

    def test_average_none_returns_per_class_array(self, multiclass_labels):
        y_true, y_pred = multiclass_labels
        p = precision_score(y_true, y_pred, average=None)
        assert p.shape == (3,)

    def test_perfect_classifier_has_precision_recall_f1_one(self):
        y = np.array([0, 1, 0, 1, 1])
        assert precision_score(y, y) == 1.0
        assert recall_score(y, y) == 1.0
        assert f1_score(y, y) == 1.0

    def test_zero_division_returns_fill_value(self):
        # A class with no predicted positives -> precision undefined -> 0 by default
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        # pos_label=1 never appears in y_pred or y_true
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=1, labels=np.array([0, 1])
        )
        assert p == 0.0
        assert s == 0

    def test_invalid_average_raises(self):
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="average"):
            precision_score(y, y, average="bogus")

    def test_binary_average_requires_two_classes(self, multiclass_labels):
        y_true, y_pred = multiclass_labels
        with pytest.raises(ValueError, match="binary"):
            precision_score(y_true, y_pred, average="binary")


# ============================================================
# ROC / AUC
# ============================================================


class TestROCAUC:
    def test_perfect_separation_gives_auc_one(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert roc_auc_score(y_true, y_score) == pytest.approx(1.0)

    def test_random_scores_give_auc_near_half(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 2000)
        y_score = rng.uniform(0, 1, 2000)
        assert roc_auc_score(y_true, y_score) == pytest.approx(0.5, abs=0.05)

    def test_agrees_with_sklearn(self):
        from sklearn.metrics import roc_auc_score as sk_auc

        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 2, 300)
        y_score = rng.uniform(0, 1, 300)
        assert roc_auc_score(y_true, y_score) == pytest.approx(sk_auc(y_true, y_score), abs=1e-9)

    def test_roc_curve_endpoints(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.6, 0.9])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        assert fpr[0] == 0.0 and tpr[0] == 0.0
        assert fpr[-1] == 1.0 and tpr[-1] == 1.0

    def test_requires_binary_labels(self):
        y_true = np.array([0, 1, 2])
        y_score = np.array([0.1, 0.5, 0.9])
        with pytest.raises(ValueError, match="binary"):
            roc_curve(y_true, y_score)


# ============================================================
# log_loss
# ============================================================


class TestLogLoss:
    def test_perfect_confident_predictions_near_zero(self):
        y_true = np.array([0, 1, 0, 1])
        proba = np.array([0.001, 0.999, 0.001, 0.999])
        assert log_loss(y_true, proba) == pytest.approx(0.0, abs=1e-2)

    def test_agrees_with_sklearn(self):
        from sklearn.metrics import log_loss as sk_ll

        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 100)
        p1 = rng.uniform(0.01, 0.99, 100)
        proba2d = np.column_stack([1 - p1, p1])
        assert log_loss(y_true, p1) == pytest.approx(sk_ll(y_true, proba2d), abs=1e-9)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="samples"):
            log_loss(np.ones(5), np.ones(3))


# ============================================================
# classification_report
# ============================================================


class TestClassificationReport:
    def test_returns_string_with_expected_headers(self, multiclass_labels):
        y_true, y_pred = multiclass_labels
        report = classification_report(y_true, y_pred)
        assert isinstance(report, str)
        for header in ("precision", "recall", "f1-score", "support", "accuracy"):
            assert header in report

    def test_custom_target_names_appear(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        report = classification_report(y_true, y_pred, target_names=["cat", "dog"])
        assert "cat" in report
        assert "dog" in report
