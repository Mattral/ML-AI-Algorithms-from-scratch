r"""
Classification Metrics
=======================
Evaluation metrics for classifiers, implemented from scratch in pure numpy.

confusion_matrix
-----------------
Row *i*, column *j* counts samples with true label ``labels[i]``
predicted as ``labels[j]``.

precision / recall / F1
-------------------------
Per class *k*, from the confusion matrix:

.. math::
    \mathrm{precision}_k = \frac{TP_k}{TP_k + FP_k}, \quad
    \mathrm{recall}_k = \frac{TP_k}{TP_k + FN_k}, \quad
    F_1 = \frac{2\,\mathrm{precision}\cdot\mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}

``average`` controls how per-class scores are combined: ``'binary'``
(report the positive class only), ``'macro'`` (unweighted mean),
``'micro'`` (global counts pooled across classes), ``'weighted'``
(mean weighted by class support), or ``None`` (return the raw
per-class array).

roc_curve / roc_auc_score
----------------------------
Binary-only. Sweeps the decision threshold over every distinct score
value and reports the true/false positive rate at each one; AUC is
the trapezoidal-rule integral of TPR over FPR.

log_loss
---------
.. math::
    -\frac1n \sum_i \log \hat p_i(y_i)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

_AVERAGE_OPTIONS = {None, "binary", "macro", "micro", "weighted"}

# numpy >= 2.0 renamed trapz -> trapezoid; numpy < 2.0 only has trapz.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ──────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────


def _validate_labels(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[NDArray, NDArray]:
    y_true_arr = np.asarray(y_true).flatten()
    y_pred_arr = np.asarray(y_pred).flatten()
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"y_true has {y_true_arr.shape[0]} samples but y_pred has {y_pred_arr.shape[0]}."
        )
    if y_true_arr.shape[0] == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    return y_true_arr, y_pred_arr


def _safe_divide(numerator: FloatArray, denominator: FloatArray, fill_value: float) -> FloatArray:
    out = np.full_like(numerator, fill_value=float(fill_value), dtype=np.float64)
    mask = denominator > 0
    out[mask] = numerator[mask] / denominator[mask]
    return out


# ──────────────────────────────────────────────────────────────────────────
# Confusion matrix
# ──────────────────────────────────────────────────────────────────────────


def confusion_matrix(
    y_true: ArrayLike, y_pred: ArrayLike, labels: ArrayLike | None = None
) -> NDArray[np.int64]:
    """Return the confusion matrix ``C`` where ``C[i, j]`` is the count of
    samples with true label ``labels[i]`` predicted as ``labels[j]``."""
    y_true_arr, y_pred_arr = _validate_labels(y_true, y_pred)
    if labels is None:
        labels_arr = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
    else:
        labels_arr = np.asarray(labels)

    label_to_idx = {label: i for i, label in enumerate(labels_arr)}
    n = labels_arr.size
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true_arr, y_pred_arr, strict=True):
        ti, pi = label_to_idx.get(t), label_to_idx.get(p)
        if ti is not None and pi is not None:
            cm[ti, pi] += 1
    return cm


def _per_class_counts(
    y_true: NDArray, y_pred: NDArray, labels: NDArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)
    return tp, fp, fn, support


# ──────────────────────────────────────────────────────────────────────────
# Simple scalar metrics
# ──────────────────────────────────────────────────────────────────────────


def accuracy_score(
    y_true: ArrayLike, y_pred: ArrayLike, sample_weight: ArrayLike | None = None
) -> float:
    """Fraction (or weighted fraction) of exactly-correct predictions."""
    y_true_arr, y_pred_arr = _validate_labels(y_true, y_pred)
    correct = (y_true_arr == y_pred_arr).astype(np.float64)
    if sample_weight is None:
        return float(np.mean(correct))
    w = np.asarray(sample_weight, dtype=np.float64).flatten()
    return float(np.average(correct, weights=w))


# ──────────────────────────────────────────────────────────────────────────
# Precision / Recall / F1
# ──────────────────────────────────────────────────────────────────────────


def precision_recall_fscore_support(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: str | None = "binary",
    labels: ArrayLike | None = None,
    pos_label: object = 1,
    zero_division: float = 0.0,
):
    """Compute precision, recall, F1, and support, jointly (one confusion
    matrix pass). Returns 4 scalars if ``average`` is not ``None``, else 4
    arrays of length ``n_classes``."""
    if average not in _AVERAGE_OPTIONS:
        raise ValueError(f"average must be one of {_AVERAGE_OPTIONS}.")
    y_true_arr, y_pred_arr = _validate_labels(y_true, y_pred)
    labels_arr = (
        np.unique(np.concatenate([y_true_arr, y_pred_arr]))
        if labels is None
        else np.asarray(labels)
    )

    tp, fp, fn, support = _per_class_counts(y_true_arr, y_pred_arr, labels_arr)
    precision = _safe_divide(tp, tp + fp, zero_division)
    recall = _safe_divide(tp, tp + fn, zero_division)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall, zero_division)

    if average is None:
        return precision, recall, f1, support

    if average == "binary":
        if labels_arr.size != 2:
            raise ValueError(
                "average='binary' requires exactly 2 classes; use 'macro', "
                "'micro', 'weighted', or None for multiclass problems."
            )
        idx = int(np.flatnonzero(labels_arr == pos_label)[0]) if pos_label in labels_arr else 1
        return float(precision[idx]), float(recall[idx]), float(f1[idx]), int(support[idx])

    if average == "macro":
        return float(precision.mean()), float(recall.mean()), float(f1.mean()), int(support.sum())

    if average == "micro":
        tp_s, fp_s, fn_s = tp.sum(), fp.sum(), fn.sum()
        p = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else zero_division
        r = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else zero_division
        f = 2 * p * r / (p + r) if (p + r) > 0 else zero_division
        return float(p), float(r), float(f), int(support.sum())

    # weighted
    total = support.sum()
    weights = support / total if total > 0 else np.zeros_like(support)
    return (
        float(np.sum(precision * weights)),
        float(np.sum(recall * weights)),
        float(np.sum(f1 * weights)),
        int(total),
    )


def precision_score(y_true: ArrayLike, y_pred: ArrayLike, **kwargs) -> float | FloatArray:
    p, _, _, _ = precision_recall_fscore_support(y_true, y_pred, **kwargs)
    return p


def recall_score(y_true: ArrayLike, y_pred: ArrayLike, **kwargs) -> float | FloatArray:
    _, r, _, _ = precision_recall_fscore_support(y_true, y_pred, **kwargs)
    return r


def f1_score(y_true: ArrayLike, y_pred: ArrayLike, **kwargs) -> float | FloatArray:
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred, **kwargs)
    return f


# ──────────────────────────────────────────────────────────────────────────
# ROC / AUC
# ──────────────────────────────────────────────────────────────────────────


def roc_curve(
    y_true: ArrayLike, y_score: ArrayLike, pos_label: object | None = None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return (fpr, tpr, thresholds) for a binary classification problem,
    sweeping the threshold over every distinct value of ``y_score``."""
    y_true_arr = np.asarray(y_true).flatten()
    y_score_arr = np.asarray(y_score, dtype=np.float64).flatten()
    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError(
            f"y_true has {y_true_arr.shape[0]} samples but y_score has {y_score_arr.shape[0]}."
        )
    classes = np.unique(y_true_arr)
    if classes.size != 2:
        raise ValueError("roc_curve requires a binary y_true (exactly 2 classes).")
    if pos_label is None:
        pos_label = classes[-1]
    y_bin = (y_true_arr == pos_label).astype(np.float64)

    order = np.argsort(-y_score_arr, kind="mergesort")
    y_sorted = y_bin[order]
    scores_sorted = y_score_arr[order]

    distinct_idx = np.flatnonzero(np.diff(scores_sorted))
    threshold_idx = np.r_[distinct_idx, y_sorted.size - 1]

    tps = np.cumsum(y_sorted)[threshold_idx]
    fps = (threshold_idx + 1) - tps

    tps = np.r_[0.0, tps]
    fps = np.r_[0.0, fps]
    thresholds = np.r_[np.inf, scores_sorted[threshold_idx]]

    n_pos, n_neg = y_bin.sum(), y_bin.size - y_bin.sum()
    tpr = tps / n_pos if n_pos > 0 else np.zeros_like(tps)
    fpr = fps / n_neg if n_neg > 0 else np.zeros_like(fps)
    return fpr, tpr, thresholds


def roc_auc_score(y_true: ArrayLike, y_score: ArrayLike, pos_label: object | None = None) -> float:
    """Area under the ROC curve (trapezoidal rule)."""
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    return float(_trapezoid(tpr, fpr))


# ──────────────────────────────────────────────────────────────────────────
# Log loss
# ──────────────────────────────────────────────────────────────────────────


def log_loss(
    y_true: ArrayLike, y_pred_proba: ArrayLike, eps: float = 1e-15, labels: ArrayLike | None = None
) -> float:
    """Cross-entropy / binomial-or-multinomial deviance.

    ``y_pred_proba`` may be a 1-D array of positive-class probabilities
    (binary shorthand) or a 2-D array of shape ``(n_samples, n_classes)``.
    """
    y_true_arr = np.asarray(y_true).flatten()
    proba = np.asarray(y_pred_proba, dtype=np.float64)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    if proba.shape[0] != y_true_arr.shape[0]:
        raise ValueError(
            f"y_true has {y_true_arr.shape[0]} samples but y_pred_proba has {proba.shape[0]}."
        )

    proba = np.clip(proba, eps, 1.0 - eps)
    proba = proba / proba.sum(axis=1, keepdims=True)

    labels_arr = np.unique(y_true_arr) if labels is None else np.asarray(labels)
    if labels_arr.size != proba.shape[1]:
        raise ValueError(
            f"y_pred_proba has {proba.shape[1]} columns but {labels_arr.size} labels were found."
        )
    label_to_idx = {label: i for i, label in enumerate(labels_arr)}
    col = np.array([label_to_idx[t] for t in y_true_arr])
    n = y_true_arr.shape[0]
    return float(-np.mean(np.log(proba[np.arange(n), col])))


# ──────────────────────────────────────────────────────────────────────────
# Classification report
# ──────────────────────────────────────────────────────────────────────────


def classification_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: ArrayLike | None = None,
    target_names: list[str] | None = None,
    digits: int = 2,
    zero_division: float = 0.0,
) -> str:
    """Return a sklearn-style formatted text report of the main per-class
    classification metrics, plus accuracy and macro/weighted averages."""
    y_true_arr, y_pred_arr = _validate_labels(y_true, y_pred)
    labels_arr = (
        np.unique(np.concatenate([y_true_arr, y_pred_arr]))
        if labels is None
        else np.asarray(labels)
    )
    names = (
        [str(n) for n in target_names]
        if target_names is not None
        else [str(label) for label in labels_arr]
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average=None, labels=labels_arr, zero_division=zero_division
    )
    acc = accuracy_score(y_true_arr, y_pred_arr)
    macro = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", labels=labels_arr, zero_division=zero_division
    )
    weighted = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="weighted", labels=labels_arr, zero_division=zero_division
    )

    headers = ["precision", "recall", "f1-score", "support"]
    name_width = max(len(n) for n in names + ["weighted avg", "macro avg"])
    col_width = max(len(h) for h in headers) + 2

    lines = []
    header_line = " " * (name_width + 2) + "".join(h.rjust(col_width) for h in headers)
    lines.append(header_line)
    lines.append("")
    for name, p, r, f, s in zip(names, precision, recall, f1, support, strict=True):
        row = (
            f"{name:<{name_width}}  "
            + "".join(f"{v:>{col_width}.{digits}f}" for v in (p, r, f))
            + f"{int(s):>{col_width}d}"
        )
        lines.append(row)
    lines.append("")

    total_support = int(support.sum())
    acc_row = (
        f"{'accuracy':<{name_width}}  "
        + " " * (col_width * 2)
        + f"{acc:>{col_width}.{digits}f}"
        + f"{total_support:>{col_width}d}"
    )
    lines.append(acc_row)

    for label, (p, r, f, s) in (("macro avg", macro), ("weighted avg", weighted)):
        row = (
            f"{label:<{name_width}}  "
            + "".join(f"{v:>{col_width}.{digits}f}" for v in (p, r, f))
            + f"{int(s):>{col_width}d}"
        )
        lines.append(row)

    return "\n".join(lines)
