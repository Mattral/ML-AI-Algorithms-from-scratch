import numpy as np
import pytest

from mlscratch.unsupervised.kmeans import KMeans


def _best_label_mapping(true_labels, predicted_labels):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(true_labels, predicted_labels)
    row_ind, col_ind = linear_sum_assignment(matrix.max() - matrix)
    new_labels = np.zeros_like(predicted_labels)
    for true_label, pred_label in zip(row_ind, col_ind):
        new_labels[predicted_labels == pred_label] = true_label
    return new_labels


def test_kmeans_recovers_blobs():
    """KMeans should recover well-separated clusters from blob data."""
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans as SKKMeans

    X, y = make_blobs(
        n_samples=240,
        centers=3,
        cluster_std=0.45,
        random_state=0,
    )

    model = KMeans(n_clusters=3, random_state=0).fit(X)
    assert model.cluster_centers_.shape == (3, X.shape[1])
    assert model.labels_.shape == (X.shape[0],)

    theirs = SKKMeans(n_clusters=3, random_state=0, init="k-means++", n_init=1, max_iter=300).fit(X)
    mapped = _best_label_mapping(y, model.labels_)
    assert np.mean(mapped == y) >= 0.90
    assert model.inertia_ <= 1.2 * theirs.inertia_


def test_kmeans_predict_before_fit_raises():
    """predict() before fit() must raise RuntimeError."""
    model = KMeans(n_clusters=2)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.ones((5, 2)))


def test_kmeans_invalid_input_raises():
    """Invalid input shape must raise ValueError."""
    model = KMeans(n_clusters=2)
    with pytest.raises(ValueError, match="2D"):
        model.fit(np.ones((5,)))
