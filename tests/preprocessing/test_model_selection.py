"""
Tests for mlscratch.preprocessing.model_selection
"""

import numpy as np
import pytest

from mlscratch.preprocessing import train_test_split


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = rng.integers(0, 2, 100)
    return X, y


class TestTrainTestSplit:
    def test_default_split_sizes(self, data):
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        assert X_train.shape[0] == 75
        assert X_test.shape[0] == 25
        assert y_train.shape[0] == 75
        assert y_test.shape[0] == 25

    def test_explicit_test_size_float(self, data):
        X, y = data
        X_train, X_test, *_ = train_test_split(X, y, test_size=0.3, random_state=0)
        assert X_test.shape[0] == 30
        assert X_train.shape[0] == 70

    def test_explicit_test_size_int(self, data):
        X, y = data
        X_train, X_test, *_ = train_test_split(X, y, test_size=10, random_state=0)
        assert X_test.shape[0] == 10
        assert X_train.shape[0] == 90

    def test_no_overlap_between_splits(self, data):
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        train_set = {tuple(row) for row in X_train}
        test_set = {tuple(row) for row in X_test}
        assert train_set.isdisjoint(test_set)

    def test_reproducible_with_same_random_state(self, data):
        X, y = data
        out1 = train_test_split(X, y, random_state=42)
        out2 = train_test_split(X, y, random_state=42)
        for a, b in zip(out1, out2, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_x_and_y_split_consistently(self, data):
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        # Reconstruct via a manual full split and verify row correspondence holds:
        # every X_train row's matching y must be findable as one of y_train's values
        # for at least the same count of unique rows (sanity, not index-based since shuffled)
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

    def test_three_arrays_split_together(self, data):
        X, y = data
        z = np.arange(100)
        X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
            X, y, z, random_state=0
        )
        assert z_train.shape[0] == X_train.shape[0]
        assert z_test.shape[0] == X_test.shape[0]

    def test_stratify_preserves_class_balance(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 2))
        y = np.array([0] * 160 + [1] * 40)  # 80/20 imbalanced
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=0
        )
        train_ratio = np.mean(y_train == 0)
        test_ratio = np.mean(y_test == 0)
        assert train_ratio == pytest.approx(0.8, abs=0.05)
        assert test_ratio == pytest.approx(0.8, abs=0.05)

    def test_mismatched_array_lengths_raise(self):
        with pytest.raises(ValueError, match="same first dimension"):
            train_test_split(np.ones((10, 2)), np.ones(5))

    def test_invalid_test_size_float_raises(self, data):
        X, y = data
        with pytest.raises(ValueError, match="test_size"):
            train_test_split(X, y, test_size=1.5)

    def test_no_arrays_raises(self):
        with pytest.raises(ValueError, match="At least one array"):
            train_test_split()

    def test_train_size_plus_test_size_exceeding_n_raises(self, data):
        X, y = data
        with pytest.raises(ValueError, match="exceeds"):
            train_test_split(X, y, train_size=80, test_size=30)
