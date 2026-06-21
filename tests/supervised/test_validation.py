"""
Tests for mlscratch.supervised._validation

These helpers are private (leading underscore on the module), but since
five independent estimators (DecisionTree, RandomForest, SVC,
GradientBoosting, AdaBoost) all delegate their input validation to this
one module, its contract is tested directly here rather than relying
solely on indirect coverage through each estimator's own test suite.
"""

import numpy as np
import pytest

from mlscratch.supervised._validation import validate_sample_weight, validate_x, validate_xy


class TestValidateX:
    def test_accepts_2d_array(self):
        X = np.ones((5, 3))
        out = validate_x(X)
        np.testing.assert_array_equal(out, X)

    def test_coerces_list_input(self):
        out = validate_x([[1, 2], [3, 4]])
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64

    def test_coerces_to_float64(self):
        out = validate_x(np.array([[1, 2], [3, 4]], dtype=np.int32))
        assert out.dtype == np.float64

    def test_rejects_1d_array(self):
        with pytest.raises(ValueError, match="2D"):
            validate_x(np.ones(5))

    def test_rejects_3d_array(self):
        with pytest.raises(ValueError, match="2D"):
            validate_x(np.ones((2, 3, 4)))

    def test_does_not_mutate_input(self):
        X = np.ones((3, 2))
        validate_x(X)
        np.testing.assert_array_equal(X, np.ones((3, 2)))


class TestValidateXy:
    def test_matching_lengths_pass_through(self):
        X = np.ones((5, 2))
        y = np.arange(5)
        X_out, y_out = validate_xy(X, y)
        assert X_out.shape == (5, 2)
        assert y_out.shape == (5,)

    def test_flattens_column_vector_y(self):
        X = np.ones((4, 2))
        y = np.arange(4).reshape(-1, 1)
        _, y_out = validate_xy(X, y)
        assert y_out.ndim == 1
        assert y_out.shape == (4,)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="samples"):
            validate_xy(np.ones((10, 2)), np.ones(5))

    def test_error_message_reports_both_counts(self):
        with pytest.raises(ValueError, match=r"10 samples.*5"):
            validate_xy(np.ones((10, 2)), np.ones(5))

    def test_propagates_2d_check_from_validate_x(self):
        with pytest.raises(ValueError, match="2D"):
            validate_xy(np.ones(5), np.ones(5))


class TestValidateSampleWeight:
    def test_none_returns_uniform_weights(self):
        w = validate_sample_weight(None, 7)
        np.testing.assert_array_equal(w, np.ones(7))
        assert w.dtype == np.float64

    def test_valid_weights_pass_through(self):
        weights = [0.5, 1.0, 2.0]
        w = validate_sample_weight(weights, 3)
        np.testing.assert_array_equal(w, [0.5, 1.0, 2.0])

    def test_zero_weights_are_allowed(self):
        w = validate_sample_weight([0.0, 1.0, 0.0], 3)
        np.testing.assert_array_equal(w, [0.0, 1.0, 0.0])

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="entries"):
            validate_sample_weight(np.ones(5), 10)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_sample_weight([1.0, -0.5, 2.0], 3)
