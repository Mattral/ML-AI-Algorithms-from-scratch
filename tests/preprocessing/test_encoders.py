"""
Tests for mlscratch.preprocessing.encoders
"""

import numpy as np
import pytest

from mlscratch.preprocessing import LabelEncoder, OneHotEncoder

# ============================================================
# LabelEncoder
# ============================================================


class TestLabelEncoder:
    def test_fit_assigns_sorted_classes(self):
        y = np.array(["dog", "cat", "bird", "cat"])
        enc = LabelEncoder().fit(y)
        np.testing.assert_array_equal(enc.classes_, ["bird", "cat", "dog"])

    def test_transform_matches_sklearn(self):
        from sklearn.preprocessing import LabelEncoder as SKLabelEncoder

        y = np.array(["dog", "cat", "bird", "cat", "dog"])
        ours = LabelEncoder().fit_transform(y)
        theirs = SKLabelEncoder().fit_transform(y)
        np.testing.assert_array_equal(ours, theirs)

    def test_inverse_transform_round_trips(self):
        y = np.array([3, 1, 2, 1, 3])
        enc = LabelEncoder().fit(y)
        codes = enc.transform(y)
        np.testing.assert_array_equal(enc.inverse_transform(codes), y)

    def test_unseen_label_raises(self):
        enc = LabelEncoder().fit(np.array(["a", "b"]))
        with pytest.raises(ValueError, match="unseen"):
            enc.transform(np.array(["a", "c"]))

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            LabelEncoder().transform(np.array([1, 2]))

    def test_empty_fit_raises(self):
        with pytest.raises(ValueError):
            LabelEncoder().fit(np.array([]))

    def test_inverse_transform_out_of_range_raises(self):
        enc = LabelEncoder().fit(np.array(["a", "b"]))
        with pytest.raises(ValueError, match="range"):
            enc.inverse_transform(np.array([5]))


# ============================================================
# OneHotEncoder
# ============================================================


class TestOneHotEncoder:
    def test_basic_encoding_matches_sklearn(self):
        from sklearn.preprocessing import OneHotEncoder as SKOneHotEncoder

        X = np.array([["a"], ["b"], ["a"], ["c"]])
        ours = OneHotEncoder().fit_transform(X)
        theirs = SKOneHotEncoder(sparse_output=False).fit_transform(X)
        np.testing.assert_allclose(ours, theirs)

    def test_multi_column_independent_encoding(self):
        X = np.array([["a", "x"], ["b", "y"], ["a", "x"]])
        enc = OneHotEncoder().fit(X)
        Xt = enc.transform(X)
        # 2 categories in col0 + 2 categories in col1 = 4 output columns
        assert Xt.shape == (3, 4)

    def test_each_row_sums_to_n_columns(self):
        X = np.array([["a"], ["b"], ["c"]])
        Xt = OneHotEncoder().fit_transform(X)
        np.testing.assert_array_equal(Xt.sum(axis=1), np.ones(3))

    def test_drop_first(self):
        X = np.array([["a"], ["b"], ["c"]])
        enc = OneHotEncoder(drop_first=True).fit(X)
        Xt = enc.transform(X)
        assert Xt.shape == (3, 2)  # 3 categories - 1 dropped = 2 columns

    def test_unknown_category_raises_by_default(self):
        enc = OneHotEncoder().fit(np.array([["a"], ["b"]]))
        with pytest.raises(ValueError, match="Unknown category"):
            enc.transform(np.array([["c"]]))

    def test_unknown_category_ignored_when_configured(self):
        enc = OneHotEncoder(handle_unknown="ignore").fit(np.array([["a"], ["b"]]))
        Xt = enc.transform(np.array([["c"]]))
        np.testing.assert_array_equal(Xt, np.zeros((1, 2)))

    def test_get_feature_names(self):
        X = np.array([["red"], ["blue"]])
        enc = OneHotEncoder().fit(X)
        names = enc.get_feature_names(["color"])
        assert names == ["color_blue", "color_red"]

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            OneHotEncoder().transform(np.array([["a"]]))

    def test_1d_input_is_treated_as_single_column(self):
        X = np.array(["a", "b", "a"])
        Xt = OneHotEncoder().fit_transform(X)
        assert Xt.shape == (3, 2)

    def test_invalid_handle_unknown_raises(self):
        with pytest.raises(ValueError, match="handle_unknown"):
            OneHotEncoder(handle_unknown="bogus")
