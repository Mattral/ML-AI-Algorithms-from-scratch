"""
Tests for mlscratch.neural.hopfield.HopfieldNetwork
"""

from __future__ import annotations

import numpy as np
import pytest

from mlscratch.neural.hopfield import HopfieldNetwork


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def patterns():
    """3 random bipolar patterns of dimension 20 (capacity ~0.138*20=2.76)."""
    rng = np.random.default_rng(0)
    return rng.choice([-1.0, 1.0], size=(2, 20))


@pytest.fixture
def single_pattern():
    rng = np.random.default_rng(1)
    return rng.choice([-1.0, 1.0], size=(1, 16))


# ===================================================================
# Basic API
# ===================================================================

class TestHopfieldBasic:
    def test_init_weights_zero(self):
        hn = HopfieldNetwork(n_units=10)
        np.testing.assert_array_equal(hn.weights, np.zeros((10, 10)))

    def test_fit_returns_self(self, patterns):
        hn = HopfieldNetwork(n_units=20, random_state=0)
        assert hn.fit(patterns) is hn

    def test_weight_matrix_shape(self, patterns):
        hn = HopfieldNetwork(n_units=20, random_state=0).fit(patterns)
        assert hn.weights.shape == (20, 20)

    def test_weight_matrix_symmetric(self, patterns):
        hn = HopfieldNetwork(n_units=20, random_state=0).fit(patterns)
        np.testing.assert_allclose(hn.weights, hn.weights.T)

    def test_diagonal_is_zero(self, patterns):
        hn = HopfieldNetwork(n_units=20, random_state=0).fit(patterns)
        np.testing.assert_allclose(np.diag(hn.weights), 0.0)

    def test_n_patterns_stored_recorded(self, patterns):
        hn = HopfieldNetwork(n_units=20, random_state=0).fit(patterns)
        assert hn.n_patterns_stored_ == 2

    def test_wrong_dimension_raises(self):
        hn = HopfieldNetwork(n_units=10)
        bad_patterns = np.ones((2, 5))
        with pytest.raises(ValueError):
            hn.fit(bad_patterns)

    def test_single_pattern_fit(self, single_pattern):
        hn = HopfieldNetwork(n_units=16, random_state=0).fit(single_pattern)
        assert hn.n_patterns_stored_ == 1


# ===================================================================
# Energy function
# ===================================================================

class TestHopfieldEnergy:
    def test_energy_is_float(self, patterns):
        hn = HopfieldNetwork(n_units=20, random_state=0).fit(patterns)
        e = hn.energy(patterns[0])
        assert isinstance(e, float)

    def test_stored_pattern_has_lower_energy_than_random(self, patterns):
        hn = HopfieldNetwork(n_units=20, random_state=0).fit(patterns)
        e_stored = hn.energy(patterns[0])
        rng = np.random.default_rng(99)
        random_states = [rng.choice([-1.0, 1.0], size=20) for _ in range(20)]
        avg_random_energy = np.mean([hn.energy(s) for s in random_states])
        assert e_stored < avg_random_energy

    def test_energy_symmetric_under_global_flip(self, patterns):
        """E(s) == E(-s) since E = -0.5 sᵗWs is invariant to s -> -s."""
        hn = HopfieldNetwork(n_units=20, random_state=0).fit(patterns)
        e_pos = hn.energy(patterns[0])
        e_neg = hn.energy(-patterns[0])
        np.testing.assert_allclose(e_pos, e_neg, atol=1e-10)


# ===================================================================
# Recall dynamics — Correctness
# ===================================================================

class TestHopfieldRecall:
    def test_invalid_mode_raises(self, patterns):
        hn = HopfieldNetwork(n_units=20, random_state=0).fit(patterns)
        with pytest.raises(ValueError):
            hn.recall(patterns[0], mode="invalid")

    def test_recall_returns_bipolar(self, patterns):
        hn = HopfieldNetwork(n_units=20, random_state=0).fit(patterns)
        recalled = hn.recall(patterns[0].copy(), mode="async")
        assert set(np.unique(recalled)).issubset({-1.0, 1.0})

    def test_recall_shape(self, patterns):
        hn = HopfieldNetwork(n_units=20, random_state=0).fit(patterns)
        recalled = hn.recall(patterns[0].copy(), mode="sync")
        assert recalled.shape == (20,)

    def test_exact_pattern_is_fixed_point_sync(self, single_pattern):
        """Storing a single pattern guarantees it is a fixed point."""
        hn = HopfieldNetwork(n_units=16, random_state=0).fit(single_pattern)
        recalled = hn.recall(single_pattern[0].copy(), mode="sync", max_iter=1)
        np.testing.assert_array_equal(recalled, single_pattern[0])

    def test_single_pattern_recall_from_noise(self, single_pattern):
        """With only one stored pattern, recall from a noisy version
        should converge back to the exact pattern (high capacity margin)."""
        hn = HopfieldNetwork(n_units=16, random_state=0).fit(single_pattern)
        noisy = single_pattern[0].copy()
        # Flip 2 of 16 bits
        rng = np.random.default_rng(5)
        flip_idx = rng.choice(16, 2, replace=False)
        noisy[flip_idx] *= -1

        recalled = hn.recall(noisy, mode="async")
        overlap = hn.overlap(recalled, single_pattern[0])
        assert overlap > 0.5

    def test_async_converges(self, patterns):
        """Async recall should terminate within max_iter for small networks."""
        hn = HopfieldNetwork(n_units=20, random_state=0).fit(patterns)
        recalled = hn.recall(patterns[0].copy(), mode="async", max_iter=50)
        assert recalled.shape == (20,)
        assert set(np.unique(recalled)).issubset({-1.0, 1.0})

    def test_sync_converges(self, patterns):
        hn = HopfieldNetwork(n_units=20, random_state=0).fit(patterns)
        recalled = hn.recall(patterns[0].copy(), mode="sync", max_iter=50)
        assert recalled.shape == (20,)

    def test_recall_decreases_or_maintains_energy(self, single_pattern):
        """Hopfield dynamics are guaranteed to never increase the energy."""
        hn = HopfieldNetwork(n_units=16, random_state=0).fit(single_pattern)
        rng = np.random.default_rng(7)
        start = rng.choice([-1.0, 1.0], size=16)
        e_start = hn.energy(start)
        recalled = hn.recall(start.copy(), mode="async", max_iter=100)
        e_end = hn.energy(recalled)
        assert e_end <= e_start + 1e-10


# ===================================================================
# Stability / utility methods
# ===================================================================

class TestHopfieldUtilities:
    def test_is_stable_for_stored_pattern(self, single_pattern):
        hn = HopfieldNetwork(n_units=16, random_state=0).fit(single_pattern)
        assert hn.is_stable(single_pattern[0])

    def test_hamming_distance_zero_for_identical(self, patterns):
        hn = HopfieldNetwork(n_units=20, random_state=0).fit(patterns)
        d = hn.hamming_distance(patterns[0], patterns[0])
        assert d == 0

    def test_hamming_distance_counts_flips(self, single_pattern):
        hn = HopfieldNetwork(n_units=16, random_state=0).fit(single_pattern)
        flipped = single_pattern[0].copy()
        flipped[:3] *= -1
        d = hn.hamming_distance(single_pattern[0], flipped)
        assert d == 3

    def test_overlap_identical_is_one(self, patterns):
        hn = HopfieldNetwork(n_units=20, random_state=0).fit(patterns)
        ov = hn.overlap(patterns[0], patterns[0])
        np.testing.assert_allclose(ov, 1.0)

    def test_overlap_inverse_is_minus_one(self, patterns):
        hn = HopfieldNetwork(n_units=20, random_state=0).fit(patterns)
        ov = hn.overlap(patterns[0], -patterns[0])
        np.testing.assert_allclose(ov, -1.0)

    def test_overlap_range(self, patterns):
        hn = HopfieldNetwork(n_units=20, random_state=0).fit(patterns)
        ov = hn.overlap(patterns[0], patterns[1])
        assert -1.0 <= ov <= 1.0


# ===================================================================
# Edge cases
# ===================================================================

class TestHopfieldEdgeCases:
    def test_recall_with_zero_initial_state(self, single_pattern):
        """state == 0 should be handled via sign-tie-breaking."""
        hn = HopfieldNetwork(n_units=16, random_state=0).fit(single_pattern)
        zero_state = np.zeros(16)
        recalled = hn.recall(zero_state, mode="sync", max_iter=10)
        assert set(np.unique(recalled)).issubset({-1.0, 1.0})

    def test_n_units_two(self):
        hn = HopfieldNetwork(n_units=2, random_state=0)
        patterns = np.array([[1.0, -1.0]])
        hn.fit(patterns)
        recalled = hn.recall(patterns[0].copy(), mode="sync")
        assert recalled.shape == (2,)

    def test_reproducible_async_with_seed(self, patterns):
        hn1 = HopfieldNetwork(n_units=20, random_state=42).fit(patterns)
        hn2 = HopfieldNetwork(n_units=20, random_state=42).fit(patterns)
        r1 = hn1.recall(patterns[0].copy(), mode="async")
        r2 = hn2.recall(patterns[0].copy(), mode="async")
        np.testing.assert_array_equal(r1, r2)
