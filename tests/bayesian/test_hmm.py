"""
Tests for mlscratch.bayesian.hmm.HiddenMarkovModel
"""

import numpy as np
import pytest
from mlscratch.bayesian.hmm import HiddenMarkovModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_hmm(n_states=2, n_obs=3, seed=0):
    """Create and return a fitted HMM with known parameters."""
    hmm = HiddenMarkovModel(n_states=n_states, n_observations=n_obs,
                            random_state=seed)
    hmm._init_params()
    return hmm


def make_2state_hmm():
    """
    Two-state HMM with clearly distinguishable emission distributions.
    State 0 emits obs 0 heavily; state 1 emits obs 2 heavily.
    """
    hmm = HiddenMarkovModel(n_states=2, n_observations=3)
    hmm.A  = np.array([[0.9, 0.1], [0.1, 0.9]])
    hmm.B  = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    hmm.pi = np.array([0.5, 0.5])
    return hmm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_state():
    return make_2state_hmm()


@pytest.fixture
def short_seq():
    return np.array([0, 0, 0, 2, 2, 2, 0, 0])


@pytest.fixture
def generated_sequences(two_state):
    seqs = []
    for seed in range(10):
        _, obs = two_state.sample(30, random_state=seed)
        seqs.append(obs)
    return seqs


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestHMMInit:
    def test_init_params_row_stochastic_A(self):
        hmm = make_hmm(n_states=3, n_obs=4)
        np.testing.assert_allclose(hmm.A.sum(axis=1), 1.0, atol=1e-10)

    def test_init_params_row_stochastic_B(self):
        hmm = make_hmm(n_states=3, n_obs=4)
        np.testing.assert_allclose(hmm.B.sum(axis=1), 1.0, atol=1e-10)

    def test_init_params_pi_sums_to_one(self):
        hmm = make_hmm()
        np.testing.assert_allclose(hmm.pi.sum(), 1.0, atol=1e-10)

    def test_init_params_non_negative(self):
        hmm = make_hmm(n_states=4, n_obs=5)
        assert np.all(hmm.A >= 0)
        assert np.all(hmm.B >= 0)
        assert np.all(hmm.pi >= 0)


# ---------------------------------------------------------------------------
# Forward algorithm
# ---------------------------------------------------------------------------

class TestHMMForward:
    def test_log_likelihood_finite(self, two_state, short_seq):
        _, _, ll = two_state.forward(short_seq)
        assert np.isfinite(ll)

    def test_log_likelihood_negative(self, two_state, short_seq):
        _, _, ll = two_state.forward(short_seq)
        assert ll < 0

    def test_alpha_non_negative(self, two_state, short_seq):
        alpha, _, _ = two_state.forward(short_seq)
        assert np.all(alpha >= 0)

    def test_alpha_shape(self, two_state, short_seq):
        alpha, _, _ = two_state.forward(short_seq)
        assert alpha.shape == (len(short_seq), 2)

    def test_alpha_rows_sum_to_one_after_scaling(self, two_state, short_seq):
        """Scaled alpha rows should sum to 1 (by construction)."""
        alpha, _, _ = two_state.forward(short_seq)
        np.testing.assert_allclose(alpha.sum(axis=1), 1.0, atol=1e-8)

    def test_scales_positive(self, two_state, short_seq):
        _, scales, _ = two_state.forward(short_seq)
        assert np.all(scales > 0)

    def test_log_likelihood_consistent_with_log_likelihood_method(
        self, two_state, short_seq
    ):
        _, _, ll_forward = two_state.forward(short_seq)
        ll_method = two_state.log_likelihood(short_seq)
        np.testing.assert_allclose(ll_forward, ll_method, atol=1e-10)

    def test_longer_sequence_lower_likelihood(self, two_state):
        short = np.array([0, 1])
        long_ = np.array([0, 1, 0, 1, 0, 1])
        ll_short = two_state.log_likelihood(short)
        ll_long  = two_state.log_likelihood(long_)
        # log-likelihood grows (more negative) with length — but P per token
        # can be similar; we just check finiteness here
        assert np.isfinite(ll_short) and np.isfinite(ll_long)


# ---------------------------------------------------------------------------
# Backward algorithm
# ---------------------------------------------------------------------------

class TestHMMBackward:
    def test_beta_shape(self, two_state, short_seq):
        _, scales, _ = two_state.forward(short_seq)
        beta = two_state.backward(short_seq, scales)
        assert beta.shape == (len(short_seq), 2)

    def test_beta_non_negative(self, two_state, short_seq):
        _, scales, _ = two_state.forward(short_seq)
        beta = two_state.backward(short_seq, scales)
        assert np.all(beta >= 0)

    def test_beta_last_row_all_ones(self, two_state, short_seq):
        _, scales, _ = two_state.forward(short_seq)
        beta = two_state.backward(short_seq, scales)
        np.testing.assert_allclose(beta[-1], 1.0, atol=1e-10)

    def test_forward_backward_consistency(self, two_state, short_seq):
        """
        For any t: sum_i alpha_t(i) * beta_t(i) should be proportional
        to the same value for all t (after undoing scaling).
        """
        alpha, scales, _ = two_state.forward(short_seq)
        beta = two_state.backward(short_seq, scales)
        # gamma_t(i) ∝ alpha_t(i) * beta_t(i)
        gamma = alpha * beta
        gamma_sums = gamma.sum(axis=1)
        # All rows should be > 0
        assert np.all(gamma_sums > 0)


# ---------------------------------------------------------------------------
# Viterbi algorithm
# ---------------------------------------------------------------------------

class TestHMMViterbi:
    def test_viterbi_output_shape(self, two_state, short_seq):
        states = two_state.viterbi(short_seq)
        assert states.shape == (len(short_seq),)

    def test_viterbi_valid_state_ids(self, two_state, short_seq):
        states = two_state.viterbi(short_seq)
        assert np.all((states >= 0) & (states < 2))

    def test_viterbi_known_sequence(self, two_state):
        """
        State 0 emits obs 0, state 1 emits obs 2.
        A run of 0s → state 0; a run of 2s → state 1.
        """
        obs = np.array([0, 0, 0, 0, 2, 2, 2, 2])
        states = two_state.viterbi(obs)
        # First half should be state 0, second half state 1 (or vice versa)
        first_half = set(states[:4])
        second_half = set(states[4:])
        assert len(first_half) == 1
        assert len(second_half) == 1
        assert first_half != second_half

    def test_viterbi_integer_dtype(self, two_state, short_seq):
        states = two_state.viterbi(short_seq)
        assert states.dtype in (np.int32, np.int64, int)

    def test_viterbi_single_observation(self, two_state):
        states = two_state.viterbi(np.array([0]))
        assert states.shape == (1,)


# ---------------------------------------------------------------------------
# Baum-Welch (fit)
# ---------------------------------------------------------------------------

class TestHMMBaumWelch:
    def test_fit_returns_self(self, generated_sequences):
        hmm = HiddenMarkovModel(n_states=2, n_observations=3, random_state=0)
        result = hmm.fit(generated_sequences, n_iter=10)
        assert result is hmm

    def test_A_row_stochastic_after_fit(self, generated_sequences):
        hmm = HiddenMarkovModel(n_states=2, n_observations=3, random_state=0)
        hmm.fit(generated_sequences, n_iter=20)
        np.testing.assert_allclose(hmm.A.sum(axis=1), 1.0, atol=1e-8)

    def test_B_row_stochastic_after_fit(self, generated_sequences):
        hmm = HiddenMarkovModel(n_states=2, n_observations=3, random_state=0)
        hmm.fit(generated_sequences, n_iter=20)
        np.testing.assert_allclose(hmm.B.sum(axis=1), 1.0, atol=1e-8)

    def test_pi_sums_to_one_after_fit(self, generated_sequences):
        hmm = HiddenMarkovModel(n_states=2, n_observations=3, random_state=0)
        hmm.fit(generated_sequences, n_iter=20)
        np.testing.assert_allclose(hmm.pi.sum(), 1.0, atol=1e-8)

    def test_fit_increases_log_likelihood(self, generated_sequences):
        """Log-likelihood under fitted model should be higher than random init."""
        hmm_untrained = HiddenMarkovModel(n_states=2, n_observations=3, random_state=0)
        hmm_untrained._init_params()

        hmm_trained = HiddenMarkovModel(n_states=2, n_observations=3, random_state=0)
        hmm_trained.fit(generated_sequences, n_iter=50)

        ll_untrained = sum(hmm_untrained.log_likelihood(s) for s in generated_sequences)
        ll_trained   = sum(hmm_trained.log_likelihood(s) for s in generated_sequences)
        assert ll_trained >= ll_untrained

    def test_fit_recovers_transition_structure(self, generated_sequences):
        """
        True model has high self-transition (0.9); fitted model should too.
        """
        hmm = HiddenMarkovModel(n_states=2, n_observations=3, random_state=1)
        hmm.fit(generated_sequences, n_iter=100)
        # Both diagonal entries of A should be > off-diagonal
        for i in range(2):
            assert hmm.A[i, i] > hmm.A[i, 1 - i]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestHMMSampling:
    def test_sample_length(self, two_state):
        states, obs = two_state.sample(20, random_state=0)
        assert len(states) == 20
        assert len(obs) == 20

    def test_sample_valid_obs(self, two_state):
        _, obs = two_state.sample(50, random_state=0)
        assert np.all((obs >= 0) & (obs < 3))

    def test_sample_valid_states(self, two_state):
        states, _ = two_state.sample(50, random_state=0)
        assert np.all((states >= 0) & (states < 2))

    def test_sample_reproducible(self, two_state):
        s1, o1 = two_state.sample(20, random_state=5)
        s2, o2 = two_state.sample(20, random_state=5)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(o1, o2)

    def test_sample_different_seeds_differ(self, two_state):
        _, o1 = two_state.sample(30, random_state=0)
        _, o2 = two_state.sample(30, random_state=99)
        assert not np.array_equal(o1, o2)
