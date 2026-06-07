"""
Tests for mlscratch.bayesian.kalman_filter.KalmanFilter

Uses analytically verifiable 1-D and 2-D state-space models.
"""

import numpy as np
import pytest
from mlscratch.bayesian.kalman_filter import KalmanFilter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def constant_velocity_model():
    """
    2-D state [position, velocity]; 1-D observation (position only).

    x_{t+1} = F x_t + q,   F = [[1, dt], [0, 1]],  Q = diag(0.01, 0.01)
    z_t     = H x_t + r,   H = [[1, 0]],             R = [[0.5]]
    """
    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.array([[0.5]])
    x0 = np.array([0.0, 1.0])    # pos=0, vel=1
    P0 = np.eye(2)
    return KalmanFilter(F, H, Q, R, x0=x0, P0=P0)


@pytest.fixture
def identity_1d():
    """Simplest possible 1-D constant model."""
    F = np.array([[1.0]])
    H = np.array([[1.0]])
    Q = np.array([[1e-4]])
    R = np.array([[1.0]])
    return KalmanFilter(F, H, Q, R)


@pytest.fixture
def constant_position_observations():
    """50 noisy observations of a stationary object at position 5."""
    rng = np.random.default_rng(0)
    return 5.0 + rng.normal(0, 1, (50, 1))


@pytest.fixture
def linear_motion_observations():
    """100 noisy position observations of object moving at velocity 1."""
    rng = np.random.default_rng(42)
    t = np.arange(100)
    return (t.astype(float) + rng.normal(0, 0.5, 100)).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

class TestKFBasic:
    def test_filter_returns_self(self, constant_velocity_model,
                                  constant_position_observations):
        kf = constant_velocity_model
        assert kf.filter(constant_position_observations) is kf

    def test_x_filt_shape(self, constant_velocity_model,
                           constant_position_observations):
        kf = constant_velocity_model.filter(constant_position_observations)
        assert kf.x_filt_.shape == (50, 2)

    def test_p_filt_shape(self, constant_velocity_model,
                           constant_position_observations):
        kf = constant_velocity_model.filter(constant_position_observations)
        assert kf.P_filt_.shape == (50, 2, 2)

    def test_x_pred_shape(self, constant_velocity_model,
                           constant_position_observations):
        kf = constant_velocity_model.filter(constant_position_observations)
        assert kf.x_pred_.shape == (50, 2)

    def test_p_pred_shape(self, constant_velocity_model,
                           constant_position_observations):
        kf = constant_velocity_model.filter(constant_position_observations)
        assert kf.P_pred_.shape == (50, 2, 2)

    def test_log_likelihood_finite(self, constant_velocity_model,
                                    constant_position_observations):
        kf = constant_velocity_model.filter(constant_position_observations)
        assert np.isfinite(kf.log_likelihood_)

    def test_log_likelihood_negative(self, constant_velocity_model,
                                      constant_position_observations):
        kf = constant_velocity_model.filter(constant_position_observations)
        assert kf.log_likelihood_ < 0

    def test_p_filt_symmetric(self, constant_velocity_model,
                               constant_position_observations):
        kf = constant_velocity_model.filter(constant_position_observations)
        for t in range(len(constant_position_observations)):
            P = kf.P_filt_[t]
            np.testing.assert_allclose(P, P.T, atol=1e-10)

    def test_p_filt_positive_definite(self, constant_velocity_model,
                                       constant_position_observations):
        kf = constant_velocity_model.filter(constant_position_observations)
        for t in range(len(constant_position_observations)):
            eigvals = np.linalg.eigvalsh(kf.P_filt_[t])
            assert np.all(eigvals > 0)

    def test_predict_obs(self, constant_velocity_model,
                          constant_position_observations):
        kf = constant_velocity_model.filter(constant_position_observations)
        obs_pred = kf.predict_obs(kf.x_filt_[0])
        assert obs_pred.shape == (1,)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestKFCorrectness:
    def test_stationary_object_estimate_converges(self, identity_1d,
                                                   constant_position_observations):
        """1-D KF should converge to true position ≈ 5.0."""
        kf = identity_1d.filter(constant_position_observations)
        final_estimate = kf.x_filt_[-1, 0]
        assert abs(final_estimate - 5.0) < 0.5

    def test_uncertainty_decreases_over_time(self, identity_1d,
                                              constant_position_observations):
        """Posterior variance P should decrease as data accumulates."""
        kf = identity_1d.filter(constant_position_observations)
        p_early = kf.P_filt_[0, 0, 0]
        p_late  = kf.P_filt_[-1, 0, 0]
        assert p_late < p_early

    def test_velocity_model_tracks_linear_motion(
        self, constant_velocity_model, linear_motion_observations
    ):
        """KF should recover velocity ≈ 1 from noisy position measurements."""
        kf = constant_velocity_model.filter(linear_motion_observations)
        # Final velocity estimate (state index 1)
        vel_estimate = kf.x_filt_[-1, 1]
        assert abs(vel_estimate - 1.0) < 0.3

    def test_filtered_positions_close_to_true(
        self, constant_velocity_model, linear_motion_observations
    ):
        """Filtered positions should track the true linear trajectory."""
        kf = constant_velocity_model.filter(linear_motion_observations)
        true_positions = np.arange(100).astype(float)
        filtered_positions = kf.x_filt_[:, 0]
        rmse = np.sqrt(np.mean((filtered_positions - true_positions) ** 2))
        assert rmse < 5.0   # generous tolerance

    def test_log_likelihood_improves_with_better_noise_model(
        self, linear_motion_observations
    ):
        """Better-tuned R should give higher log-likelihood."""
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.01

        kf_good = KalmanFilter(F, H, Q, np.array([[0.25]])).filter(
            linear_motion_observations
        )
        kf_bad  = KalmanFilter(F, H, Q, np.array([[100.0]])).filter(
            linear_motion_observations
        )
        assert kf_good.log_likelihood_ > kf_bad.log_likelihood_


# ---------------------------------------------------------------------------
# RTS Smoother
# ---------------------------------------------------------------------------

class TestRTSSmoother:
    def test_smooth_requires_filter_first(self, constant_velocity_model):
        with pytest.raises(RuntimeError):
            constant_velocity_model.smooth()

    def test_smoother_output_shapes(
        self, constant_velocity_model, constant_position_observations
    ):
        kf = constant_velocity_model.filter(constant_position_observations)
        x_s, P_s = kf.smooth()
        assert x_s.shape == (50, 2)
        assert P_s.shape == (50, 2, 2)

    def test_smoother_uncertainty_leq_filter(
        self, constant_velocity_model, constant_position_observations
    ):
        """Smoothed uncertainty should be ≤ filtered uncertainty at each step."""
        kf = constant_velocity_model.filter(constant_position_observations)
        x_s, P_s = kf.smooth()
        for t in range(len(constant_position_observations)):
            # Trace of covariance (total uncertainty)
            assert np.trace(P_s[t]) <= np.trace(kf.P_filt_[t]) + 1e-9

    def test_smoother_last_step_equals_filter(
        self, constant_velocity_model, constant_position_observations
    ):
        """At the last time step, smoother = filter (no future data)."""
        kf = constant_velocity_model.filter(constant_position_observations)
        x_s, P_s = kf.smooth()
        T = len(constant_position_observations)
        np.testing.assert_allclose(x_s[T - 1], kf.x_filt_[T - 1], atol=1e-10)
        np.testing.assert_allclose(P_s[T - 1], kf.P_filt_[T - 1], atol=1e-10)

    def test_smoother_p_symmetric(
        self, constant_velocity_model, constant_position_observations
    ):
        kf = constant_velocity_model.filter(constant_position_observations)
        _, P_s = kf.smooth()
        for t in range(len(constant_position_observations)):
            np.testing.assert_allclose(P_s[t], P_s[t].T, atol=1e-10)

    def test_smoother_improves_rmse(
        self, constant_velocity_model, linear_motion_observations
    ):
        """Smoothed estimate should be at least as good as filtered."""
        kf = constant_velocity_model.filter(linear_motion_observations)
        x_s, _ = kf.smooth()
        true_pos = np.arange(100).astype(float)
        rmse_filt   = np.sqrt(np.mean((kf.x_filt_[:, 0] - true_pos) ** 2))
        rmse_smooth = np.sqrt(np.mean((x_s[:, 0] - true_pos) ** 2))
        assert rmse_smooth <= rmse_filt + 0.5   # smoother should match or beat


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestKFEdgeCases:
    def test_single_observation(self):
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[1.0]])
        kf = KalmanFilter(F, H, Q, R).filter(np.array([[3.0]]))
        assert kf.x_filt_.shape == (1, 1)

    def test_high_dimensional_state(self):
        """4-D state, 2-D observation."""
        rng = np.random.default_rng(0)
        F = np.eye(4)
        H = rng.standard_normal((2, 4))
        Q = np.eye(4) * 0.1
        R = np.eye(2) * 0.5
        Z = rng.standard_normal((20, 2))
        kf = KalmanFilter(F, H, Q, R).filter(Z)
        assert kf.x_filt_.shape == (20, 4)
