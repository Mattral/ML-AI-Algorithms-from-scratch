"""
Kalman Filter and RTS Smoother
================================
A sequential Bayesian filter for linear-Gaussian state-space models.

State-space model
-----------------
    x_t = F x_{t-1} + q_t,   q_t ~ N(0, Q)   (transition)
    z_t = H x_t + r_t,        r_t ~ N(0, R)   (observation)

Kalman Filter (forward pass)
----------------------------
Predict:
    x̂_{t|t-1} = F x̂_{t-1|t-1}
    P_{t|t-1}  = F P_{t-1|t-1} F^T + Q

Update:
    K_t = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}
    x̂_{t|t} = x̂_{t|t-1} + K_t (z_t - H x̂_{t|t-1})
    P_{t|t}  = (I - K_t H) P_{t|t-1}

RTS Smoother (backward pass) — Rauch-Tung-Striebel
---------------------------------------------------
Refines the filtered estimates using all future observations.

Only numpy is used.
"""

import numpy as np


class KalmanFilter:
    """
    Linear Kalman Filter with optional RTS smoother.

    Parameters
    ----------
    F : ndarray (state_dim, state_dim)
        State transition matrix.
    H : ndarray (obs_dim, state_dim)
        Observation matrix.
    Q : ndarray (state_dim, state_dim)
        Process noise covariance.
    R : ndarray (obs_dim, obs_dim)
        Observation noise covariance.
    x0 : ndarray (state_dim,) or None
        Initial state estimate.  Defaults to zeros.
    P0 : ndarray (state_dim, state_dim) or None
        Initial state covariance.  Defaults to identity * 1e6.
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ):
        self.F = np.array(F, dtype=float)
        self.H = np.array(H, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)

        state_dim = self.F.shape[0]
        self.x0 = np.array(x0, dtype=float) if x0 is not None else np.zeros(state_dim)
        self.P0 = np.array(P0, dtype=float) if P0 is not None else np.eye(state_dim) * 1e6

        # Stored after filter()
        self.x_filt_ = None    # (T, state_dim) filtered means
        self.P_filt_ = None    # (T, state_dim, state_dim) filtered covs
        self.x_pred_ = None    # (T, state_dim) predicted means
        self.P_pred_ = None    # (T, state_dim, state_dim) predicted covs
        self.log_likelihood_ = None

    # ------------------------------------------------------------------
    # Kalman Filter
    # ------------------------------------------------------------------

    def filter(self, Z: np.ndarray) -> "KalmanFilter":
        """
        Run the Kalman filter over observation sequence Z.

        Parameters
        ----------
        Z : ndarray of shape (T, obs_dim)

        Returns
        -------
        self
        """
        T = len(Z)
        state_dim = self.F.shape[0]
        obs_dim = self.H.shape[0]
        I = np.eye(state_dim)

        x_filt = np.zeros((T, state_dim))
        P_filt = np.zeros((T, state_dim, state_dim))
        x_pred = np.zeros((T, state_dim))
        P_pred = np.zeros((T, state_dim, state_dim))
        log_lik = 0.0

        x = self.x0.copy()
        P = self.P0.copy()

        for t in range(T):
            # --- Predict ---
            x_p = self.F @ x
            P_p = self.F @ P @ self.F.T + self.Q

            x_pred[t] = x_p
            P_pred[t] = P_p

            # --- Update ---
            S = self.H @ P_p @ self.H.T + self.R    # innovation covariance
            K = P_p @ self.H.T @ np.linalg.inv(S)   # Kalman gain

            innovation = Z[t] - self.H @ x_p
            x = x_p + K @ innovation
            P = (I - K @ self.H) @ P_p

            # Log-likelihood contribution
            sign, log_det = np.linalg.slogdet(S)
            if sign > 0:
                log_lik -= 0.5 * (
                    obs_dim * np.log(2 * np.pi) + log_det
                    + innovation @ np.linalg.inv(S) @ innovation
                )

            x_filt[t] = x
            P_filt[t] = P

        self.x_filt_ = x_filt
        self.P_filt_ = P_filt
        self.x_pred_ = x_pred
        self.P_pred_ = P_pred
        self.log_likelihood_ = log_lik
        return self

    # ------------------------------------------------------------------
    # RTS Smoother
    # ------------------------------------------------------------------

    def smooth(self) -> tuple:
        """
        Rauch-Tung-Striebel smoother.  Must call filter() first.

        Returns
        -------
        x_smooth : ndarray (T, state_dim)
        P_smooth : ndarray (T, state_dim, state_dim)
        """
        if self.x_filt_ is None:
            raise RuntimeError("Call filter() before smooth().")

        T = len(self.x_filt_)
        state_dim = self.F.shape[0]

        x_smooth = self.x_filt_.copy()
        P_smooth = self.P_filt_.copy()

        for t in range(T - 2, -1, -1):
            P_pred_inv = np.linalg.inv(self.P_pred_[t + 1])
            G = self.P_filt_[t] @ self.F.T @ P_pred_inv  # smoother gain
            x_smooth[t] = self.x_filt_[t] + G @ (
                x_smooth[t + 1] - self.x_pred_[t + 1]
            )
            P_smooth[t] = (
                self.P_filt_[t]
                + G @ (P_smooth[t + 1] - self.P_pred_[t + 1]) @ G.T
            )

        return x_smooth, P_smooth

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def predict_obs(self, x_state: np.ndarray) -> np.ndarray:
        """Return expected observation given state estimate."""
        return self.H @ x_state
