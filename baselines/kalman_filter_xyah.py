"""
DeepSORT-style Kalman filter.
State space: [cx, cy, a, h, vx, vy, va, vh]
where (cx, cy) = centre, a = aspect ratio, h = height.

Faithful to:
  Wojke et al., "Simple Online and Realtime Tracking with a Deep
  Association Metric", ICIP 2017.
  https://github.com/nwojke/deep_sort
"""

import numpy as np
import scipy.linalg

# Table for the 0.95 quantile of the chi-square distribution with N
# degrees of freedom (used for Mahalanobis gating).
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.0705,
    6: 12.5916,
    7: 14.0671,
    8: 15.5073,
    9: 16.9190,
}


class KalmanFilterXYAH:
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    Measurement space: (cx, cy, a, h)
    State space:       (cx, cy, a, h, vx, vy, va, vh)
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # State transition matrix (constant velocity model)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Measurement matrix (observe position only)
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty weights (std = weight * h)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray (cx, cy, a, h)

        Returns
        -------
        mean     : ndarray shape (8,)
        covariance: ndarray shape (8, 8)
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot([self._motion_mat, covariance, self._motion_mat.T])
            + motion_cov
        )
        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            [self._update_mat, covariance, self._update_mat.T]
        )
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Vectorised predict for multiple tracks."""
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T  # (N, 8)
        motion_cov = [np.diag(sq) for sq in sqr]

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.asarray(
            [
                np.linalg.multi_dot([self._motion_mat, cov, self._motion_mat.T]) + mc
                for cov, mc in zip(covariance, motion_cov)
            ]
        )
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            [kalman_gain, projected_cov, kalman_gain.T]
        )
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Compute Mahalanobis distance for gating."""
        projected_mean, projected_cov = self.project(mean, covariance)
        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(projected_cov)
        d = measurements - projected_mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor,
            d.T,
            lower=True,
            check_finite=False,
            overwrite_b=True,
        )
        return np.sum(z * z, axis=0)
