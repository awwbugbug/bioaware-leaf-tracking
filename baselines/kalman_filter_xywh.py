"""
ByteTrack / BoT-SORT style Kalman filter.
State space: [cx, cy, w, h, vx, vy, vw, vh]

Faithful to:
  Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
  Every Detection Box", ECCV 2022.
  https://github.com/ifzhang/ByteTrack

  Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian
  Tracking", arXiv 2206.14651.
  https://github.com/NirAharon/BoT-SORT
"""

import numpy as np
import scipy.linalg


class KalmanFilterXYWH:
    """
    Kalman filter with state (cx, cy, w, h, vx, vy, vw, vh).
    Used by ByteTrack and BoT-SORT.
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """measurement: (cx, cy, w, h)"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot([self._motion_mat, covariance, self._motion_mat.T])
            + motion_cov
        )
        return mean, covariance

    def multi_predict(self, multi_mean, multi_covariance):
        if len(multi_mean) == 0:
            return multi_mean, multi_covariance
        std_pos = [
            self._std_weight_position * multi_mean[:, 2],
            self._std_weight_position * multi_mean[:, 3],
            self._std_weight_position * multi_mean[:, 2],
            self._std_weight_position * multi_mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * multi_mean[:, 2],
            self._std_weight_velocity * multi_mean[:, 3],
            self._std_weight_velocity * multi_mean[:, 2],
            self._std_weight_velocity * multi_mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = [np.diag(sq) for sq in sqr]
        multi_mean = np.dot(multi_mean, self._motion_mat.T)
        multi_covariance = np.asarray(
            [
                np.linalg.multi_dot([self._motion_mat, cov, self._motion_mat.T]) + mc
                for cov, mc in zip(multi_covariance, motion_cov)
            ]
        )
        return multi_mean, multi_covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            [self._update_mat, covariance, self._update_mat.T]
        )
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_cov = covariance - np.linalg.multi_dot(
            [kalman_gain, projected_cov, kalman_gain.T]
        )
        return new_mean, new_cov
