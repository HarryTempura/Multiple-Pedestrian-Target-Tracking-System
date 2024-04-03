import numpy as np
import scipy.linalg

from main import opt

# 自由度为N的卡方分布的0.95四位小数表（包含N=1，...，9的值）。取自MATLAB/Octave's chi2inv函数并作为Mahalanobis门控门限。
chi2inv95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919}


class KalmanFilter(object):
    def __init__(self):
        ndim, dt = 4, 1
        # 创建卡尔曼滤波器模型矩阵
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """
        从无关联的检测中创建轨迹
        :param measurement:
        :return:
        """
        mean_pos = measurement
        mean_val = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_val]
        std = [
            2 * self._std_weight_position * measurement[3], 2 * self._std_weight_position * measurement[3], 1e-2,
            2 * self._std_weight_position * measurement[3], 10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3], 1e-5, 10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))

        return mean, covariance

    def predict(self, mean, covariance):
        """
        运行卡尔曼滤波预测步骤
        :param mean:
        :param covariance:
        :return:
        """
        std_pos = [
            self._std_weight_position * mean[3], self._std_weight_position * mean[3], 1e-2,
            self._std_weight_position * mean[3]
        ]
        std_val = [
            self._std_weight_velocity * mean[3], self._std_weight_velocity * mean[3], 1e-5,
            self._std_weight_velocity * mean[3]
        ]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_val]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance, confidence=.0):
        """
        将状态分布投射到测量空间
        :param mean:
        :param covariance:
        :param confidence:
        :return:
        """
        std = [
            self._std_weight_position * mean[3], self._std_weight_position * mean[3], 1e-2,
            self._std_weight_position * mean[3]
        ]

        if opt.NSA:
            std = [(1 - confidence) * x for x in std]

        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))

        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement, confidence=.0):
        """
        卡尔曼滤波更新
        :param mean:
        :param covariance:
        :param measurement:
        :param confidence:
        :return:
        """
        projected_mean, projected_cov = self.project(mean, covariance, confidence)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_factor(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """
        计算状态分布和测量之间的门控距离
        :param mean:
        :param covariance:
        :param measurements:
        :param only_position:
        :return:
        """
        mean, covariance = self.project(mean, covariance)

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
