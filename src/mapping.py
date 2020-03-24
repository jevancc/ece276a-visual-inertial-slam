import itertools
import numpy as np
import scipy.linalg
from .utils import *
from .robot import *


class EKFLandmarkMapping:

    def __init__(self,
                 n_landmarks,
                 robot_cam_T_imu,
                 robot_cam_intrinsic_calib,
                 robot_cam_baseline,
                 process_noise_covariance=None,
                 observation_noise_covariance=None,
                 prior_pose_covariance=None,
                 prior_landmark_covariance=None):

        if prior_landmark_covariance is None:
            prior_landmark_covariance = 5e-3 * np.eye(3)
        if prior_pose_covariance is None:
            prior_pose_covariance = 1e-3 * np.eye(6)
        if observation_noise_covariance is None:
            observation_noise_covariance = 100 * np.eye(4)
        if process_noise_covariance is None:
            process_noise_covariance = 1e-3 * np.eye(6)

        self.xU = np.eye(4)
        self.W = process_noise_covariance

        self.n_landmarks = n_landmarks

        self._n_initialized = 0
        self._initialized_maxid = 0
        self._initialized_mask = np.zeros((n_landmarks), dtype=bool)

        self.xm = np.zeros((n_landmarks, 3))

        self.P = np.kron(np.eye(n_landmarks), prior_landmark_covariance)
        self.V = observation_noise_covariance

        self.oTi = robot_cam_T_imu

        K = robot_cam_intrinsic_calib
        self.b = robot_cam_baseline
        self.M = np.block([[K[:2, :], np.array([[0, 0]]).T], [K[:2, :], np.array([[-K[0, 0] * self.b, 0]]).T]])

    @property
    def initialized_maxid(self):
        return self._initialized_maxid

    @property
    def n_initialized(self):
        return self._n_initialized

    @property
    def xUp(self):
        return self.xU[:3, 3].reshape(-1, 1)

    @property
    def oTw(self):
        return self.oTi @ self.xU

    def predict(self, u, tau):
        F = scipy.linalg.expm(-tau * wedge(u))
        self.xU = F @ self.xU

    def _make_zmap(self, z):
        assert z.ndim == 2 and z.shape[0] == 4
        return np.array(np.where(z.sum(axis=0) > -4), dtype=np.int32).reshape(-1)

    def _init_landmark(self, z, zmap):
        mask = np.invert(self._initialized_mask[zmap])
        zmap = zmap[mask]
        if zmap.size > 0:
            wTo = np.linalg.inv(self.oTw)
            self._initialized_mask[zmap] = True
            z = z[:, zmap]

            M = self.M
            b = self.b
            wcoord = np.ones((4, zmap.size))
            wcoord[0, :] = (z[0, :] - M[0, 2]) * b / (z[0, :] - z[2, :])
            wcoord[1, :] = (z[1, :] - M[1, 2]) * (-M[2, 3]) / (M[1, 1] * (z[0, :] - z[2, :]))
            wcoord[2, :] = -M[2, 3] / (z[0, :] - z[2, :])
            wcoord = wTo @ wcoord
            self.xm[zmap, :] = wcoord[:3, :].T

            self._n_initialized = np.sum(self._initialized_mask)
            self._initialized_maxid = max(zmap.max() + 1, self._initialized_maxid)

    def _make_H(self, z, zmap):
        n_observations = zmap.size
        n_updates = self._initialized_maxid

        P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)

        xm = np.hstack([self.xm[zmap, :], np.ones((n_observations, 1))])
        H = np.zeros((n_observations * 4, n_updates * 3))

        for i in range(n_observations):
            obi = zmap[i]

            H[i * 4:(i + 1) * 4,
              obi * 3:(obi + 1) * 3] = self.M @ dpidq(self.oTw @ xm[i, :].reshape(-1, 1)) @ self.oTw @ P.T

        return H

    def _make_xm_P(self, z, zmap):
        n_observations = zmap.size
        n_updates = self._initialized_maxid

        xm = self.xm[:n_updates, :]
        P = self.P[:n_updates * 3, :n_updates * 3]

        return xm, P

    def _make_z(self, z, zmap):
        return z[:, zmap].reshape(-1, 1, order='F')

    def _make_predicted_z(self, z, zmap):
        n_observations = zmap.size

        xm = np.hstack([self.xm[zmap, :], np.ones((n_observations, 1))])
        zp = self.M @ pi(self.oTw @ xm.T)
        return zp.reshape(-1, 1, order='F')

    def _update_value_xm_P(self, xm, P, zmap):
        n_observations = zmap.size
        n_updates = self._initialized_maxid

        self.xm[:n_updates, :] = xm
        self.P[:n_updates * 3, :n_updates * 3] = P

    def update(self, z):
        zmap = self._make_zmap(z)
        if zmap.size > 0:
            n_observations = zmap.size
            self._init_landmark(z, zmap)

            H = self._make_H(z, zmap)
            xm, P = self._make_xm_P(z, zmap)
            zp = self._make_predicted_z(z, zmap)
            z = self._make_z(z, zmap)

            V = np.kron(np.eye(n_observations), self.V)
            PHT = P @ H.T
            K = np.linalg.solve((H @ PHT + V).T, PHT.T).T

            xm += (K @ (z - zp)).reshape(-1, 3)
            P = (np.eye(K.shape[0]) - K @ H) @ P

            self._update_value_xm_P(xm, P, zmap)
