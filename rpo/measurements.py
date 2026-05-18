from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorParams:
    sigma_range: float = 0.10
    sigma_bearing: float = 1e-3
    rng_seed: int | None = 0

    @property
    def R(self):
        return np.diag([self.sigma_range**2,
                        self.sigma_bearing**2,
                        self.sigma_bearing**2])


def measure(state):
    r = state[:3]
    rho = float(np.linalg.norm(r))
    if rho < 1e-9:
        return np.zeros(3)
    az = np.arctan2(-r[1], -r[0])
    el = np.arcsin(np.clip(-r[2] / rho, -1.0, 1.0))
    return np.array([rho, az, el])


def measurement_jacobian(state):
    rx, ry, rz = state[:3]
    rho2 = rx*rx + ry*ry + rz*rz
    rho = float(np.sqrt(rho2))
    rxy2 = rx*rx + ry*ry
    rxy = float(np.sqrt(rxy2))

    H = np.zeros((3, 6))
    if rho < 1e-9:
        return H

    # range row
    H[0, 0] = rx / rho
    H[0, 1] = ry / rho
    H[0, 2] = rz / rho

    if rxy2 > 1e-18:
        H[1, 0] = -ry / rxy2
        H[1, 1] =  rx / rxy2

    if rxy > 1e-9:
        H[2, 0] = rz * rx / (rho2 * rxy)
        H[2, 1] = rz * ry / (rho2 * rxy)
        H[2, 2] = -rxy / rho2
    return H


class Sensor:
    def __init__(self, params=None):
        self.params = params or SensorParams()
        self._rng = np.random.default_rng(self.params.rng_seed)

    def sample(self, state):
        z = measure(state)
        noise = self._rng.normal(scale=[self.params.sigma_range,
                                        self.params.sigma_bearing,
                                        self.params.sigma_bearing])
        return z + noise

    @staticmethod
    def predict(state):
        return measure(state)

    @staticmethod
    def jacobian(state):
        return measurement_jacobian(state)
