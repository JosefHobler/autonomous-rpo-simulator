from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import dynamics


@dataclass
class EKFParams:
    process_accel_psd: float = 1e-6
    initial_pos_sigma: float = 50.0
    initial_vel_sigma: float = 0.5


def cw_process_noise(n, dt, accel_psd):
    B = dynamics.cw_B()
    BQB = accel_psd * (B @ B.T)

    steps = 8
    taus = np.linspace(0.0, dt, steps + 1)
    Q = np.zeros((6, 6))
    for k, tau in enumerate(taus):
        Phi = dynamics.cw_stm(n, tau)
        w = 0.5 if k == 0 or k == steps else 1.0
        Q += w * (Phi @ BQB @ Phi.T)
    Q *= dt / steps
    return 0.5 * (Q + Q.T)


def _wrap_innovation(y):
    if y.shape[0] >= 2:
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
    if y.shape[0] >= 3:
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
    return y


class PythonEKF:
    def __init__(self, n, x0, P0, params: Optional[EKFParams] = None):
        self.n = float(n)
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.params = params or EKFParams()
        self._I = np.eye(6)

    def predict(self, dt, control_dv=None):
        if control_dv is not None:
            self.x[3:] += control_dv
        F = dynamics.cw_stm(self.n, dt)
        Q = cw_process_noise(self.n, dt, self.params.process_accel_psd)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z, h_func, H_func, R):
        y = _wrap_innovation(z - h_func(self.x))
        H = H_func(self.x)
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T   # K = P H^T S^{-1}
        self.x = self.x + K @ y

        # Joseph form (keeps P symmetric and PSD even with bad K)
        I_KH = self._I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return y


class CppEKF:
    """Adapter around the pybind11 C++ core. Same API as PythonEKF."""

    def __init__(self, n, x0, P0, params, mod):
        self.n = float(n)
        self.params = params
        self._core = mod.EkfCore(6)
        self._core.set_state(np.asarray(x0, dtype=float),
                             np.asarray(P0, dtype=float))

    @property
    def x(self):
        return np.asarray(self._core.state())

    @property
    def P(self):
        return np.asarray(self._core.covariance())

    def predict(self, dt, control_dv=None):
        if control_dv is not None:
            x = self.x.copy()
            x[3:] += control_dv
            self._core.set_state(x, self.P)
        F = dynamics.cw_stm(self.n, dt)
        Q = cw_process_noise(self.n, dt, self.params.process_accel_psd)
        self._core.predict(F, Q)

    def update(self, z, h_func, H_func, R):
        x = self.x
        y = _wrap_innovation(z - h_func(x))
        H = H_func(x)
        self._core.update_linear(y, H, R)
        return y


def build_default_filter(n, s_guess, params: Optional[EKFParams] = None):
    """Build an EKF; prefer the C++ backend when the extension is built."""
    params = params or EKFParams()
    sp, sv = params.initial_pos_sigma, params.initial_vel_sigma
    P0 = np.diag([sp*sp, sp*sp, sp*sp, sv*sv, sv*sv, sv*sv])

    try:
        from . import _ekf_cpp
        return CppEKF(n, s_guess, P0, params, _ekf_cpp)
    except Exception:
        return PythonEKF(n, s_guess, P0, params)
