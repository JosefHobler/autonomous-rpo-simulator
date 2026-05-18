from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from . import dynamics


@dataclass
class GlideslopePulse:
    t: float
    dv: np.ndarray
    r_target: np.ndarray


@dataclass
class GlideslopePlan:
    pulses: List[GlideslopePulse]
    rho0: float
    rho_f: float
    cone_half_angle_rad: float

    @property
    def total_dv(self):
        return float(sum(np.linalg.norm(p.dv) for p in self.pulses))

    @property
    def duration(self):
        return self.pulses[-1].t if self.pulses else 0.0


def _waypoints(r0, rho_f, n_pulses):
    rho0 = float(np.linalg.norm(r0))
    if rho0 <= rho_f:
        raise ValueError("Initial range must exceed final range.")
    los = r0 / rho0
   
    ratios = np.linspace(0.0, 1.0, n_pulses + 1) ** 1.4
    rhos = rho0 + (rho_f - rho0) * ratios
    return [rho * los for rho in rhos]


def plan_glideslope(s0, n, n_pulses=6, closing_speed=0.5, rho_f=5.0,
                    cone_half_angle_deg=10.0):
    if closing_speed <= 0:
        raise ValueError("closing_speed must be positive")

    r0 = s0[:3].copy()
    rho0 = float(np.linalg.norm(r0))
    wpts = _waypoints(r0, rho_f, n_pulses)
    leg_dt = [float(np.linalg.norm(wpts[i] - wpts[i + 1])) / closing_speed
              for i in range(n_pulses)]

    pulses: List[GlideslopePulse] = []
    s = s0.copy()
    t = 0.0
    cone = np.deg2rad(cone_half_angle_deg)
    los0 = r0 / rho0

    for i in range(n_pulses):
        dt = leg_dt[i]
        Phi = dynamics.cw_stm(n, dt)
        Phi_rr, Phi_rv, _, _ = dynamics.split_stm(Phi)
        r_now, v_now = s[:3], s[3:]
        r_next = wpts[i + 1]

        try:
            v_req = np.linalg.solve(Phi_rv, r_next - Phi_rr @ r_now)
        except np.linalg.LinAlgError:
            dt *= 1.01
            Phi = dynamics.cw_stm(n, dt)
            Phi_rr, Phi_rv, _, _ = dynamics.split_stm(Phi)
            v_req = np.linalg.solve(Phi_rv, r_next - Phi_rr @ r_now)
            leg_dt[i] = dt

        dv = v_req - v_now

        mid = dynamics.cw_stm(n, dt / 2) @ np.concatenate([r_now, v_req])
        r_mid = mid[:3]
        _ = float(np.dot(r_mid, los0) / max(np.linalg.norm(r_mid), 1e-9))

        pulses.append(GlideslopePulse(t=t, dv=dv, r_target=r_next))
        s = dynamics.apply_impulse(s, dv)
        s = dynamics.propagate(s, n, dt)
        t += dt

    pulses.append(GlideslopePulse(t=t, dv=-s[3:].copy(), r_target=s[:3].copy()))

    return GlideslopePlan(pulses=pulses, rho0=rho0, rho_f=rho_f,
                          cone_half_angle_rad=cone)
