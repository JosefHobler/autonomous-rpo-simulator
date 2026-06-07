from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from . import ccsds, dynamics, ekf, guidance, measurements


@dataclass
class SimConfig:
    altitude_m: float = 400e3
    duration_s: float | None = None
    dt_sim: float = 0.5
    dt_meas: float = 1.0
    n_pulses: int = 6
    closing_speed: float = 0.4
    rho_f: float = 5.0
    cone_half_angle_deg: float = 12.0
    initial_relative_state: np.ndarray = field(
        default_factory=lambda: np.array([-200.0, -2000.0, 50.0, 0.05, -0.1, 0.0])
    )
    sensor: measurements.SensorParams = field(default_factory=measurements.SensorParams)
    ekf_params: ekf.EKFParams = field(default_factory=ekf.EKFParams)
    rng_seed: int = 0
    # Keep the full 6x6 covariance at every step (needed for NEES / Monte Carlo
    # filter-consistency analysis). Off by default; only the diagonal is stored.
    log_full_cov: bool = False


@dataclass
class SimResult:
    t: np.ndarray
    truth: np.ndarray
    estimate: np.ndarray
    cov_diag: np.ndarray
    measurements: np.ndarray
    guidance_pulses: list
    delta_v_total: float
    config: SimConfig
    telemetry: ccsds.TelemetryStream
    cov_full: np.ndarray | None = None   # (n_steps, 6, 6) when log_full_cov


def run(config: SimConfig | None = None) -> SimResult:
    cfg = config or SimConfig()
    # One seed fully determines a trial. Spawn independent child streams so the
    # nav-init draw and the sensor noise never share state across trials.
    nav_seed, sensor_seed = np.random.SeedSequence(cfg.rng_seed).spawn(2)
    rng = np.random.default_rng(nav_seed)
    n = dynamics.mean_motion(cfg.altitude_m)

    s_truth = cfg.initial_relative_state.astype(float).copy()
    plan = guidance.plan_glideslope(
        s_truth, n,
        n_pulses=cfg.n_pulses,
        closing_speed=cfg.closing_speed,
        rho_f=cfg.rho_f,
        cone_half_angle_deg=cfg.cone_half_angle_deg,
    )
    duration = cfg.duration_s if cfg.duration_s is not None else plan.duration + 30.0

    sensor = measurements.Sensor(measurements.SensorParams(
        sigma_range=cfg.sensor.sigma_range,
        sigma_bearing=cfg.sensor.sigma_bearing,
        rng_seed=sensor_seed,
    ))

    # nav initialized off the truth with a deliberate offset
    sp, sv = cfg.ekf_params.initial_pos_sigma, cfg.ekf_params.initial_vel_sigma
    s_guess = s_truth + rng.normal(scale=[sp, sp, sp, sv, sv, sv])
    filt = ekf.build_default_filter(n, s_guess, cfg.ekf_params)
    R = cfg.sensor.R

    tlm = ccsds.TelemetryStream()
    tlm.emit(ccsds.APID.EVENT, ccsds.encode_event("RPO_SIM_START"), 0.0)

    n_steps = int(np.ceil(duration / cfg.dt_sim)) + 1
    t_grid = np.arange(n_steps) * cfg.dt_sim
    truth_log = np.zeros((n_steps, 6))
    est_log   = np.zeros((n_steps, 6))
    cov_log   = np.zeros((n_steps, 6))
    cov_full_log = np.zeros((n_steps, 6, 6)) if cfg.log_full_cov else None
    meas_log: List[np.ndarray] = []

    pulses_remaining = list(plan.pulses)
    dv_total = 0.0
    next_meas_t = 0.0

    for k, t in enumerate(t_grid):
        # fire any guidance pulses that came due this tick
        while pulses_remaining and pulses_remaining[0].t <= t + 1e-9:
            p = pulses_remaining.pop(0)
            s_truth = dynamics.apply_impulse(s_truth, p.dv)
            filt.predict(0.0, control_dv=p.dv)
            dv_total += float(np.linalg.norm(p.dv))
            tlm.emit(ccsds.APID.GUIDANCE_CMD, ccsds.encode_guidance(p.dv), t)

        if t + 1e-9 >= next_meas_t:
            z = sensor.sample(s_truth)
            meas_log.append(np.array([t, *z]))
            tlm.emit(ccsds.APID.SENSOR_MEAS, ccsds.encode_meas(z), t)
            filt.update(z, sensor.predict, sensor.jacobian, R)
            next_meas_t += cfg.dt_meas

        truth_log[k] = s_truth
        est_log[k]   = filt.x
        cov_log[k]   = np.diag(filt.P)
        if cov_full_log is not None:
            cov_full_log[k] = filt.P

        tlm.emit(ccsds.APID.CHASER_TRUTH,
                 ccsds.encode_state(s_truth, dv_total), t)
        tlm.emit(ccsds.APID.CHASER_NAV,
                 ccsds.encode_nav(filt.x,
                                  float(np.trace(filt.P[:3, :3])),
                                  float(np.trace(filt.P[3:, 3:]))),
                 t)

        if k < n_steps - 1:
            s_truth = dynamics.propagate(s_truth, n, cfg.dt_sim)
            filt.predict(cfg.dt_sim)

    tlm.emit(ccsds.APID.EVENT,
             ccsds.encode_event(f"RPO_SIM_END dv_total={dv_total:.3f}m/s"),
             float(t_grid[-1]))

    return SimResult(
        t=t_grid,
        truth=truth_log,
        estimate=est_log,
        cov_diag=cov_log,
        measurements=np.array(meas_log) if meas_log else np.zeros((0, 4)),
        guidance_pulses=plan.pulses,
        delta_v_total=dv_total,
        config=cfg,
        telemetry=tlm,
        cov_full=cov_full_log,
    )
