from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np

from . import simulator


@dataclass
class Dispersions:
    pos_sigma: np.ndarray = field(
        default_factory=lambda: np.array([20.0, 20.0, 10.0]))
    vel_sigma: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.01]))

    def sample(self, rng) -> np.ndarray:
        scale = np.concatenate([self.pos_sigma, self.vel_sigma])
        return rng.normal(scale=scale)


@dataclass
class MCConfig:
    base: simulator.SimConfig = field(default_factory=simulator.SimConfig)
    dispersions: Dispersions = field(default_factory=Dispersions)
    n_trials: int = 200
    seed0: int = 0
    capture_speed: float = 0.05  
    workers: Optional[int] = None 


@dataclass
class TrialResult:
    seed: int
    delta_v_total: float
    final_range: float
    final_speed: float
    rms_nav_pos_err: float
    rms_nav_vel_err: float
    cone_violation: bool
    captured: bool
    t: np.ndarray
    range_profile: np.ndarray
    nees: np.ndarray


def _nees_series(truth, est, cov_full):
    err = est - truth
    out = np.empty(len(err))
    if cov_full is None:
        var = np.maximum(np.var(err, axis=0), 1e-12)  
        return np.sum(err * err / var, axis=1)
    for k in range(len(err)):
        try:
            out[k] = float(err[k] @ np.linalg.solve(cov_full[k], err[k]))
        except np.linalg.LinAlgError:
            out[k] = np.nan
    return out


def run_trial(base: simulator.SimConfig, seed: int,
              disp: Dispersions, capture_speed: float) -> TrialResult:
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    cfg = replace(
        base,
        rng_seed=seed,
        initial_relative_state=base.initial_relative_state + disp.sample(rng),
        log_full_cov=True,
    )
    res = simulator.run(cfg)

    truth, est = res.truth, res.estimate
    r = truth[:, :3]
    range_profile = np.linalg.norm(r, axis=1)

    k_cap = int(np.argmin(range_profile))
    final_range = float(range_profile[k_cap])
    final_speed = float(np.linalg.norm(truth[k_cap, 3:]))

    los0 = cfg.initial_relative_state[:3]
    los0 = los0 / max(np.linalg.norm(los0), 1e-9)
    cos_ang = (r @ los0) / np.maximum(range_profile, 1e-9)
    cone = np.deg2rad(cfg.cone_half_angle_deg)
    approaching = range_profile > cfg.rho_f
    cone_violation = bool(np.any((np.arccos(np.clip(cos_ang, -1, 1)) > cone)
                                 & approaching))

    pos_err = np.linalg.norm((est - truth)[:, :3], axis=1)
    vel_err = np.linalg.norm((est - truth)[:, 3:], axis=1)

    return TrialResult(
        seed=seed,
        delta_v_total=res.delta_v_total,
        final_range=final_range,
        final_speed=final_speed,
        rms_nav_pos_err=float(np.sqrt(np.mean(pos_err ** 2))),
        rms_nav_vel_err=float(np.sqrt(np.mean(vel_err ** 2))),
        cone_violation=cone_violation,
        captured=bool(final_range <= cfg.rho_f
                      and final_speed <= capture_speed),
        t=res.t,
        range_profile=range_profile,
        nees=_nees_series(truth, est, res.cov_full),
    )


def run_campaign(mc: MCConfig | None = None,
                 progress=None) -> List[TrialResult]:
    mc = mc or MCConfig()
    seeds = np.random.SeedSequence(mc.seed0).generate_state(mc.n_trials)
    args = [(mc.base, int(s), mc.dispersions, mc.capture_speed) for s in seeds]

    results: List[TrialResult] = []
    if mc.workers == 1:
        for i, a in enumerate(args):
            results.append(run_trial(*a))
            if progress:
                progress(i + 1, mc.n_trials)
        return results

    with ProcessPoolExecutor(max_workers=mc.workers) as pool:
        futures = [pool.submit(run_trial, *a) for a in args]
        for i, fut in enumerate(futures):
            results.append(fut.result())
            if progress:
                progress(i + 1, mc.n_trials)
    return results


# --- aggregation -----------------------------------------------------------

@dataclass
class CampaignStats:
    n_trials: int
    capture_rate: float
    cone_violation_rate: float
    dv_mean: float
    dv_p95: float
    dv_p99: float
    final_range_mean: float
    rms_nav_pos_err_mean: float
    anees_mean: float     


def summarize(results: List[TrialResult]) -> CampaignStats:
    dv = np.array([r.delta_v_total for r in results])
    min_len = min(len(r.nees) for r in results)
    nees_stack = np.vstack([r.nees[:min_len] for r in results])
    anees = np.nanmean(nees_stack)

    return CampaignStats(
        n_trials=len(results),
        capture_rate=float(np.mean([r.captured for r in results])),
        cone_violation_rate=float(np.mean([r.cone_violation for r in results])),
        dv_mean=float(np.mean(dv)),
        dv_p95=float(np.percentile(dv, 95)),
        dv_p99=float(np.percentile(dv, 99)),
        final_range_mean=float(np.mean([r.final_range for r in results])),
        rms_nav_pos_err_mean=float(np.mean([r.rms_nav_pos_err for r in results])),
        anees_mean=float(anees),
    )


def anees_bounds(n_trials, dof=6, alpha=0.05):
    from math import sqrt
    z = _norm_ppf(1.0 - alpha / 2.0)
    k = dof * n_trials

    def chi2_q(sign):
        # Wilson-Hilferty
        return k * (1.0 - 2.0 / (9 * k) + sign * z * sqrt(2.0 / (9 * k))) ** 3

    return chi2_q(-1) / n_trials, chi2_q(+1) / n_trials


def _norm_ppf(p):
    """Acklam's approximation to the standard-normal inverse CDF"""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = (-2 * np.log(p)) ** 0.5
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    q = (-2 * np.log(1 - p)) ** 0.5
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
