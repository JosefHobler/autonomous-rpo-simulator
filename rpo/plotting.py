"""Diagnostic plots + 3D approach animation for the RPO sim."""
from __future__ import annotations

from typing import Optional

import numpy as np


def plot_summary(result, save_path: Optional[str] = None, show=False):
    import matplotlib.pyplot as plt

    truth = result.truth
    est   = result.estimate
    cov   = result.cov_diag
    t     = result.t

    err   = est - truth
    sigma = np.sqrt(cov)

    fig = plt.figure(figsize=(13, 9))
    gs  = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    # trajectory in the orbital plane (radial vs along-track)
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_traj.plot(truth[:, 1], truth[:, 0], "C0-",  lw=1.6, label="truth")
    ax_traj.plot(est[:, 1],   est[:, 0],   "C1--", lw=1.0, label="EKF")
    ax_traj.scatter([0], [0], c="k", marker="*", s=120, zorder=5, label="target")
    for p in result.guidance_pulses:
        ax_traj.scatter(p.r_target[1], p.r_target[0],
                        marker="x", s=40, c="C3", zorder=4)
    ax_traj.set_xlabel("along-track y [m]")
    ax_traj.set_ylabel("radial x [m]")
    ax_traj.set_title("Relative trajectory (LVLH x-y)")
    ax_traj.invert_xaxis()   # right-to-left closure reads better this way
    ax_traj.grid(alpha=0.3)
    ax_traj.legend(loc="best")
    ax_traj.set_aspect("equal", adjustable="datalim")

    # position error with 3-sigma envelope
    for i, lbl in enumerate(("x", "y", "z")):
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(t, err[:, i], "C0-", lw=1.0)
        ax.fill_between(t, -3 * sigma[:, i], 3 * sigma[:, i],
                        color="C0", alpha=0.15, label=r"$\pm 3\sigma$")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(f"e_{lbl} [m]")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_title("EKF position error")
            ax.legend(loc="upper right", fontsize=8)
        if i == 2:
            ax.set_xlabel("time [s]")

    # range / range-rate / cumulative dv
    rho = np.linalg.norm(truth[:, :3], axis=1)
    rho_dot = np.gradient(rho, t)

    ax_rho = fig.add_subplot(gs[0, 2])
    ax_rho.plot(t, rho, "C2-")
    ax_rho.set_ylabel("range [m]")
    ax_rho.grid(alpha=0.3)
    ax_rho.set_title("Closure profile")

    ax_rdot = fig.add_subplot(gs[1, 2])
    ax_rdot.plot(t, rho_dot, "C3-")
    ax_rdot.set_ylabel("range rate [m/s]")
    ax_rdot.axhline(0, color="k", lw=0.5)
    ax_rdot.grid(alpha=0.3)

    ax_dv = fig.add_subplot(gs[2, 2])
    cumdv = np.zeros_like(t)
    for p in result.guidance_pulses:
        cumdv[t >= p.t] += float(np.linalg.norm(p.dv))
    ax_dv.step(t, cumdv, "C4-", where="post")
    ax_dv.set_xlabel("time [s]")
    ax_dv.set_ylabel("cumulative |dv| [m/s]")
    ax_dv.grid(alpha=0.3)

    fig.suptitle(
        f"RPO sim  -  total dv = {result.delta_v_total:.3f} m/s,  "
        f"final range = {rho[-1]:.2f} m",
        fontsize=12,
    )

    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_mc_summary(results, stats, save_path: Optional[str] = None, show=False):
    """Monte Carlo campaign figure: delta-v hist + CDF, range spaghetti with a
    percentile envelope, and ensemble ANEES against its chi-square bounds."""
    import matplotlib.pyplot as plt

    from . import montecarlo

    dv = np.array([r.delta_v_total for r in results])

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # delta-v distribution
    ax_dv = fig.add_subplot(gs[0, 0])
    ax_dv.hist(dv, bins=min(40, max(8, len(dv) // 5)),
               color="C4", alpha=0.8, edgecolor="white")
    ax_dv.axvline(stats.dv_mean, color="k", lw=1.2, label=f"mean {stats.dv_mean:.3f}")
    ax_dv.axvline(stats.dv_p95, color="C3", ls="--", lw=1.0, label=f"p95 {stats.dv_p95:.3f}")
    ax_dv.axvline(stats.dv_p99, color="C1", ls=":", lw=1.0, label=f"p99 {stats.dv_p99:.3f}")
    ax_dv.set_xlabel("total |dv| [m/s]")
    ax_dv.set_ylabel("trials")
    ax_dv.set_title("Delta-v distribution")
    ax_dv.legend(fontsize=8)
    ax_dv.grid(alpha=0.3)

    # delta-v empirical CDF
    ax_cdf = fig.add_subplot(gs[0, 1])
    xs = np.sort(dv)
    ax_cdf.plot(xs, np.linspace(0, 1, len(xs)), "C4-")
    ax_cdf.axhline(0.95, color="C3", ls="--", lw=0.8)
    ax_cdf.set_xlabel("total |dv| [m/s]")
    ax_cdf.set_ylabel("P(dv <= x)")
    ax_cdf.set_title("Delta-v CDF")
    ax_cdf.grid(alpha=0.3)

    # range spaghetti + percentile envelope (truncated to the shortest trial)
    ax_rho = fig.add_subplot(gs[1, 0])
    min_len = min(len(r.range_profile) for r in results)
    t = results[0].t[:min_len]
    rng_stack = np.vstack([r.range_profile[:min_len] for r in results])
    for r in results[:60]:                       # cap drawn lines for clarity
        ax_rho.plot(r.t[:min_len], r.range_profile[:min_len],
                    color="C0", alpha=0.08, lw=0.8)
    lo = np.percentile(rng_stack, 5, axis=0)
    hi = np.percentile(rng_stack, 95, axis=0)
    med = np.percentile(rng_stack, 50, axis=0)
    ax_rho.fill_between(t, lo, hi, color="C0", alpha=0.25, label="5-95 %")
    ax_rho.plot(t, med, "C0-", lw=1.6, label="median")
    ax_rho.set_xlabel("time [s]")
    ax_rho.set_ylabel("range [m]")
    ax_rho.set_title(f"Closure ({len(results)} trials)")
    ax_rho.legend(fontsize=8)
    ax_rho.grid(alpha=0.3)

    # ensemble ANEES vs chi-square consistency band
    ax_nees = fig.add_subplot(gs[1, 1])
    nees_stack = np.vstack([r.nees[:min_len] for r in results])
    anees = np.nanmean(nees_stack, axis=0)
    lo_b, hi_b = montecarlo.anees_bounds(len(results))
    ax_nees.plot(t, anees, "C2-", lw=1.0, label="ensemble ANEES")
    ax_nees.axhline(6, color="k", lw=0.8, label="dof = 6")
    ax_nees.axhspan(lo_b, hi_b, color="C2", alpha=0.15,
                    label=f"95% band [{lo_b:.2f}, {hi_b:.2f}]")
    # the initial transient (large nav-init offset vs P0) dwarfs steady state;
    # cap the axis so the consistency band stays readable
    ax_nees.set_ylim(0, max(12.0, hi_b * 1.6))
    ax_nees.set_xlabel("time [s]")
    ax_nees.set_ylabel("ANEES")
    ax_nees.set_title("Filter consistency")
    ax_nees.legend(fontsize=8)
    ax_nees.grid(alpha=0.3)

    fig.suptitle(
        f"Monte Carlo  -  capture {100*stats.capture_rate:.1f}%,  "
        f"dv mean {stats.dv_mean:.3f} (p95 {stats.dv_p95:.3f}) m/s,  "
        f"ANEES {stats.anees_mean:.2f}",
        fontsize=12,
    )

    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def animate(result, save_path: Optional[str] = None, fps=30, stride=2, show=False):
    """3D animation of the chaser approaching the target in LVLH."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3D projection

    truth = result.truth[::stride]
    est   = result.estimate[::stride]
    t     = result.t[::stride]

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("along-track y [m]")
    ax.set_ylabel("radial x [m]")
    ax.set_zlabel("cross-track z [m]")
    ax.set_title("Chaser approach in LVLH frame")

    span = max(np.max(np.abs(truth[:, :3])), 1.0)
    ax.set_xlim(-1.05 * span, 1.05 * span)
    ax.set_ylim(-1.05 * span, 1.05 * span)
    ax.set_zlim(-1.05 * span, 1.05 * span)

    ax.scatter([0], [0], [0], c="k", marker="*", s=200, label="target")
    for p in result.guidance_pulses:
        ax.scatter(p.r_target[1], p.r_target[0], p.r_target[2],
                   c="C3", marker="x", s=30)

    truth_line, = ax.plot([], [], [], "C0-",  lw=1.5, label="truth")
    est_line,   = ax.plot([], [], [], "C1--", lw=1.0, label="EKF")
    chaser,     = ax.plot([], [], [], "C0o", ms=8)
    ax.legend(loc="upper right", framealpha=0.9)

    txt = ax.text2D(
        0.02, 0.97, "",
        transform=ax.transAxes,
        fontsize=10, family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="0.7", alpha=0.85),
    )

    def init():
        for ln in (truth_line, est_line, chaser):
            ln.set_data([], [])
            ln.set_3d_properties([])
        txt.set_text("")
        return truth_line, est_line, chaser, txt

    def step(i):
        truth_line.set_data(truth[:i + 1, 1], truth[:i + 1, 0])
        truth_line.set_3d_properties(truth[:i + 1, 2])
        est_line.set_data(est[:i + 1, 1], est[:i + 1, 0])
        est_line.set_3d_properties(est[:i + 1, 2])
        chaser.set_data([truth[i, 1]], [truth[i, 0]])
        chaser.set_3d_properties([truth[i, 2]])
        rho = float(np.linalg.norm(truth[i, :3]))
        txt.set_text(f"t = {t[i]:7.1f} s\nrange = {rho:7.1f} m")
        return truth_line, est_line, chaser, txt

    anim = FuncAnimation(fig, step, frames=len(t),
                         init_func=init, interval=1000.0 / fps, blit=False)

    if save_path:
        writer = "pillow" if save_path.lower().endswith(".gif") else "ffmpeg"
        anim.save(save_path, writer=writer, fps=fps)
    if show:
        plt.show()
    return anim
