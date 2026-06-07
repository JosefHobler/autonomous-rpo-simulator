"""Monte Carlo RPO campaign: disperse the initial state, run many trials,
report delta-v / capture / filter-consistency statistics."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rpo import simulator, montecarlo, plotting, ccsds, measurements, ekf


def _base_config(seed):
    return simulator.SimConfig(
        rng_seed=seed,
        altitude_m=800e3,
        dt_sim=0.5,
        dt_meas=2.0,
        n_pulses=8,
        closing_speed=0.7,
        rho_f=2.0,
        cone_half_angle_deg=8.0,
        duration_s=None,
        initial_relative_state=np.array([
            -300.0, -5000.0, 250.0,
              0.10,   -0.30,  -0.15,
        ]),
        sensor=measurements.SensorParams(sigma_range=0.5, sigma_bearing=3e-3),
        ekf_params=ekf.EKFParams(
            process_accel_psd=1e-5,
            initial_pos_sigma=150.0,
            initial_vel_sigma=1.5,
        ),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None,
                    help="process count; 1 = serial")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    mc = montecarlo.MCConfig(
        base=_base_config(args.seed),
        dispersions=montecarlo.Dispersions(),
        n_trials=args.trials,
        seed0=args.seed,
        workers=args.workers,
    )

    print(f"Running Monte Carlo: {args.trials} trials "
          f"(workers={args.workers or 'auto'}) ...")

    def progress(done, total):
        if done % max(1, total // 20) == 0 or done == total:
            print(f"  {done:4d}/{total} trials", end="\r", flush=True)

    results = montecarlo.run_campaign(mc, progress=progress)
    print()
    stats = montecarlo.summarize(results)

    lo, hi = montecarlo.anees_bounds(stats.n_trials)
    consistent = "ok" if lo <= stats.anees_mean <= hi else "OUT OF BAND"
    print(f"  trials            : {stats.n_trials}")
    print(f"  capture rate      : {100*stats.capture_rate:.1f}%")
    print(f"  cone violations   : {100*stats.cone_violation_rate:.1f}%")
    print(f"  dv mean / p95/p99 : {stats.dv_mean:.3f} / "
          f"{stats.dv_p95:.3f} / {stats.dv_p99:.3f} m/s")
    print(f"  final range mean  : {stats.final_range_mean:.2f} m")
    print(f"  RMS nav pos err   : {stats.rms_nav_pos_err_mean:.3f} m")
    print(f"  ANEES mean        : {stats.anees_mean:.2f} "
          f"(band [{lo:.2f}, {hi:.2f}] -> {consistent})")

    # round the campaign summary through CCSDS, same as the rest of the project
    tlm = ccsds.TelemetryStream()
    tlm.emit(ccsds.APID.CAMPAIGN_SUMMARY,
             ccsds.encode_campaign(stats.n_trials, stats.capture_rate,
                                   stats.dv_mean, stats.dv_p95, stats.dv_p99,
                                   stats.anees_mean),
             0.0)
    tlm_path = os.path.join(args.out, "mc_summary.bin")
    tlm.write(tlm_path)
    decoded = ccsds.decode_campaign(
        ccsds.TelemetryStream.read(tlm_path)[0].payload)
    print(f"  wrote {tlm_path}  (decoded back: capture={decoded['capture_rate']:.3f})")

    csv_path = os.path.join(args.out, "mc_results.csv")
    with open(csv_path, "w") as f:
        f.write("seed,delta_v_total,final_range,final_speed,"
                "rms_nav_pos_err,rms_nav_vel_err,cone_violation,captured\n")
        for r in results:
            f.write(f"{r.seed},{r.delta_v_total:.6f},{r.final_range:.6f},"
                    f"{r.final_speed:.6f},{r.rms_nav_pos_err:.6f},"
                    f"{r.rms_nav_vel_err:.6f},{int(r.cone_violation)},"
                    f"{int(r.captured)}\n")
    print(f"  wrote {csv_path}")

    if not args.no_plot:
        png_path = os.path.join(args.out, "mc_summary.png")
        plotting.plot_mc_summary(results, stats, save_path=png_path)
        print(f"  wrote {png_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
