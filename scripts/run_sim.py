from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rpo import simulator, plotting, ccsds, measurements, ekf


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--no-anim", action="store_true", help="skip the 3D animation render")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    cfg = simulator.SimConfig(
        rng_seed=args.seed,
        altitude_m=800e3,
        dt_sim=0.5,
        dt_meas=2.0,
        n_pulses=8,
        closing_speed=0.7,
        rho_f=2.0,  
        cone_half_angle_deg=8.0,
        initial_relative_state=np.array([
            -300.0, -5000.0, 250.0,
              0.10,   -0.30,  -0.15,
        ]),
        sensor=measurements.SensorParams(
            sigma_range=0.5,
            sigma_bearing=3e-3,
        ),
        ekf_params=ekf.EKFParams(
            process_accel_psd=1e-5,
            initial_pos_sigma=150.0,
            initial_vel_sigma=1.5,
        ),
    )
    print(f"Running RPO sim ... initial state = {cfg.initial_relative_state}")
    res = simulator.run(cfg)

    rho_final = float(np.linalg.norm(res.truth[-1, :3]))
    print(f"duration       : {res.t[-1]:.1f} s")
    print(f"guidance pulses: {len(res.guidance_pulses)}")
    print(f"total dv       : {res.delta_v_total:.3f} m/s")
    print(f"final range    : {rho_final:.2f} m")
    print(f"packets emitted: {len(res.telemetry.packets)}")

    summary_path = os.path.join(args.out, "summary.png")
    plotting.plot_summary(res, save_path=summary_path)
    print(f"  wrote {summary_path}")

    tlm_path = os.path.join(args.out, "telemetry.bin")
    res.telemetry.write(tlm_path)
    total_bytes = sum(len(p) for p in res.telemetry.packets)
    print(f"  wrote {tlm_path}  ({total_bytes} bytes)")

    pkts = ccsds.TelemetryStream.read(tlm_path)
    events = [p for p in pkts if p.apid == int(ccsds.APID.EVENT)]
    print(f"  decoded {len(pkts)} packets; {len(events)} EVENT messages:")
    for e in events:
        print(f"    [{e.timestamp.seconds:7.1f} s] {ccsds.decode_event(e.payload)}")

    if not args.no_anim:
        anim_path = os.path.join(args.out, "approach.gif")
        stride = max(1, len(res.t) // 150)
        try:
            plotting.animate(res, save_path=anim_path, fps=15, stride=stride)
            print(f"wrote {anim_path}  stride={stride}")
        except Exception as e:
            print(f"animation skipped: {e!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
