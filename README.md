# RPO Simulator

Small simulation of a chaser doing a glideslope rendezvous with a target on a
circular orbit. Truth dynamics, guidance, EKF, sensor model and CCSDS telemetry

<img width="800" height="600" alt="approach" src="https://github.com/user-attachments/assets/69ebeab3-358c-46fa-877f-c7e176c3751f" />
<br/>
<img width="1544" height="1183" alt="summary" src="https://github.com/user-attachments/assets/017404ae-bc67-455f-846b-da520ce06552" />

What is included:

- Clohessy–Wiltshire relative dynamics.
- Multi-impulse glideslope guidance (Hablani formulation). Waypoints aligned with the initial Line-of-Sight (LOS) vector, per-leg delta-v from a CW two-point BVP.
- Range / az / el sensor with Gaussian noise + analytical Jacobian.
- EKF on the 6-element state vector. Pure Python by default. A header-only
  C++ core with pybind11 bindings is available for better performance.
- CCSDS 133.0-B-2 Space Packets (primary header, CDS short time, typed payloads).
  Every truth/nav/sensor/guidance event gets emitted, written to `.bin`, and read
  back to validate packet serialization and telemetry framing.
- summary plot + 3D animation of the approach in LVLH.

```
rpo/
  dynamics.py     CW system matrix + closed-form STM
  guidance.py     glideslope planner (multi-impulse CW BVP)
  measurements.py range/az/el sensor and Jacobian
  ekf.py          EKF (auto-loads the C++ core if you built it)
  ccsds.py        Space Packet header, CDS time, payload codecs
  simulator.py    top-level loop
  montecarlo.py   dispersed-IC campaign, parallel trials, NEES consistency
  plotting.py     summary figure + 3D animation + MC campaign figure
cpp/
  include/rpo/ekf_core.hpp   header-only Eigen EKF (everything lives here)
  src/pybindings.cpp         pybind11 module 'rpo._ekf_cpp'
  tests/test_ekf_core.cpp    initial check, no Python needed
  CMakeLists.txt             builds the initial check + the pybind11 module
scripts/
  run_sim.py                 'python scripts/run_sim.py'
  run_mc.py                  'python scripts/run_mc.py' (Monte Carlo campaign)
tests/                       pytest, 29 tests
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_sim.py --out out
```

Output:

- `out/summary.png` — trajectory, EKF error envelopes, range/range-rate and a
  cumulative-delta-v staircase.
- `out/approach.gif` — 3D animation.
- `out/telemetry.bin` — raw concatenated CCSDS packets.

Inspect the telemetry:

```python
from rpo import ccsds
pkts = ccsds.TelemetryStream.read("out/telemetry.bin")
events = [p for p in pkts if p.apid == int(ccsds.APID.EVENT)]
for e in events:
    print(e.timestamp.seconds, ccsds.decode_event(e.payload))
```

## Monte Carlo

```bash
python scripts/run_mc.py --trials 200 --workers 8 --out out
```

Disperses the initial relative state (1-sigma defaults: 20/20/10 m,
0.02/0.02/0.01 m/s), reseeds the noise streams per trial, and runs the trials
across a process pool. One integer seed fully determines a trial so a campaign is reproducible.

Output:

- `out/mc_summary.png`: delta-v histogram + CDF, range closure with a 5-95 %
  envelope, and the ensemble ANEES against its chi-square consistency band.
- `out/mc_results.csv`: per-trial delta-v, capture, cone violation, RMS nav error.
- `out/mc_summary.bin`: a single CCSDS `CAMPAIGN_SUMMARY` packet (capture rate,
  delta-v percentiles, ANEES) framed like the rest of the telemetry.

The headline analysis is filter consistency: with N independent runs the
ensemble-averaged NEES is chi-square distributed, so it should sit inside
`anees_bounds(N)`. Drifting above means the EKF is overconfident (P too small);
below means it is conservative. Capture is judged at closest approach, since the
sim coasts ~30 s past the terminal burn.

```python
from rpo import montecarlo as mc, simulator
results = mc.run_campaign(mc.MCConfig(base=simulator.SimConfig(), n_trials=200))
stats = mc.summarize(results)
print(stats.capture_rate, stats.dv_p95, stats.anees_mean)
```

## C++ EKF core

Python EKF gives the same numbers as the C++ one. I built the C++ core for (a) benchmarking, or
(b) lifting the class straight into a flight-style codebase.

```bash
pip install pybind11
EIGEN_INCLUDE_DIR=/usr/include/eigen3 pip install .
```

On Windows the included setup.py also probes
`C:/vcpkg/installed/x64-windows/include/eigen3`. If you've got Eigen anywhere
else, set `EIGEN_INCLUDE_DIR` explicitly.

Once built, `ekf.build_default_filter` finds `rpo._ekf_cpp` on its own. Quick
check:

```python
from rpo import ekf, dynamics
import numpy as np
n = dynamics.mean_motion(400e3)
filt = ekf.build_default_filter(n, np.zeros(6))
print(type(filt).__name__)  
```

There's a standalone C++ initial check at `cpp/tests/test_ekf_core.cpp`:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
./cpp/build/ekf_smoke
```

## Conventions

- **Frame**: LVLH on the target. x radial out, y along-track, z cross-track.
- **State**: chaser pos+vel relative to the target, 6 elements, SI.
- **Time**: simulator clock. t=0 at release. The CCSDS CDS time codes are
  mission-relative.
- **Thrust**: instantaneous impulses. The truth model takes the impulse exactly
  and the EKF sees the same commanded delta-v.
- **Sensor**: range / az / el in the chaser body frame, which we assume is aligned
  with LVLH. Defaults: σ_range = 0.1 m, σ_bearing = 1 mrad.

## Tests

```bash
pytest -q
```
