import numpy as np

from rpo import dynamics, guidance


def test_glideslope_terminal_range_close_to_target():
    """A planned glideslope should drive the chaser to within rho_f of the
    target in the open-loop truth dynamics (no noise, no thrust error)."""
    n = dynamics.mean_motion(400e3)
    s0 = np.array([-200.0, -2000.0, 50.0, 0.05, -0.1, 0.0])
    plan = guidance.plan_glideslope(s0, n, n_pulses=6, closing_speed=0.4, rho_f=5.0)

    s = s0.copy()
    t_prev = 0.0
    for p in plan.pulses:
        if p.t > t_prev:
            s = dynamics.propagate(s, n, p.t - t_prev)
        s = dynamics.apply_impulse(s, p.dv)
        t_prev = p.t
    rho_final = float(np.linalg.norm(s[:3]))
    # The glideslope's terminal waypoint sits at rho_f (terminal hold point),
    # and the final braking burn nulls velocity there. So we should be at
    # rho_f to several decimal places, with negligible residual velocity.
    assert abs(rho_final - 5.0) < 1e-3
    assert np.linalg.norm(s[3:]) < 1e-6


def test_glideslope_total_dv_reasonable():
    """Sanity-check fuel cost is in the single-digit m/s range for a 2 km
    transfer at 0.4 m/s closing speed."""
    n = dynamics.mean_motion(400e3)
    s0 = np.array([-200.0, -2000.0, 50.0, 0.05, -0.1, 0.0])
    plan = guidance.plan_glideslope(s0, n, n_pulses=6, closing_speed=0.4, rho_f=5.0)
    assert 0.5 < plan.total_dv < 30.0
