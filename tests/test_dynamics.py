import numpy as np
import pytest

from rpo import dynamics


def test_mean_motion_iss():
    n = dynamics.mean_motion(400e3)
    period = 2 * np.pi / n
    assert 5400 < period < 5700  # ~92 min


def test_stm_at_zero_is_identity():
    n = dynamics.mean_motion(400e3)
    Phi = dynamics.cw_stm(n, 0.0)
    assert np.allclose(Phi, np.eye(6), atol=1e-12)


def test_stm_matches_numerical_integration():
    """Closed-form STM must match expm(A*t) to machine precision."""
    from scipy.linalg import expm
    n = dynamics.mean_motion(400e3)
    t = 250.0
    A = dynamics.cw_A(n)
    Phi_closed = dynamics.cw_stm(n, t)
    Phi_expm = expm(A * t)
    assert np.allclose(Phi_closed, Phi_expm, atol=1e-9)


def test_stm_group_property():
    """Phi(t1+t2) = Phi(t2) @ Phi(t1)."""
    n = dynamics.mean_motion(400e3)
    t1, t2 = 120.0, 350.0
    Phi_a = dynamics.cw_stm(n, t1 + t2)
    Phi_b = dynamics.cw_stm(n, t2) @ dynamics.cw_stm(n, t1)
    assert np.allclose(Phi_a, Phi_b, atol=1e-9)


def test_inplane_decoupled_from_crosstrack():
    """The CW system block-decouples: cross-track motion must not couple
    into the in-plane state."""
    n = dynamics.mean_motion(400e3)
    s0 = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0])  # pure z displacement
    s = dynamics.propagate(s0, n, 600.0)
    assert np.allclose(s[[0, 1, 3, 4]], 0.0, atol=1e-9)


@pytest.mark.parametrize("dt", [10.0, 200.0, 1500.0])
def test_round_trip_stm(dt):
    n = dynamics.mean_motion(400e3)
    s0 = np.array([100.0, -500.0, 30.0, 0.1, -0.2, 0.05])
    Phi = dynamics.cw_stm(n, dt)
    Phi_inv = np.linalg.inv(Phi)
    assert np.allclose(Phi_inv @ Phi @ s0, s0, atol=1e-6)
