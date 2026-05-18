from __future__ import annotations

import numpy as np

MU_EARTH = 3.986004418e14
R_EARTH  = 6_378_137.0


def mean_motion(alt_m):
    a = R_EARTH + alt_m
    return float(np.sqrt(MU_EARTH / a**3))


def cw_A(n):
    A = np.zeros((6, 6))
    A[0, 3] = A[1, 4] = A[2, 5] = 1.0
    A[3, 0] =  3 * n * n
    A[3, 4] =  2 * n
    A[4, 3] = -2 * n
    A[5, 2] = -n * n
    return A


def cw_B():
    B = np.zeros((6, 3))
    B[3:, :] = np.eye(3)
    return B


def cw_stm(n, t):
    """Closed-form CW state transition. s(t) = Phi @ s(0)."""
    nt = n * t
    s, c = np.sin(nt), np.cos(nt)
    return np.array([
        [4 - 3*c,        0,  0,  s/n,             2*(1 - c)/n,         0],
        [6*(s - nt),     1,  0, -2*(1 - c)/n,    (4*s - 3*nt)/n,       0],
        [0,              0,  c,  0,               0,                   s/n],
        [3*n*s,          0,  0,  c,               2*s,                 0],
        [-6*n*(1 - c),   0,  0, -2*s,             4*c - 3,             0],
        [0,              0, -n*s, 0,              0,                   c],
    ])


def split_stm(Phi):
    return Phi[:3, :3], Phi[:3, 3:], Phi[3:, :3], Phi[3:, 3:]


def propagate(s0, n, dt):
    return cw_stm(n, dt) @ s0


def apply_impulse(s, dv):
    out = s.copy()
    out[3:] += dv
    return out
