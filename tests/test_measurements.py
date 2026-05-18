import numpy as np

from rpo import measurements


def test_measurement_self_consistency():
    s = np.array([100.0, -500.0, 50.0, 0.0, 0.0, 0.0])
    z = measurements.measure(s)
    rho_truth = np.linalg.norm(s[:3])
    assert np.isclose(z[0], rho_truth)
    # Reconstruct r from (rho, az, el) and confirm.
    rho, az, el = z
    r_re = np.array([
        -rho * np.cos(el) * np.cos(az),
        -rho * np.cos(el) * np.sin(az),
        -rho * np.sin(el),
    ])
    assert np.allclose(r_re, s[:3], atol=1e-9)


def test_jacobian_matches_finite_difference():
    s = np.array([120.0, -480.0, 65.0, 0.1, -0.2, 0.05])
    H = measurements.measurement_jacobian(s)
    # Velocity columns must be exactly zero.
    assert np.allclose(H[:, 3:], 0.0)
    eps = 1e-6
    H_fd = np.zeros((3, 6))
    for i in range(3):
        ds = np.zeros(6); ds[i] = eps
        H_fd[:, i] = (measurements.measure(s + ds) - measurements.measure(s - ds)) / (2 * eps)
    assert np.allclose(H[:, :3], H_fd[:, :3], atol=1e-5)


def test_sensor_noise_seeded_repeatable():
    p = measurements.SensorParams(rng_seed=123)
    s = np.array([100.0, -500.0, 50.0, 0.0, 0.0, 0.0])
    a = measurements.Sensor(p)
    b = measurements.Sensor(p)
    for _ in range(5):
        assert np.allclose(a.sample(s), b.sample(s))
