import numpy as np

from rpo import dynamics, ekf, measurements


def test_ekf_converges_on_truth():
    """With dense, low-noise measurements the EKF should drive position
    error well below the initial uncertainty."""
    rng = np.random.default_rng(0)
    n = dynamics.mean_motion(400e3)
    s_truth = np.array([-150.0, -1500.0, 30.0, 0.05, -0.1, 0.0])

    init_sigma_pos = 30.0
    init_sigma_vel = 0.3
    s0 = s_truth + rng.normal(scale=[init_sigma_pos] * 3 + [init_sigma_vel] * 3)
    P0 = np.diag([init_sigma_pos**2] * 3 + [init_sigma_vel**2] * 3)

    filt = ekf.PythonEKF(n, s0, P0)
    sensor = measurements.Sensor(measurements.SensorParams(
        sigma_range=0.05, sigma_bearing=5e-4, rng_seed=1))
    R = sensor.params.R

    dt = 0.5
    for k in range(400):
        z = sensor.sample(s_truth)
        filt.update(z, sensor.predict, sensor.jacobian, R)
        s_truth = dynamics.propagate(s_truth, n, dt)
        filt.predict(dt)

    err = filt.x - s_truth
    pos_err = np.linalg.norm(err[:3])
    assert pos_err < 5.0    # huge improvement vs init 30 m / axis


def test_covariance_stays_psd():
    n = dynamics.mean_motion(400e3)
    s0 = np.array([100.0, -500.0, 0.0, 0.0, 0.0, 0.0])
    P0 = np.diag([20.0, 20.0, 20.0, 0.5, 0.5, 0.5]) ** 2
    filt = ekf.PythonEKF(n, s0, P0)
    sensor = measurements.Sensor(measurements.SensorParams(rng_seed=2))
    s_truth = s0.copy()
    for _ in range(50):
        z = sensor.sample(s_truth)
        filt.update(z, sensor.predict, sensor.jacobian, sensor.params.R)
        filt.predict(1.0)
        s_truth = dynamics.propagate(s_truth, n, 1.0)
        eigs = np.linalg.eigvalsh(filt.P)
        assert eigs.min() > -1e-9   # PSD up to numerical noise
