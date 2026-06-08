import numpy as np

from rpo import montecarlo as mc, simulator, ccsds


def _small_config():
    """A short base simulation so the campaign tests stay fast"""
    return simulator.SimConfig(
        altitude_m=400e3,
        dt_sim=1.0,
        dt_meas=2.0,
        n_pulses=4,
        closing_speed=1.0,
        rho_f=5.0,
        initial_relative_state=np.array([-150.0, -1500.0, 30.0, 0.05, -0.1, 0.0]),
    )


def test_seed_makes_run_reproducible():
    cfg = _small_config()
    a = simulator.run(cfg)
    b = simulator.run(cfg)
    assert np.allclose(a.truth, b.truth)
    assert np.allclose(a.estimate, b.estimate)


def test_different_seeds_diverge():
    import dataclasses
    cfg = _small_config()
    a = simulator.run(cfg)
    b = simulator.run(dataclasses.replace(cfg, rng_seed=1))
    assert not np.allclose(a.estimate, b.estimate)


def test_full_cov_logging_opt_in():
    cfg = _small_config()
    assert simulator.run(cfg).cov_full is None
    import dataclasses
    res = simulator.run(dataclasses.replace(cfg, log_full_cov=True))
    assert res.cov_full is not None
    assert res.cov_full.shape == (len(res.t), 6, 6)
    assert np.allclose(np.diagonal(res.cov_full, axis1=1, axis2=2), res.cov_diag)


def test_campaign_serial_runs_and_summarizes():
    m = mc.MCConfig(base=_small_config(), n_trials=6, workers=1)
    results = mc.run_campaign(m)
    assert len(results) == 6
    assert len({r.seed for r in results}) == 6

    stats = mc.summarize(results)
    assert stats.n_trials == 6
    assert 0.0 <= stats.capture_rate <= 1.0
    assert stats.dv_p99 >= stats.dv_p95 >= 0.0
    assert np.isfinite(stats.anees_mean) and stats.anees_mean > 0


def test_campaign_is_reproducible():
    m = mc.MCConfig(base=_small_config(), n_trials=4, seed0=7, workers=1)
    a = mc.summarize(mc.run_campaign(m))
    b = mc.summarize(mc.run_campaign(m))
    assert a.dv_mean == b.dv_mean
    assert a.anees_mean == b.anees_mean


def test_dispersion_changes_initial_state():
    rng = np.random.default_rng(0)
    disp = mc.Dispersions()
    d = disp.sample(rng)
    assert d.shape == (6,)
    assert np.any(d != 0.0)


def test_anees_bounds_bracket_dof():
    lo, hi = mc.anees_bounds(n_trials=200, dof=6)
    assert lo < 6 < hi
    lo2, hi2 = mc.anees_bounds(n_trials=2000, dof=6)
    assert (hi2 - lo2) < (hi - lo)


def test_campaign_summary_ccsds_roundtrip():
    m = mc.MCConfig(base=_small_config(), n_trials=4, workers=1)
    s = mc.summarize(mc.run_campaign(m))
    payload = ccsds.encode_campaign(s.n_trials, s.capture_rate, s.dv_mean,
                                    s.dv_p95, s.dv_p99, s.anees_mean)
    back = ccsds.decode_campaign(payload)
    assert back["n_trials"] == s.n_trials
    assert back["dv_mean"] == s.dv_mean
    assert back["anees_mean"] == s.anees_mean
