"""fin_skills.macro.seasonal_adjustment - the revision that arrives with no new data.

Properties under test: the X-11 core recovers a known seasonal pattern; the factors are
time-varying and normalised locally; a concurrent publication matrix rewrites the value of
a month whose unadjusted data never changed, while a frozen-factor one never does; the
trap fires on today's SA history and clears on the real-time diagonal; the X-13 probe
reports the exact statsmodels exception when the Census binary is absent.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires
from fin_skills.macro import seasonal_adjustment as sa

START = 96


@pytest.fixture(scope="module")
def sim():
    return sa.simulate_nsa()


@pytest.fixture(scope="module")
def matrices(sim):
    nsa = sim["nsa"]
    return {"concurrent": sa.publish_concurrent(nsa, START),
            "projected": sa.publish_projected(nsa, START),
            "frozen": sa.publish_frozen(nsa, START)}


def test_centered_ma_is_the_2x12_filter():
    x = np.ones(40)
    ma = sa.centered_ma(x)
    assert np.isnan(ma[:6]).all() and np.isnan(ma[-6:]).all()
    assert np.allclose(ma[6:-6], 1.0)
    # a pure 12-month cycle averages away
    cyc = np.sin(2 * np.pi * np.arange(60) / 12) + 5.0
    assert np.allclose(sa.centered_ma(cyc)[6:-6], 5.0, atol=1e-12)


def test_seasonal_factors_recover_a_known_pattern_and_are_time_varying(sim):
    fac = sa.seasonal_factors(sim["nsa"])
    assert fac.shape == sim["nsa"].shape
    assert (fac > 0).all()
    # normalised: a centred 12-term average of the log factors is ~0. X-11 normalises in
    # a single pass, so this is approximate - a few 1e-4 in logs, an order of magnitude
    # below the smallest rewrite the script measures, and stable under further passes.
    lf = np.log(fac)
    assert np.abs(sa.centered_ma(lf)[6:-6]).max() < 1e-3
    # time-varying: the same calendar month does not carry one constant factor
    jan = fac[0::12]
    assert jan.std() > 1e-4
    # and they track the truth: adjusting removes most of the seasonal
    adj = sa.adjust(sim["nsa"])
    err = np.abs(adj / sim["truth_sa"] - 1.0)
    raw = np.abs(sim["nsa"] / sim["truth_sa"] - 1.0)
    assert err[12:-12].mean() < 0.25 * raw[12:-12].mean()


def test_simulation_is_seeded(sim):
    again = sa.simulate_nsa()
    assert np.array_equal(sim["nsa"], again["nsa"])
    assert not np.array_equal(sim["nsa"], sa.simulate_nsa(seed=sa.SEED + 1)["nsa"])
    assert (sim["nsa"] > 0).all()


def test_publication_matrices_are_causal(matrices):
    for name, P in matrices.items():
        n = P.shape[0]
        for m in (START, 150, n - 1):
            assert np.isfinite(P[m, :m + 1]).all(), name
            assert np.isnan(P[m, m + 1:]).all(), name       # nothing published ahead
        assert np.isnan(P[START - 1]).all(), name


def test_the_trap_a_settled_month_is_rewritten_under_concurrent_and_not_when_frozen(matrices):
    prof_c = sa.rewrite_profile(matrices["concurrent"], START)
    prof_f = sa.rewrite_profile(matrices["frozen"], START)
    # ages past the 2-month NSA revision window cannot be new data
    settled = [a for a in prof_c.index if a > sa.NSA_REVISION_WINDOW]
    assert settled and all(not prof_c.loc[a, "new_data_possible"] for a in settled)
    assert all(prof_c.loc[a, "mean_abs_pct"] > 0 for a in settled)
    assert all(prof_f.loc[a, "mean_abs_pct"] == 0.0 for a in prof_f.index)
    # the biggest single rewrite lands when the centred filter first reaches the month
    assert prof_c.loc[6, "mean_abs_pct"] > prof_c.loc[12, "mean_abs_pct"]
    assert prof_c.loc[12, "mean_abs_pct"] > prof_c.loc[24, "mean_abs_pct"]
    sd = sa.settled_drift(matrices["concurrent"], START)
    assert sd["n"] > 50
    assert sd["mean_path_pct"] >= sd["mean_abs_net_pct"] > 0     # it wanders, then lands
    assert sa.settled_drift(matrices["frozen"], START)["mean_path_pct"] == 0.0


def test_the_phantom_signal_is_absent_from_a_frozen_regime(matrices):
    ph = sa.phantom_signal(matrices["concurrent"], START)
    assert ph["n"] > 150
    assert ph["sd_revision"] > 0 and ph["rev_over_final"] > 0.5
    assert 0.05 < ph["unexplained"] < 0.9
    assert ph["sign_flip"] > 0.05
    flat = sa.phantom_signal(matrices["frozen"], START)
    assert flat["sd_revision"] == pytest.approx(0.0, abs=1e-8)
    assert flat["sign_flip"] == 0.0
    assert flat["corr"] == pytest.approx(1.0)


def test_re_adjustment_buys_accuracy_and_sells_stability(sim, matrices):
    truth = sim["truth_sa"]
    acc_c = sa.accuracy(matrices["concurrent"], truth, START)
    acc_f = sa.accuracy(matrices["frozen"], truth, START)
    # concurrent: the settled history is more accurate than what was published live
    assert acc_c["rmse_final_pct"] < acc_c["rmse_realtime_pct"]
    # frozen: nothing changes, so the two are identical - and it is the least accurate
    assert acc_f["rmse_final_pct"] == pytest.approx(acc_f["rmse_realtime_pct"])
    assert acc_f["rmse_final_pct"] > acc_c["rmse_final_pct"]


def test_projected_regime_rewrites_annually_not_monthly(sim, matrices):
    P = matrices["projected"]
    n = P.shape[0]
    # a settled month's published value is unchanged in most months and jumps in one
    moves = [abs(P[m, m - 18] / P[m - 1, m - 18] - 1.0)
             for m in range(START + 20, n) if np.isfinite(P[m - 1, m - 18])]
    moves = np.asarray(moves)
    assert (moves < 1e-12).mean() > 0.7           # mostly frozen
    assert moves.max() > 1e-6                     # and it does move, once a year


def test_latest_by_month_carries_one_factor_per_calendar_position(sim):
    fac = sa.seasonal_factors(sim["nsa"])
    last = sa.latest_by_month(fac)
    assert last.shape == (12,)
    for k in range(12):
        assert last[k] == fac[np.arange(k, fac.size, 12)[-1]]


@requires("statsmodels")
def test_x13_status_reports_the_binary_and_its_exact_exception_when_absent():
    st = sa.x13_status()
    assert st["statsmodels"] is True
    assert "x13as" in st["binary_names"] and "x12a" in st["binary_names"]
    if not st["binary_found"]:
        assert st["exception"] == "statsmodels.tools.sm_exceptions.X13NotFoundError"
        assert "not found on path" in st["message"]
        assert "X13PATH" in st["message"]
        assert sa.compare_with_x13(np.arange(120, dtype=float) + 100) is None


@pytest.mark.slow
def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.macro.seasonal_adjustment")
    for head in ("1. What the agency says it does",
                 "2. How far a settled month's PUBLISHED value moves",
                 "3. The phantom signal", "4. Freezing the factors",
                 "5. X-13ARIMA-SEATS in Python"):
        assert head in out
    assert "concurrent methodology" in out and "5 years" in out
    assert out.strip().splitlines()[-1].startswith("Rule:")
    assert all(ord(c) < 128 for c in out)
