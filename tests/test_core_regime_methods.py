"""fin_skills.core.regime_methods - rule-based detectors: honest forms use data through t-1."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires
from fin_skills.core import regime_methods as rm

LO, HI = rm.TEST_START, rm.N


@pytest.fixture(scope="module")
def panel():
    return rm.simulate_multi(rm.N, rm.SEED, rm.RATIO)


def test_multi_asset_simulation_is_seeded_and_correlation_switches_with_the_regime(panel):
    R, s = panel
    R2, s2 = rm.simulate_multi(rm.N, rm.SEED, rm.RATIO)
    assert np.array_equal(R, R2) and np.array_equal(s, s2)
    assert R.shape == (rm.N, rm.K_ASSETS) and set(np.unique(s)) == {0, 1}
    calm = np.corrcoef(R[s == 0].T)[0, 1]
    turb = np.corrcoef(R[s == 1].T)[0, 1]
    assert abs(calm - rm.CORR[0]) < 0.1 and abs(turb - rm.CORR[1]) < 0.1


def test_trailing_vol_and_expanding_quantile_are_known_at_the_close_of_t_minus_1(panel):
    r = panel[0][:, 0]
    rv = rm.trailing_vol(r, rm.VOL_WINDOW)
    assert np.isnan(rv[:rm.VOL_WINDOW]).all() and np.isfinite(rv[rm.VOL_WINDOW])
    assert rv[rm.VOL_WINDOW] == pytest.approx(r[:rm.VOL_WINDOW].std(ddof=1))
    shocked = r.copy()
    shocked[500] += 0.10
    rv2 = rm.trailing_vol(shocked, rm.VOL_WINDOW)
    assert np.array_equal(np.nan_to_num(rv2[:501]), np.nan_to_num(rv[:501]))   # t=500 unchanged
    assert rv2[501] > rv[501]                                                   # t+1 sees it
    q = rm.expanding_quantile(rv, 0.75, rm.MIN_HISTORY)
    q2 = rm.expanding_quantile(rv2, 0.75, rm.MIN_HISTORY)
    assert np.array_equal(np.nan_to_num(q[:502]), np.nan_to_num(q2[:502]))


def test_turbulence_expanding_is_causal_and_full_sample_leaks(panel):
    R, _ = panel
    honest = rm.turbulence(R, rm.MIN_HISTORY, full_sample=False)
    leaky = rm.turbulence(R, rm.MIN_HISTORY, full_sample=True)
    assert np.isnan(honest[:rm.MIN_HISTORY]).all() and np.isfinite(honest[rm.MIN_HISTORY:]).all()
    assert np.isfinite(leaky).all() and (leaky >= 0).all()
    R2 = R.copy()
    R2[-1] += 0.05
    assert np.array_equal(np.nan_to_num(rm.turbulence(R2, rm.MIN_HISTORY, False)[:-1]),
                          np.nan_to_num(honest[:-1]))
    assert np.abs(rm.turbulence(R2, rm.MIN_HISTORY, True)[:-1] - leaky[:-1]).max() > 1e-6


def test_delays_into_and_evaluate_on_the_oracle(panel):
    R, s = panel
    r = R[:, 0]
    assert rm.delays_into(s, s, LO, HI, 1) == 0.0 and rm.delays_into(s, s, LO, HI, 0) == 0.0
    lagged = np.r_[0, s[:-1]]
    assert rm.delays_into(s, lagged, LO, HI, 1) == 1.0
    row = rm.evaluate("oracle lagged 1 day", "true regime at t-1", lagged, r, s, LO, HI)
    assert row["false_alarms"] == 0 and row["missed"] == 0
    bh = rm.evaluate("buy & hold", "nothing", np.zeros(rm.N), r, s, LO, HI)
    assert bh["flag_share"] == 0.0 and bh["switches"] == 0
    assert bh["sharpe"] == pytest.approx(rm.strategy_stats(r[LO:HI], np.ones(HI - LO))["sharpe"])
    assert bh["missed"] == sum(1 for t in range(LO, HI) if s[t] == 1 and s[t - 1] == 0)


@requires("statsmodels")
def test_demo_runs_the_detector_table_and_counts_its_degrees_of_freedom(run_main):
    out = run_main("fin_skills.core.regime_methods")
    assert "=== Detectors on the same regimes" in out
    assert "LEAKS" in out and "Researcher degrees of freedom used above: 9" in out
    assert out.strip().splitlines()[-1].startswith("total runtime")
