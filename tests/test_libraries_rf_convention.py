"""fin_skills.libraries.rf_convention - quantstats' rf is annual, empyrical's is per period."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import has_module, requires
from fin_skills.libraries.rf_convention import (PERIODS, _sharpe_annual_rf,
                                                _sharpe_per_period_rf, demo)


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0005, 0.01, PERIODS * 4), index=pd.bdate_range("2021-01-01", periods=PERIODS * 4))


def test_annual_convention_de_annualises_geometrically(returns):
    per_period = (1.05) ** (1 / PERIODS) - 1
    assert _sharpe_annual_rf(returns, 0.05) == pytest.approx(_sharpe_per_period_rf(returns, per_period))
    assert _sharpe_annual_rf(returns, 0.0) == pytest.approx(_sharpe_per_period_rf(returns, 0.0))


def test_the_same_0_05_passed_raw_per_period_is_off_by_252(returns):
    no_rf = _sharpe_per_period_rf(returns, 0.0)
    raw = _sharpe_per_period_rf(returns, 0.05)
    over = _sharpe_per_period_rf(returns, 0.05 / PERIODS)
    # subtracting a constant leaves the sd alone, so the Sharpe shift is linear in the rate
    assert (no_rf - raw) / (no_rf - over) == pytest.approx(PERIODS)
    assert raw < -10                                          # reads as a data bug
    assert abs(over - _sharpe_annual_rf(returns, 0.05)) < 0.05  # /252 is close, not identical


def test_demo_is_seeded_and_carries_the_four_reference_numbers():
    a, b = demo(seed=0), demo(seed=0)
    for key in ("qs_rf_0.05_annual", "ep_rf_0.05_raw", "ep_rf_0.05_over_252", "no_rf"):
        assert a[key] == b[key]
    assert a["no_rf"] != demo(seed=1)["no_rf"]
    assert a["ep_rf_0.05_raw"] < -10 < a["ep_rf_0.05_over_252"]
    assert (a["no_rf"] - a["ep_rf_0.05_raw"]) / (a["no_rf"] - a["ep_rf_0.05_over_252"]) == pytest.approx(PERIODS)


@pytest.mark.skipif(has_module("quantstats") and has_module("empyrical"),
                    reason="both libraries are installed")
def test_demo_has_no_live_keys_without_the_libraries():
    assert not any(k.startswith("LIVE_") for k in demo())


@requires("quantstats")
@requires("empyrical")
def test_installed_libraries_match_the_reference_implementations():
    r = demo()
    assert r["LIVE_qs.stats.sharpe(rf=0.05)"] == pytest.approx(r["qs_rf_0.05_annual"], abs=1e-9)
    assert r["LIVE_ep.sharpe_ratio(risk_free=0.05)"] == pytest.approx(r["ep_rf_0.05_raw"], abs=1e-9)
    assert r["LIVE_ep.sharpe_ratio(risk_free=0.05/252)"] == pytest.approx(r["ep_rf_0.05_over_252"], abs=1e-9)


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.libraries.rf_convention")
    assert "Rule: quantstats/ffn/PyPortfolioOpt take an ANNUAL rate." in out
