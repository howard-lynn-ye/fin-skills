"""fin_skills.libraries.greeks_scaling - vollib's greeks are already scaled; the textbook's are not."""
from __future__ import annotations

import math

import pytest

from _helpers import QUANTLIB_CALL, VOLLIB_CALL
from conftest import requires
from fin_skills.libraries.greeks_scaling import (FLAG, K, Q, R, S, SIGMA, SKILL_MD_TABLE, T,
                                                 live_vollib, raw_greeks, scaled_greeks)


@pytest.fixture(scope="module")
def raw():
    return raw_greeks(S, K, T, R, Q, SIGMA, FLAG)


def test_raw_greeks_are_the_textbook_absolute_numbers(raw):
    for key, ref in QUANTLIB_CALL.items():
        assert raw[key] == pytest.approx(ref, abs=1e-6), key


def test_scaled_greeks_divide_by_100_365_100_and_leave_the_rest_alone(raw):
    scaled = scaled_greeks(raw)
    assert SKILL_MD_TABLE == {"vega": 100.0, "theta": 365.0, "rho": 100.0}
    for g, div in SKILL_MD_TABLE.items():
        assert scaled[g] == pytest.approx(raw[g] / div)
        assert scaled[g] == pytest.approx(VOLLIB_CALL[g], abs=1e-6)
    for g in ("price", "delta", "gamma"):
        assert scaled[g] == raw[g]


def test_put_call_parity_and_put_signs():
    call = raw_greeks(S, K, T, R, Q, SIGMA, "c")
    put = raw_greeks(S, K, T, R, Q, SIGMA, "p")
    assert call["price"] - put["price"] == pytest.approx(S * math.exp(-Q * T) - K * math.exp(-R * T))
    assert put["delta"] < 0 < call["delta"] and put["rho"] < 0 < call["rho"]
    assert put["gamma"] == pytest.approx(call["gamma"]) and put["vega"] == pytest.approx(call["vega"])


def test_cross_module_agreement_with_greeks_convention(raw):
    from fin_skills.core.greeks_convention import black_scholes
    mine = black_scholes(S, K, T, R, Q, SIGMA, "c")
    for key in raw:
        assert raw[key] == pytest.approx(mine[key], abs=1e-9), key


@pytest.mark.skipif(live_vollib() is not None, reason="py_vollib is installed")
def test_live_vollib_is_none_without_the_library():
    assert live_vollib() is None


@requires("py_vollib")
def test_installed_vollib_returns_the_scaled_convention(raw):
    live = live_vollib()
    assert live is not None
    scaled = scaled_greeks(raw)
    for g in ("price", "delta", "gamma", "vega", "theta", "rho"):
        assert live[g] == pytest.approx(scaled[g], abs=1e-8), g
    for g, div in SKILL_MD_TABLE.items():
        assert raw[g] / live[g] == pytest.approx(div, rel=1e-6), g


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.libraries.greeks_scaling")
    assert "Rule: vollib hands you SCREEN units" in out
