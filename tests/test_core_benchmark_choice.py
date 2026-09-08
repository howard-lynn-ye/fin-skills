"""fin_skills.core.benchmark_choice - the same fills score differently against every benchmark."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fin_skills.core.benchmark_choice import (benchmarks, cost_bps, make_day, run,
                                              schedule_front, schedule_opportunistic,
                                              schedule_twap, schedule_vwap)


@pytest.fixture(scope="module")
def day():
    return make_day()


def test_make_day_is_seeded_and_drifts_against_the_buyer(day):
    px, vol = day
    px2, vol2 = make_day(seed=7)
    assert np.array_equal(px, px2) and np.array_equal(vol, vol2)
    assert not np.array_equal(px, make_day(seed=8)[0])
    # the +120 bps drift is imposed and the noise is pinned at both ends
    assert px[-1] / px[0] == pytest.approx(math.exp(120 / 1e4), rel=1e-4)
    assert len(px) == 390 and (vol > 0).all()


def test_cost_sign_convention():
    assert cost_bps(101.0, 100.0) == pytest.approx(100.0)          # buyer paid up: a cost
    assert cost_bps(101.0, 100.0, side=-1) == pytest.approx(-100.0)  # seller got more: a gain


def test_schedules_sum_to_the_order(day):
    px, vol = day
    for sched in (schedule_twap, schedule_vwap, schedule_opportunistic, schedule_front):
        fills = sched(px, vol, 100_000)
        assert fills.sum() == pytest.approx(100_000)
        assert (fills >= 0).all()
    assert np.allclose(schedule_twap(px, vol, 390), 1.0)
    assert np.allclose(schedule_vwap(px, vol, 1.0), vol / vol.sum())


def test_opportunistic_only_trades_the_cheap_prints(day):
    px, vol = day
    fills = schedule_opportunistic(px, vol, 100_000, quantile=0.4)
    assert (px[fills > 0] <= np.quantile(px, 0.4)).all()


def test_front_loaded_finishes_inside_the_first_fraction(day):
    px, vol = day
    fills = schedule_front(px, vol, 100_000, frac=0.15)
    assert fills[int(390 * 0.15):].sum() == 0
    assert run("front", schedule_front, px, vol, float(px[0]), 100_000)["done_by_min"] <= 59


def test_vwap_schedule_scores_zero_against_vwap_but_eats_the_drift(day):
    # the documented point: a zero VWAP score is compatible with any amount of shortfall
    px, vol = day
    r = run("vwap", schedule_vwap, px, vol, float(px[0]), 100_000)
    assert abs(r["interval VWAP"]) < 1e-9
    assert r["decision (arrival)"] > 0
    assert r["decision (arrival)"] == pytest.approx(cost_bps(r["avg_fill"], float(px[0])))


def test_benchmarks_are_the_documented_five(day):
    px, vol = day
    b = benchmarks(px, vol, float(px[0]))
    assert set(b) == {"decision (arrival)", "interval VWAP", "interval TWAP", "close", "open"}
    assert b["interval TWAP"] == pytest.approx(px.mean())
    assert b["interval VWAP"] == pytest.approx(np.sum(px * vol) / np.sum(vol))


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.core.benchmark_choice")
    assert "Rule: quote implementation shortfall against the DECISION price" in out
