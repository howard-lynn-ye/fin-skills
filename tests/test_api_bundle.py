"""fin_skills.api.bundle - one container, one call, and the aliases that make it uniform."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.api import Bundle, Suite, check, coverage, input_names, registry, slots, vocabulary
from fin_skills.api.bundle import ALIASES, DATA_SLOTS, _slot_of


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def run():
    """A small, seeded research run: 400 days of strategy returns, turnover, bars, a
    causal signal function, and a regime label with several episodes."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2022-01-03", periods=400)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=idx, name="close")
    bars = pd.DataFrame({"open": close.shift(1).bfill(), "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": rng.integers(1_000, 5_000, 400)}, index=idx)
    returns = pd.Series(rng.normal(0.0005, 0.01, 400), index=idx, name="strategy")
    turnover = pd.Series(np.abs(rng.normal(0.1, 0.02, 400)), index=idx)
    regime = pd.Series(np.repeat([0, 1, 0, 1, 0, 1, 0, 1], 50), index=idx)
    return dict(close=close, bars=bars, returns=returns, turnover=turnover, regime=regime,
                underlying=close.pct_change().fillna(0.0), position=pd.Series(1.0, index=idx))


# ------------------------------------------------------------------ vocabulary
def test_vocabulary_covers_every_guard_input_after_aliasing():
    vocab = vocabulary()
    for cls in registry():
        for k in cls.required + cls.optional:
            assert _slot_of(cls.name, k) in vocab, (cls.name, k)
    # every alias points at a curated slot, and its source keyword is a real input
    curated = {s.name for s in DATA_SLOTS}
    names = input_names()
    for guard, amap in ALIASES.items():
        for keyword, slot in amap.items():
            assert slot in curated, (guard, keyword, slot)
            assert guard in names[keyword], (guard, keyword)


def test_slots_lists_curated_first_with_the_guards_each_one_reaches():
    listed = slots()
    assert [s.name for s in listed[:len(DATA_SLOTS)]] == [s.name for s in DATA_SLOTS]
    by_name = {s.name: s for s in listed}
    assert "cost_curve" in by_name["returns"].doc and "rf_convention" in by_name["returns"].doc
    assert "regime_coverage" in by_name["returns"].doc          # via its `strategy` alias
    assert "warmup_probe" in by_name["close"].doc               # via its `closes` alias
    assert all(s.doc.isascii() for s in listed)


def test_unknown_slot_is_rejected_with_a_hint():
    with pytest.raises(TypeError, match="unknown slot"):
        Bundle(retruns=pd.Series([0.1]))
    with pytest.raises(TypeError, match="did you mean.*returns"):
        Bundle(retruns=pd.Series([0.1]))


# ------------------------------------------------------------------ construction checks
def test_shape_and_order_mistakes_are_caught_at_construction(run):
    with pytest.raises(TypeError, match="'returns' expects a Series"):
        Bundle(returns=run["bars"])
    with pytest.raises(TypeError, match="'bars' expects a DataFrame"):
        Bundle(bars=run["close"])
    with pytest.raises(TypeError, match="'signal_fn' expects a callable"):
        Bundle(signal_fn="sma")
    with pytest.raises(TypeError, match="'rf' expects a number"):
        Bundle(rf="5%")
    with pytest.raises(TypeError, match="unsorted DatetimeIndex"):
        Bundle(close=run["close"].iloc[::-1])
    b = Bundle(close=run["close"], rf=0.05, flag="c")                # fine
    assert b.has("close", "rf") and "flag" in b and len(b) == 3
    assert "close=Series[400]" in repr(b)


def test_none_values_are_dropped_and_bundle_is_immutable(run):
    b = Bundle(returns=run["returns"], rf=None)
    assert "rf" not in b
    b2 = b.with_(rf=0.02)
    assert "rf" in b2 and "rf" not in b
    assert b2.without("rf").slots().keys() == b.slots().keys()
    with pytest.raises(AttributeError):
        b.nope


# ------------------------------------------------------------------ aliases reach guards
def test_one_returns_slot_feeds_three_guards_through_aliases(run):
    b = Bundle(returns=run["returns"], turnover=run["turnover"], rf=0.05,
               dates=run["returns"].index, underlying_returns=run["underlying"],
               position=run["position"], regime_labels=run["regime"])
    cov = b.coverage()
    assert {"cost_curve", "rf_convention", "regime_coverage"} <= set(cov.ready)
    assert b.inputs_for("regime_coverage")["strategy"] is run["returns"]   # alias resolved
    assert b.inputs_for("cost_curve")["returns"] is run["returns"]
    assert b.missing_for("survivorship_audit") == ["prices"]


def test_close_and_bars_reach_the_signal_guards(run):
    b = Bundle(close=run["close"], bars=run["bars"],
               signal_fn=lambda d: d["close"].rolling(20).mean(),
               indicator=lambda x: pd.Series(x).rolling(20).mean().to_numpy())
    cov = b.coverage(["assert_causal", "warmup_probe", "adjustment_check"])
    assert cov.ready == ["assert_causal", "warmup_probe"]
    assert cov.missing == {"adjustment_check": ["actions"]}
    assert cov.unlocks() == {"actions": ["adjustment_check"]}
    assert "one slot away" in cov.summary() and "+ actions" in cov.summary()


# ------------------------------------------------------------------ check()
def test_check_runs_ready_guards_and_reports_the_rest(run):
    b = Bundle(returns=run["returns"], turnover=run["turnover"], rf=0.05,
               bars=run["bars"], signal_fn=lambda d: d["close"].rolling(20).mean())
    report = check(b)
    assert {"cost_curve", "rf_convention", "assert_causal"} <= set(report.ran)
    assert set(report.ran).isdisjoint(report.skipped)
    assert report.skipped["survivorship_audit"] == ["prices"]
    # a causal signal passes; a peeking one fails - through the same container
    assert next(r for r in report if r.guard == "assert_causal").passed
    peek = check(b.with_(signal_fn=lambda d: d["close"].shift(-1)), guards=["assert_causal"])
    assert not peek.passed and peek[0].guard == "assert_causal"


def test_check_accepts_a_mapping_and_extra_slots(run):
    report = check({"returns": run["returns"]}, turnover=0.1, guards=["cost_curve"])
    assert report.ran == ["cost_curve"]
    report2 = check(returns=run["returns"], rf=0.05, guards=["rf_convention"])
    assert report2.ran == ["rf_convention"]


def test_rejected_inputs_are_recorded_not_raised_unless_strict(run):
    # `turnover` is an open slot (scalar or Series), so the Bundle lets a string through;
    # cost_curve refuses it. One bad input must not abort the other guards.
    b = Bundle(returns=run["returns"], turnover="lots", rf=0.05)
    report = check(b, guards=["cost_curve", "rf_convention"])
    assert report.ran == ["rf_convention"]
    assert list(report.rejected) == ["cost_curve"] and report.rejected["cost_curve"].isascii()
    assert "REJECT  cost_curve" in report.summary() and "1 rejected" in report.summary()
    with pytest.raises(TypeError):
        check(b, guards=["cost_curve"], strict=True)


def test_check_of_an_empty_bundle_runs_nothing_and_skips_everything():
    report = check(Bundle())
    assert report.ran == [] and report.passed
    assert set(report.skipped) == {c.name for c in registry() if c.required}
    assert "ran 0 guard(s)" in report.summary()


# ------------------------------------------------------------------ Suite
def test_suite_is_a_reusable_named_subset(run):
    s = Suite("cost_curve", "rf_convention")
    assert s.names == ["cost_curve", "rf_convention"] and "Suite(" in repr(s)
    b = Bundle(returns=run["returns"], turnover=run["turnover"])
    assert s.coverage(b).missing == {"rf_convention": ["rf"]}
    report = s.check(b, rf=0.01)
    assert report.ran == ["cost_curve", "rf_convention"]
    assert "cost_curve" in s.describe()
    with pytest.raises(KeyError):
        Suite("no_such_guard")
    with pytest.raises(ValueError):
        Suite()


def test_coverage_function_matches_bundle_method(run):
    b = Bundle(returns=run["returns"], turnover=run["turnover"])
    assert coverage(b).ready == b.coverage().ready
    assert coverage({"returns": run["returns"], "turnover": run["turnover"]}).ready == b.coverage().ready
