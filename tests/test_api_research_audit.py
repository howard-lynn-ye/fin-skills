"""fin_skills.api.guards.research_audit - the staged reality check over a Bundle.

The property under test is the ORDER: the audit stops at the first stage that fails, and
the stages after it do not run. Deflating a Sharpe computed on leaked data is meaningless,
so an audit that reported a PBO for a bundle whose signal peeks would be worse than one
that reported nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fin_skills.api as api
from fin_skills.api import Bundle
from fin_skills.api.guards.research_audit import (STAGE_NAMES, STAGES, AuditReport,
                                                  deflation_outcome, pbo_outcome,
                                                  reality_check)

T = 1008
PPY = 252


def _run(seed: int = 11, true_sr: float = 3.0, n_cfg: int = 30, cost_bps: float = 2.0) -> dict:
    """One research run as a dict of Bundle slots: bars, a causal signal, a panel of
    `n_cfg` configurations of which column 0 carries `true_sr`, and the winner's returns.

    `returns` is the BEST column by in-sample Sharpe, not column 0 - that is what a
    research run actually reports, and with true_sr = 0 it gives the audit the realistic
    case: a winner with a healthy in-sample Sharpe and nothing behind it.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=T)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, T))), index=idx, name="close")
    bars = pd.DataFrame({"open": close.shift(1).bfill(), "high": close * 1.01,
                         "low": close * 0.99, "close": close,
                         "volume": rng.integers(1000, 5000, T)}, index=idx)
    panel = rng.normal(0.0, 0.01, (T, n_cfg))
    panel[:, 0] += true_sr * 0.01 / np.sqrt(PPY)
    frame = pd.DataFrame(panel, index=idx, columns=[f"cfg{i}" for i in range(n_cfg)])
    sharpes = frame.mean() / frame.std(ddof=1)
    labels = pd.Series(np.tile(np.repeat(["calm", "stress"], 126), 4), index=idx)
    top = str(sharpes.idxmax())
    return dict(returns=frame[top], turnover=pd.Series(0.02, index=idx), cost_bps=cost_bps,
                bars=bars, signal_fn=lambda d: d["close"].rolling(20).mean(),
                model_returns=frame, sharpes=list(map(float, sharpes.values)),
                best_sharpe=float(sharpes[top]), n_obs=T, periods_per_year=PPY,
                n_blocks=10, benchmark_returns=pd.Series(0.0, index=idx), seed=3, reps=200,
                dates=idx, underlying_returns=close.pct_change().fillna(0.0),
                position=pd.Series(1.0, index=idx), regime_labels=labels, how="ex-ante")


@pytest.fixture(scope="module")
def good() -> Bundle:
    return Bundle(**_run())


@pytest.fixture(scope="module")
def noise() -> Bundle:
    return Bundle(**_run(seed=7, true_sr=0.0))


# ------------------------------------------------------------------ the stage table
def test_the_stages_are_the_sceptics_order_and_every_guard_name_resolves():
    assert STAGE_NAMES == ("integrity", "cost", "deflation", "overfitting", "regime")
    for stage in STAGES:
        assert stage.question and stage.question.isascii()
        for name in stage.guards:
            assert api.get(name).name == name
    natives = [s.native for s in STAGES if s.native]
    assert natives == ["deflation", "pbo"]
    # result_manifest is deliberately not a gate the audit clears for you
    assert "result_manifest" not in {g for s in STAGES for g in s.guards}


def test_the_guard_is_registered_with_the_bundle_vocabulary():
    g = api.get("research_audit")
    assert g.skill == "backtest-overfitting" and g.summary
    assert set(g.required) == {"best_sharpe", "n_obs"}
    assert "n_trials" in g.optional and "model_returns" in g.optional
    assert "n_trials" in api.vocabulary()
    assert g.missing({"best_sharpe": 1.0, "n_obs": 100}) == ["n_trials or ledger or sharpes"]
    assert g.missing({"best_sharpe": 1.0, "n_obs": 100, "n_trials": 5}) == []


# ------------------------------------------------------------------ the guard itself
def test_the_guard_passes_a_real_edge_and_fails_pure_noise(good, noise):
    g = api.get("research_audit")
    ok = g.run(**good.inputs_for(g))
    assert ok.passed, ok.summary()
    assert ok.evidence["pbo"] < ok.evidence["no_skill_line"]
    assert ok.evidence["minbtl"]["credible"] and ok.evidence["haircut"]["survives"]
    assert ok.summary().isascii()

    bad = g.run(**noise.inputs_for(g))
    assert not bad.passed
    reasons = " ".join(f.message for f in bad.errors)
    assert "minimum backtest length" in reasons.lower() or "adjusted for" in reasons


def test_without_a_panel_the_guard_says_pbo_was_not_computed(good):
    g = api.get("research_audit")
    kw = good.without("model_returns").inputs_for(g)
    r = g.run(**kw)
    assert "PBO was not computed" in " ".join(f.message for f in r.warnings)
    assert "pbo" not in r.evidence


def test_the_trial_count_comes_from_the_ledger_when_one_is_given(tmp_path):
    from fin_skills.core.trial_ledger import TrialLedger

    led = TrialLedger(tmp_path / "trials.jsonl")
    ids = [led.record("grid", {"i": i}) for i in range(300)]
    led.complete(ids[0], {"sharpe": 0.15, "n_obs": T})
    for i in range(1, 300):
        led.abandon(ids[i], "shelved")
    out = deflation_outcome(0.15, T, PPY, ledger=led)
    assert out.evidence["n_trials"] == 300          # registered, not the one completed
    assert "registered count" in out.evidence["trial_count_source"]
    # declaring a smaller count than the ledger holds is corrected, with a warning
    out2 = deflation_outcome(0.15, T, PPY, n_trials=5, ledger=led)
    assert out2.evidence["n_trials"] == 300
    assert any("below the 300 trials" in f.message for f in out2.findings)


def test_deflation_needs_a_trial_count_from_somewhere():
    with pytest.raises(ValueError, match="needs a trial count"):
        deflation_outcome(0.1, T, PPY)
    with pytest.raises(ValueError, match="correction must be"):
        deflation_outcome(0.1, T, PPY, n_trials=10, correction="sidak")


def test_pbo_outcome_warns_about_correlated_configurations_and_thin_panels():
    from fin_skills.core.overfitting import crossover_grid

    grid = crossover_grid(np.random.default_rng(4000), T)
    out = pbo_outcome(grid, n_blocks=10)
    assert out.evidence["mean_abs_correlation"] > 0.5
    assert any("effective trials" in f.message for f in out.findings)

    rng = np.random.default_rng(1)
    tiny = rng.normal(0, 0.01, (T, 3))
    assert any("only 3 configurations" in f.message for f in pbo_outcome(tiny, 10).findings)


# ------------------------------------------------------------------ THE assembly
def test_the_audit_clears_every_stage_on_a_bundle_with_a_real_edge(good):
    r = reality_check(good)
    assert isinstance(r, AuditReport)
    assert r.passed and r.stopped_at is None, r.summary()
    assert [s.name for s in r.stages] == list(STAGE_NAMES)
    assert all(s.status == "pass" for s in r.stages), [(s.name, s.status) for s in r.stages]
    assert r.unevaluated == []
    text = r.summary()
    assert text.isascii() and "PASS" in text
    by = r.by_name
    assert "assert_causal" in by["integrity"].ran and "cost_curve" in by["cost"].ran
    assert "trial_ledger" in by["deflation"].ran
    assert by["overfitting"].evidence["pbo"] < 0.5
    assert "regime_coverage" in by["regime"].ran


def test_a_leaky_bundle_stops_at_integrity_and_nothing_after_it_runs(good):
    leaky = good.with_(signal_fn=lambda d: d["close"].shift(-1))
    r = reality_check(leaky)
    assert r.stopped_at == "integrity" and not r.passed
    assert "LOOK-AHEAD" in r.reason
    by = r.by_name
    assert by["integrity"].status == "fail"
    for later in ("cost", "deflation", "overfitting", "regime"):
        assert by[later].status == "not reached"
        assert by[later].results == [] and by[later].findings == []
    assert "STOPPED AT STAGE 'integrity'" in r.summary()
    assert "measures nothing" in r.summary()


def test_a_bundle_that_dies_on_costs_never_reaches_the_deflation_arithmetic(good):
    # the same real edge, traded 100x as often at 20 bps round-trip
    r = reality_check(good.with_(turnover=pd.Series(2.0, index=good.returns.index),
                                 cost_bps=20.0))
    assert r.stopped_at == "cost", r.summary()
    assert r.by_name["integrity"].status == "pass"
    assert r.by_name["deflation"].status == "not reached"
    assert "breakeven" in r.reason


def test_pure_noise_survives_integrity_and_costs_and_dies_at_deflation(noise):
    r = reality_check(noise.with_(cost_bps=0.5))
    assert r.by_name["integrity"].status == "pass"
    assert r.stopped_at == "deflation", r.summary()
    assert r.by_name["overfitting"].status == "not reached"


def test_pbo_is_reached_when_deflation_is_skipped_and_still_catches_the_search(noise):
    """Run the two later stages alone: a noise panel whose Sharpe was NOT deflated still
    fails PBO, which is the check that does not need a trial count at all."""
    r = reality_check(noise, stages=["overfitting", "regime"])
    assert [s.name for s in r.stages] == ["overfitting", "regime"]
    assert r.stopped_at == "overfitting"
    assert "coin flip" in r.reason
    with pytest.raises(ValueError, match="unknown stage"):
        reality_check(noise, stages=["integrity", "nope"])


def test_an_empty_bundle_evaluates_nothing_and_says_so():
    r = reality_check(Bundle())
    assert r.passed and r.unevaluated == list(STAGE_NAMES)
    assert all(s.status == "not evaluated" for s in r.stages)
    assert "not evaluated:" in r.summary()
    strict = reality_check(Bundle(), on_unknown="stop")
    assert strict.stopped_at == "integrity" and not strict.passed
    assert "treats unknown as failed" in strict.reason
    with pytest.raises(ValueError, match="on_unknown"):
        reality_check(Bundle(), on_unknown="ignore")


def test_reality_check_accepts_a_mapping_and_loose_slots(good):
    r = reality_check(dict(good.slots()))
    assert r.passed
    r2 = reality_check(good.without("cost_bps"), cost_bps=2.0)
    assert r2.passed
    assert reality_check(None, **good.slots()).passed


def test_a_stage_reports_the_guards_it_had_no_input_for(good):
    r = reality_check(good)
    integ = r.by_name["integrity"]
    assert "assert_causal" in integ.ran
    assert "survivorship_audit" in integ.unavailable        # no `prices` slot
    assert "guard(s) had no input" in r.summary()
    # pit_universe declares no required inputs; it must not count as "evaluated" on a
    # bundle that supplies it nothing.
    assert "pit_universe" in integ.unavailable


def test_a_seeded_audit_is_deterministic():
    a = reality_check(Bundle(**_run()))
    b = reality_check(Bundle(**_run()))
    assert a.stopped_at == b.stopped_at
    assert (a.by_name["overfitting"].evidence["pbo"]
            == b.by_name["overfitting"].evidence["pbo"])
    assert a.summary() == b.summary()
