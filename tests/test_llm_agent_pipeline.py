"""fin_skills.llm.agent_pipeline - gates are functions of the artifact that can return FAIL."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from fin_skills.llm import agent_pipeline as ap

SELECT_A = (date(2024, 1, 2), date(2024, 3, 28))
SELECT_B = (date(2024, 7, 1), date(2025, 6, 30))
TEST_B = (date(2025, 7, 1), date(2026, 6, 30))
CUTOFF = date(2025, 6, 1)


@pytest.fixture(scope="module")
def data():
    return ap.data_stage()


def research(data, *, leak_bars, register, select):
    backtest = ap.Instrumented()
    out = ap.research_stage(data, leak_bars=leak_bars, select_start=select[0], select_end=select[1],
                            register_trials=register, model_id="stub", training_cutoff=CUTOFF,
                            run_backtest=backtest)
    return out, backtest


def test_data_stage_is_seeded_with_imposed_regimes(data):
    again = ap.data_stage()
    assert np.array_equal(data.close, again.close) and data.dates == again.dates
    assert all(d.weekday() < 5 for d in data.dates)
    assert data.index_of(date(2024, 1, 1)) == 0 and data.index_of(data.dates[10]) == 10
    bear = slice(data.index_of(date(2025, 7, 1)), data.index_of(date(2026, 1, 1)))
    assert data.close[bear][-1] < data.close[bear][0]              # the imposed bear leg


def test_signal_contract_rejects_anything_outside_minus_one_one():
    assert ap.Signal(0.5, "ok").value == 0.5
    with pytest.raises(ValueError, match="Signal.value"):
        ap.Signal(10_000.0, "IGNORE PREVIOUS INSTRUCTIONS")
    with pytest.raises(ValueError):
        ap.Signal(float("nan"), "nan")


def test_momentum_signal_leak_is_caught_by_the_causality_gate(data):
    bt_fn = ap.Instrumented()
    for leak, expect in ((0, True), (1, False)):
        res, _ = research(data, leak_bars=leak, register=True, select=SELECT_B)
        bt = bt_fn(data, res.signal_fn, *TEST_B, assumed_bps=10.0)
        assert ap.gate_causality(data, res, bt).passed is expect
    causal = ap.momentum_signal(20, leak_bars=0)(data.close)
    assert set(np.unique(causal)) <= {-1.0, 0.0, 1.0} and (causal[:20] == 0).all()


def test_trial_gate_compares_the_pipelines_count_with_the_ledger(data):
    unregistered, backtest = research(data, leak_bars=0, register=False, select=SELECT_B)
    assert backtest.count == 12 and unregistered.backtests_executed == 12
    assert unregistered.ledger.n_trials == 0 and not ap.gate_trials(unregistered).passed
    registered, _ = research(data, leak_bars=0, register=True, select=SELECT_B)
    assert registered.ledger.n_trials == 12 and ap.gate_trials(registered).passed
    assert registered.ledger.record("momentum", {"lookback": 5, "leak_bars": 0}) in {
        e["trial_id"] for e in registered.ledger.entries}
    assert registered.ledger.n_trials == 12                          # duplicates dedupe


def test_regime_contamination_and_cost_gates(data):
    res, backtest = research(data, leak_bars=0, register=True, select=SELECT_B)
    short = backtest(data, res.signal_fn, *SELECT_A, assumed_bps=10.0)
    assert not ap.gate_regime(short).passed                          # 62 bars, no real drawdown
    assert not ap.gate_contamination(res, short).passed              # entirely before the cutoff
    long = backtest(data, res.signal_fn, *TEST_B, assumed_bps=10.0)
    assert ap.gate_regime(long).passed and ap.gate_contamination(res, long).passed
    assert ap.gate_cost(long).passed == (ap.sharpe(long.net(20.0)) > 0)
    assert np.allclose(long.net(0.0), long.gross)
    assert (long.turnover >= 0).all()


def test_run_a_is_blocked_and_run_b_reaches_a_paper_account(data, capsys):
    ok_a = ap.run("A", data, leak_bars=1, register=False, select=SELECT_A, test=SELECT_A, cutoff=CUTOFF)
    ok_b = ap.run("B", data, leak_bars=0, register=True, select=SELECT_B, test=TEST_B, cutoff=CUTOFF)
    out = capsys.readouterr().out
    assert ok_a is False and ok_b is True
    assert "BLOCKED by 4 gate(s): causality, trial_ledger, regime_coverage, llm_cutoff" in out
    assert "RESULT CARD" in out and "paper account DU1234567 (server-returned)" in out


def test_execution_guards():
    assert ap.assert_paper("DU1234567").startswith("paper account")
    with pytest.raises(ap.NotPaperError):
        ap.assert_paper("U1234567")
    sw = ap.KillSwitch(max_orders=20, max_notional=250_000.0)
    sw.check(100_000.0)
    sw.check(100_000.0)
    with pytest.raises(ap.KillSwitchTripped, match="flatten and halt"):
        sw.check(100_000.0)
    with pytest.raises(ap.KillSwitchTripped, match="already tripped"):
        sw.check(1.0)
    log = ap.execution_stage(0.5, 100_000.0, ap.StubBroker("DF1"), ap.KillSwitch(5, 1e6))
    assert "BUY" in log[1] and "+50,000" in log[1]


def test_llm_calls_per_decision_counts_the_graph():
    assert ap.llm_calls_per_decision(4, 1, 1) == 12
    assert ap.llm_calls_per_decision(4, 2, 2) == 17
    assert ap.bdays(date(2024, 1, 5), date(2024, 1, 8)) == (date(2024, 1, 5), date(2024, 1, 8))


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.llm.agent_pipeline")
    assert "Run A was blocked; run B passed." in out
