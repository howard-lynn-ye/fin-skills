"""fin_skills.core.result_manifest - a result card that refuses to render without its provenance."""
from __future__ import annotations

import json

import pytest

from fin_skills.core.result_manifest import (CostModel, DataSource, ResultCard, Split,
                                             TrialCount, Universe)


def complete_card(**overrides) -> ResultCard:
    kw = dict(
        strategy_id="core-etf-trend-v2",
        universe=Universe(source="EODHD", asof="2026-01-02", includes_delisted=True,
                          n_names=512, membership_rule="ADV>1e6 at each rebalance"),
        data=[DataSource("EODHD EOD", "2026-01-02T09:00Z", "backward-adjusted")],
        split=Split(scheme="CombinatorialPurgedCV", purge="20D", embargo="5D",
                    train="2010-2019", test="2020-2026"),
        costs=CostModel(spread_bps=2.0, slippage_bps=3.0, impact_model="sqrt(participation)",
                        borrow_bps=50.0, cash_rate_series="^IRX"),
        trials=TrialCount(n=137, ledger_path="research/trials.jsonl"),
        metrics={"sharpe_net": 0.62, "annualization": 252, "rf_convention": "annual, geometric"},
        cost_curve={0: 1.10, 5: 0.88, 10: 0.62, 20: 0.15, 50: -0.44},
        benchmark={"name": "SPY", "capm_alpha": -0.002, "alpha_t": -0.31},
        falsifier="Fails if net-of-cost excess return over SPY is <=0 across the locked window.",
        regimes_covered=["2020 crash", "2022 rate shock"],
    )
    kw.update(overrides)
    return ResultCard(**kw)


def test_complete_card_has_no_problems_and_renders():
    card = complete_card()
    assert card.problems() == []
    text = card.render()
    assert text.startswith("STRATEGY RESULT CARD - core-etf-trend-v2")
    assert "VERDICT         : unprofitable at 50 bps round-trip" in text
    assert "WARNINGS" not in text and "LLM cutoff" not in text
    assert json.loads(card.to_json())["trials"]["n"] == 137


def test_verdict_reads_where_the_strategy_dies_on_the_curve():
    assert complete_card().verdict() == "unprofitable at 50 bps round-trip"
    assert complete_card(cost_curve={0: 1.0, 10: 0.0, 20: -1.0}).verdict() == "unprofitable at 10 bps round-trip"
    assert complete_card(cost_curve={"0": 1.0, "10": 0.5, "20": 0.2}).verdict() == "still positive at 20 bps round-trip"
    assert complete_card(cost_curve={}).verdict() == "unknown"


@pytest.mark.parametrize("override, phrase", [
    (dict(universe=Universe("x", "2026-01-01", False, 5, "r")), "UPPER BOUND"),
    (dict(metrics={"sharpe_net": 1.9}), "missing 'annualization'"),
    (dict(cost_curve={0: 1.0, 10: 0.5}), "at least 3 points"),
    (dict(trials=TrialCount(1, "ledger")), "trial count <=1"),
    (dict(trials=TrialCount(9, "")), "no trial ledger path"),
    (dict(benchmark={}), "no benchmark comparison"),
    (dict(benchmark={"name": "SPY"}), "no factor-adjusted alpha"),
    (dict(regimes_covered=[]), "no regime coverage"),
    (dict(falsifier=""), "no falsifier"),
])
def test_each_missing_piece_is_named(override, phrase):
    probs = complete_card(**override).problems()
    assert any(phrase in p for p in probs), probs
    with pytest.raises(ValueError, match="REFUSING to render"):
        complete_card(**override).render()


def test_non_strict_render_lists_the_warnings_and_the_llm_cutoff():
    card = complete_card(falsifier="", llm_training_cutoff="2025-06-01")
    text = card.render(strict=False)
    assert "WARNINGS" in text and "! no falsifier stated" in text
    assert "LLM cutoff      : 2025-06-01 vs test 2020-2026" in text


def test_demo_card_is_incomplete_on_purpose(run_main):
    out = run_main("fin_skills.core.result_manifest")
    assert "problems found:" in out and "NONE STATED" in out
    assert out.count("  - ") >= 7
