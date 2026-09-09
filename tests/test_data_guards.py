"""The four DATA gates: green on a clean synthetic dataset, red on a corrupted one.

survivorship_audit, adjustment_check, pit_fundamentals and reconcile_sources need no
backtest, no strategy and no engine - only what an adapter returns. That is the point of
`to_bundle()`: these five gates should fail before a strategy is ever written, not after a
Sharpe has been computed and believed.
"""
from __future__ import annotations

import pandas as pd
import pytest

from _data_fixtures import (RESTATEMENT_AS_OF, StubAdapter, bad_tick_vintage, clean_bars,
                            clean_fundamentals, mislabelled_bars, rewritten_vintage,
                            split_ticker, survivor_only_bars)
from fin_skills.api import Suite, check
from fin_skills.data import Cache, guard_convention, pit_used, to_bundle

DATA_GATES = ("survivorship_audit", "adjustment_check", "pit_fundamentals",
              "reconcile_sources")


@pytest.fixture(scope="module")
def good():
    return clean_bars(n_names=60, years=8, n_dead=20)


@pytest.fixture(scope="module")
def facts():
    return clean_fundamentals()


def _one(report, name):
    hits = [r for r in report if r.guard == name]
    assert len(hits) == 1, f"{name} did not run: {report.summary()}"
    return hits[0]


# ---------------------------------------------------------------------------- green
def test_all_four_gates_are_reachable_from_data_alone(good, facts, tmp_path):
    """One coverage statement: with bars, fundamentals and a second vintage, exactly these
    four guards are ready with no backtest artefacts at all."""
    t = split_ticker(good)
    cache = Cache(tmp_path / "c")
    key = cache.put(good, good.provenance)
    div = cache.refetch(key, StubAdapter(rewritten_vintage(good, t)))

    bundle = to_bundle(good, fundamentals=facts, ticker=t, as_of=RESTATEMENT_AS_OF)
    bundle = bundle.with_(other=div.new_values,
                          used=pit_used(facts, RESTATEMENT_AS_OF, tag="Revenues"),
                          expected=guard_convention(good.adjustment))
    ready = set(bundle.coverage(DATA_GATES).ready)
    assert ready == set(DATA_GATES), bundle.coverage(DATA_GATES).summary()

    report = Suite(*DATA_GATES).check(bundle)
    assert len(report) == 4
    assert all(r.passed for r in report), report.summary()


def test_survivorship_audit_is_green_on_a_panel_where_names_actually_die(good):
    r = _one(check(to_bundle(good), guards=["survivorship_audit"]), "survivorship_audit")
    assert r.passed
    assert r.evidence["verdict"].startswith("PLAUSIBLY BIAS-FREE")
    assert r.evidence["n_ended_early"] == 20
    assert r.evidence["n_missing_delisted"] == 0


def test_adjustment_check_is_green_when_the_declaration_matches_the_data(good):
    t = split_ticker(good)
    bundle = to_bundle(good, ticker=t, expected=guard_convention(good.adjustment))
    r = _one(check(bundle, guards=["adjustment_check"]), "adjustment_check")
    assert r.passed
    assert r.evidence["convention"] == guard_convention(good.adjustment)


def test_pit_fundamentals_is_green_on_the_vintage_that_was_on_file(facts):
    used = pit_used(facts, RESTATEMENT_AS_OF, tag="Revenues")
    bundle = to_bundle(fundamentals=facts, as_of=RESTATEMENT_AS_OF).with_(used=used)
    r = _one(check(bundle, guards=["pit_fundamentals"]), "pit_fundamentals")
    assert r.passed
    assert r.evidence["leaky_periods"], "the fixture must contain a restatement to avoid"


def test_reconcile_sources_is_green_on_a_pure_convention_difference(good, tmp_path):
    t = split_ticker(good)
    cache = Cache(tmp_path / "c")
    key = cache.put(good, good.provenance)
    div = cache.refetch(key, StubAdapter(rewritten_vintage(good, t)))
    r = _one(check(div.to_bundle(), guards=["reconcile_sources"]), "reconcile_sources")
    assert r.passed
    assert r.evidence["verdict"].startswith("ADJUSTMENT CONVENTION ONLY")


# ------------------------------------------------------------------------------ red
def test_survivorship_audit_is_red_when_the_delisted_names_are_gone():
    bad = survivor_only_bars(n_names=60, years=8, n_dead=20)
    r = _one(check(to_bundle(bad), guards=["survivorship_audit"]), "survivorship_audit")
    assert not r.passed
    assert r.evidence["verdict"].startswith("SURVIVOR-BIASED (confirmed)")
    assert r.evidence["n_missing_delisted"] == 20
    assert len(r.evidence["missing_delisted"]) == 20
    assert not r.evidence["inflation"]["measurable"], \
        "nothing dies inside this panel, so the damage cannot even be priced from it"


def test_adjustment_check_is_red_on_a_raw_series_wearing_an_adjusted_label():
    bad = mislabelled_bars(n_names=4, years=8, n_dead=0)
    t = split_ticker(bad)
    bundle = to_bundle(bad, ticker=t, expected=guard_convention(bad.adjustment))
    r = _one(check(bundle, guards=["adjustment_check"]), "adjustment_check")
    assert not r.passed
    assert r.evidence["convention"] == "raw"
    assert any("wrong scale" in f.message for f in r.errors)


def test_pit_fundamentals_is_red_on_drop_duplicates_keep_last(facts):
    naive = pit_used(facts, RESTATEMENT_AS_OF, tag="Revenues", naive=True)
    bundle = to_bundle(fundamentals=facts, as_of=RESTATEMENT_AS_OF).with_(used=naive)
    r = _one(check(bundle, guards=["pit_fundamentals"]), "pit_fundamentals")
    assert not r.passed
    messages = "\n".join(f.message for f in r.errors)
    assert "is the vintage filed 2023-02-14, after as_of 2022-12-31" in messages
    assert "was not on file at 2022-12-31" in messages


def test_reconcile_sources_is_red_on_a_step_with_no_corporate_action(good, tmp_path):
    t = split_ticker(good)
    cache = Cache(tmp_path / "c")
    key = cache.put(good, good.provenance)
    div = cache.refetch(key, StubAdapter(bad_tick_vintage(good, t)))
    r = _one(check(div.to_bundle(), guards=["reconcile_sources"]), "reconcile_sources")
    assert not r.passed
    assert r.evidence["verdict"].startswith("DATA ERROR")


def test_the_red_dataset_fails_every_gate_it_touches(tmp_path):
    """One corrupted dataset, four failures - each attributable to a different break."""
    bad = survivor_only_bars(n_names=60, years=8, n_dead=20)  # the dead names are gone
    raw_label = mislabelled_bars(n_names=4, years=8, n_dead=0)  # raw wearing a BACK label
    t = split_ticker(raw_label)
    facts = clean_fundamentals()

    cache = Cache(tmp_path / "c")
    key = cache.put(raw_label, raw_label.provenance)
    div = cache.refetch(key, StubAdapter(bad_tick_vintage(raw_label, t)))

    bundle = to_bundle(raw_label, fundamentals=facts, ticker=t,
                       as_of=RESTATEMENT_AS_OF).with_(
        prices=to_bundle(bad).prices, listings=to_bundle(bad).listings,
        other=div.new_values, expected=guard_convention(raw_label.adjustment),
        used=pit_used(facts, RESTATEMENT_AS_OF, tag="Revenues", naive=True))

    report = Suite(*DATA_GATES).check(bundle)
    assert len(report) == 4
    assert not any(r.passed for r in report), report.summary()
    assert {r.guard for r in report} == set(DATA_GATES)


# ------------------------------------------------------------------- the manifest row
def test_a_provenance_becomes_the_result_card_data_source(good):
    ds = good.as_data_source()
    assert ds.name == "yfinance"
    assert ds.adjustment == "back"
    assert ds.retrieved_at.endswith("+00:00")
    assert clean_fundamentals().as_data_source().adjustment == "PIT, available_at<=as_of"


def test_validate_reports_findings_rather_than_raising(good):
    from fin_skills.data.validate import summary

    clean = good.validate()
    assert not [f for f in clean if f.severity == "error"]
    assert summary(clean).startswith("OK")

    survivor_only = survivor_only_bars(n_names=20, years=8, n_dead=6).validate()
    assert any(f.severity == "warning" and "UPPER BOUND" in f.message
               for f in survivor_only)
    assert all(f.severity in ("error", "warning", "info") for f in survivor_only)
