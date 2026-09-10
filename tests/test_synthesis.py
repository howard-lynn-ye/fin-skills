"""fin_skills.synthesis - the availability clock, the recorded merge, and the traceable view.

Run:  python -m pytest tests/test_synthesis.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fin_skills.api as api
from fin_skills.core.result_manifest import DataSource
from fin_skills.data.schema import Adjustment
from fin_skills.synthesis import (AvailabilityError, Disagreement, Dossier, Entity, Fact,
                                  MergePolicy, MergeRefused, SourceSeries, Timeline,
                                  as_entity, check_mergeable, combine, combine_values,
                                  merge_series, price_fact, unrecorded_disagreements)

APPLE = Entity("0000320193", "cik", name="Apple Inc.")
MSFT = Entity("0000789019", "cik")


# ------------------------------------------------------------------------- fixtures
@pytest.fixture
def price() -> Fact:
    """A price known on the day it printed - the input whose clock people forget exists."""
    return price_fact("close", APPLE, 189.4, "2024-02-02", source="vendor-a")


@pytest.fixture
def filing() -> Fact:
    """A quarter that ENDED on 2023-12-30 and only became knowable 45 days later."""
    return Fact(field="revenue", entity=APPLE, value=1.19e11,
                period_start="2023-10-01", period_end="2023-12-30",
                filed_at="2024-02-01", available_at="2024-02-14",
                kind="fundamental", source="edgar")


@pytest.fixture
def macro() -> Fact:
    return Fact(field="cpi_yoy", entity=Entity("CPIAUCSL", "series_id"), value=3.1,
                period_end="2023-12-31", filed_at="2024-01-11",
                available_at="2024-01-11", kind="macro", source="fred")


def _sources(seed: int = 11, n: int = 60):
    """Two vendors on one resolved entity: A has a gap, B has a stale print."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    truth = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    a = truth.copy()
    a.iloc[7] = np.nan                       # a vendor gap: honestly absent
    b = truth.copy()
    b.iloc[20] = b.iloc[19]                  # a stale repeat: present and WRONG
    b.iloc[40] = b.iloc[40] / 10.0           # a decimal error: present and WRONG
    return (SourceSeries("A", a, Adjustment.RAW, APPLE),
            SourceSeries("B", b, Adjustment.RAW, APPLE,
                         available_lag=pd.Timedelta("1D")),
            truth)


# ================================================================ 1. availability clock
def test_a_combined_fact_inherits_the_latest_input_clock(price, filing):
    """The case where taking the EARLIEST would look fine: every column is populated on
    2024-02-02, the row has a price in it, and half of it was published twelve days later."""
    ey = combine("earnings_yield", (price, filing), 0.062)
    assert ey.available_at == pd.Timestamp("2024-02-14")        # max, not min
    assert ey.available_at == max(price.available_at, filing.available_at)
    assert ey.available_at != min(price.available_at, filing.available_at)

    # the earliest clock is the one that LOOKS fine: on 2024-02-02 the price exists, so a
    # naive join produces a complete-looking row that is not knowable for another 12 days
    naive = min(price.available_at, filing.available_at)
    assert naive == pd.Timestamp("2024-02-02")
    assert price.known_at(naive) and not filing.known_at(naive)
    assert not ey.known_at(naive) and ey.known_at("2024-02-14")


def test_an_earlier_clock_is_refused_and_a_later_one_is_allowed(price, filing):
    with pytest.raises(AvailabilityError, match="earlier than its last input"):
        combine("earnings_yield", (price, filing), 0.062, available_at="2024-02-02")
    with pytest.raises(AvailabilityError):
        combine("earnings_yield", (price, filing), 0.062,
                available_at=filing.available_at - pd.Timedelta("1s"))
    later = combine("earnings_yield", (price, filing), 0.062, available_at="2024-02-15")
    assert later.available_at == pd.Timestamp("2024-02-15")     # an execution lag is fine
    with pytest.raises(ValueError, match="at least one input"):
        combine("x", (), 1.0)


def test_the_join_date_clock_makes_the_fact_unknowable_rather_than_wrong(price, filing):
    """The other error: stamping the whole panel with the moment the research ran."""
    join_date = pd.Timestamp("2026-09-10")
    stamped = combine("earnings_yield", (price, filing), 0.062, available_at=join_date)
    sessions = pd.bdate_range("2024-01-02", periods=250)
    tl = Timeline([price, filing, stamped])
    panel = tl.panel("earnings_yield", sessions)
    assert panel.notna().to_numpy().sum() == 0        # silently nothing, not an error
    honest = Timeline([price, filing]).combine("earnings_yield", (price, filing), 0.062)
    assert honest.available_at == pd.Timestamp("2024-02-14")


def test_the_three_timestamps_are_ordered_at_construction():
    with pytest.raises(ValueError, match="before filed_at"):
        Fact(field="revenue", entity=APPLE, value=1.0, period_end="2023-12-30",
             filed_at="2024-02-01", available_at="2024-01-15", kind="fundamental")
    with pytest.raises(ValueError, match="before period_end"):
        Fact(field="revenue", entity=APPLE, value=1.0, period_end="2023-12-30",
             filed_at="2023-11-01", available_at="2023-11-01", kind="fundamental")
    # a forecast is the one kind that may be published before its period ends
    ok = Fact(field="revenue_est", entity=APPLE, value=1.0, period_end="2024-03-30",
              filed_at="2024-01-05", available_at="2024-01-05", kind="forecast")
    assert ok.kind == "forecast" and ok.lag_days < 0
    with pytest.raises(ValueError, match="kind must be one of"):
        Fact(field="x", entity=APPLE, value=1.0, period_end="2024-01-02",
             filed_at="2024-01-02", available_at="2024-01-02", kind="nonsense")


def test_as_of_answers_for_every_kind_at_once_and_picks_the_latest_vintage(price, filing,
                                                                          macro):
    restated = Fact(field="revenue", entity=APPLE, value=1.17e11,
                    period_start="2023-10-01", period_end="2023-12-30",
                    filed_at="2024-05-02", available_at="2024-05-02",
                    kind="fundamental", source="edgar")
    tl = Timeline([price, filing, macro, restated])
    assert {f.field for f in tl.as_of("2024-01-15")} == {"cpi_yoy"}
    early = tl.as_of("2024-02-20")
    assert {f.field for f in early} == {"close", "revenue", "cpi_yoy"}
    assert next(f for f in early if f.field == "revenue").value == pytest.approx(1.19e11)
    late = tl.as_of("2024-06-01")
    assert next(f for f in late if f.field == "revenue").value == pytest.approx(1.17e11)
    assert len(tl.vintages("revenue")) == 2
    assert tl.kinds == ["fundamental", "macro", "price"]


def test_panel_is_an_as_of_filter_not_a_fill(price, filing):
    sessions = pd.bdate_range("2024-01-02", periods=60)
    tl = Timeline([filing])
    wide = tl.panel("revenue", sessions)
    assert list(wide.columns) == [APPLE.key]
    known = wide[APPLE.key].dropna()
    assert known.index.min() == pd.Timestamp("2024-02-14")
    assert wide.loc[:pd.Timestamp("2024-02-13"), APPLE.key].isna().all()
    # period_end is 45 days earlier; a period_end join would have started the series there
    assert filing.lag_days == pytest.approx(46.0)


# ====================================================================== 2. merge / sources
def test_the_merge_records_the_pick_that_combine_first_resolves_silently():
    a, b, truth = _sources()
    policy = MergePolicy(precedence=("A", "B"), tol_bps=10.0)
    res = merge_series([a, b], policy)

    assert res.n_overlap == int((a.values.notna() & b.values.notna()).sum())
    assert len(res.disagreements) == 2                  # the stale print and the decimal
    assert all(isinstance(d, Disagreement) for d in res.disagreements)
    assert res.agreement_rate == pytest.approx(1.0 - 2 / res.n_overlap)
    assert res.taken_from()["B"] == 1                   # only where A had a hole

    # the naive idiom, in the order people write when B has the better coverage
    naive = b.values.combine_first(a.values)
    differs = (naive - res.values).abs() > 1e-9
    assert int(differs.sum()) == len(res.disagreements)
    for d in res.disagreements:
        assert naive.loc[d.at] != pytest.approx(res.values.loc[d.at])
        assert d.chosen == "A" and dict(d.rejected)["B"] == pytest.approx(naive.loc[d.at])
    # combine_first cannot be asked what it decided: there is no record to read
    assert not hasattr(naive, "disagreements")


def test_merge_refuses_mismatched_adjustments():
    a, b, _ = _sources()
    flipped = SourceSeries("B", b.values, Adjustment.ANCHORED_PRESENT, APPLE)
    policy = MergePolicy(precedence=("A", "B"))
    with pytest.raises(MergeRefused, match="declared Adjustment differs"):
        merge_series([a, flipped], policy)
    with pytest.raises(MergeRefused, match="readjust"):
        check_mergeable([a, flipped], policy)
    # the escape is an explicit re-adjustment, not a flag: same convention, merge runs
    assert merge_series([a, SourceSeries("B", b.values, a.adjustment, APPLE)],
                        policy).values.notna().all()


def test_merge_refuses_an_unresolved_or_mismatched_symbology():
    a, b, _ = _sources()
    policy = MergePolicy(precedence=("A", "B"))
    ticker_a = SourceSeries("A", a.values, Adjustment.RAW, "AAPL")
    ticker_b = SourceSeries("B", b.values, Adjustment.RAW, "AAPL")
    assert as_entity("AAPL").scheme == "ticker" and not as_entity("AAPL").is_resolved
    with pytest.raises(MergeRefused, match="symbology is unresolved"):
        merge_series([ticker_a, ticker_b], policy)
    # the same two series, resolved to a permanent id, merge
    assert merge_series([a, b], policy).values.notna().any()
    # two different issuers are a boundary, not a disagreement
    other = SourceSeries("B", b.values, Adjustment.RAW, MSFT)
    with pytest.raises(MergeRefused, match="different entities"):
        merge_series([a, other], policy)
    relaxed = MergePolicy(precedence=("A", "B"), require_resolved_entity=False)
    assert merge_series([ticker_a, ticker_b], relaxed).values.notna().any()


def test_merge_refuses_an_unranked_source_and_a_bad_policy():
    a, b, _ = _sources()
    with pytest.raises(MergeRefused, match="not in the declared precedence"):
        merge_series([a, b], MergePolicy(precedence=("A",)))
    with pytest.raises(MergeRefused, match="at least two sources"):
        merge_series([a], MergePolicy(precedence=("A",)))
    with pytest.raises(ValueError, match="non-empty precedence"):
        MergePolicy(precedence=())
    with pytest.raises(ValueError, match="on_disagreement"):
        MergePolicy(precedence=("A",), on_disagreement="whatever")


def test_on_disagreement_raise_and_drop():
    a, b, _ = _sources()
    with pytest.raises(MergeRefused, match="disagree by"):
        merge_series([a, b], MergePolicy(("A", "B"), on_disagreement="raise"))
    dropped = merge_series([a, b], MergePolicy(("A", "B"), on_disagreement="drop"))
    assert dropped.n_dropped == 2
    assert int(dropped.values.isna().sum()) == 2


def test_merged_facts_carry_the_sources_and_the_slower_clock():
    a, b, _ = _sources()
    res = merge_series([a, b], MergePolicy(("A", "B")))
    facts = res.to_facts()
    both = [f for f in facts if len(f.inputs) == 2]
    assert both, "overlapping timestamps must carry both vendors as inputs"
    f = both[0]
    assert f.available_at == max(i.available_at for i in f.inputs)
    assert f.available_at == f.period_end + pd.Timedelta("1D")   # B is the slow one
    assert {i.source for i in f.inputs} == {"A", "B"}
    flagged = [x for x in facts if x.disagreements]
    assert len(flagged) == len(res.disagreements)
    assert not unrecorded_disagreements(facts, tol_bps=10.0)


# ================================================================== 3. dossier / explain
def _clean_dossier():
    a, b, _ = _sources()
    res = merge_series([a, b], MergePolicy(("A", "B"), tol_bps=10.0))
    tl = res.to_timeline()
    last_price = tl.latest("close", "2024-04-01", entity=APPLE)
    filing = Fact(field="revenue", entity=APPLE, value=1.19e11, period_start="2023-10-01",
                  period_end="2023-12-30", filed_at="2024-02-01",
                  available_at="2024-02-14", kind="fundamental", source="edgar")
    macro = Fact(field="cpi_yoy", entity=Entity("CPIAUCSL", "series_id"), value=3.1,
                 period_end="2023-12-31", filed_at="2024-01-11",
                 available_at="2024-01-11", kind="macro", source="fred")
    doc = Fact(field="risk_factor", entity=APPLE, value="supply concentration",
               period_end="2023-09-30", filed_at="2023-11-03",
               available_at="2023-11-03", kind="document", source="edgar-10k")
    tl.extend([filing, macro, doc])
    shares = Fact(field="shares", entity=APPLE, value=1.55e10, period_start="2023-10-01",
                  period_end="2023-12-30", filed_at="2024-02-01",
                  available_at="2024-02-14", kind="fundamental", source="edgar")
    tl.add(shares)
    rps = tl.combine("revenue_per_share", (filing, shares), 1.19e11 / 1.55e10,
                     kind="derived")
    tl.combine("ps_ratio", (last_price, rps, macro), 12.3, kind="derived")
    return Dossier(entity=APPLE, as_of="2024-04-01", timeline=tl, title="APPLE DOSSIER")


def test_explain_names_every_input_including_the_transitive_ones():
    d = _clean_dossier()
    ex = d.explain("ps_ratio")
    named = {f"{f.field}@{f.entity.key}" for f in ex.inputs}
    # direct inputs
    assert {"close@" + APPLE.key, "revenue_per_share@" + APPLE.key,
            "cpi_yoy@series_id:CPIAUCSL"} <= named
    # transitive: revenue and shares sit UNDER revenue_per_share, and under the price the
    # two vendor prints that were merged
    assert {"revenue@" + APPLE.key, "shares@" + APPLE.key} <= named
    assert sum(1 for f in ex.inputs if f.source in ("A", "B")) == 2
    assert ex.sources() == ["A", "B", "edgar", "fred"]
    assert ex.complete and not ex.problems
    assert all(v for v in ex.vintages().values())
    assert len(ex.steps) == len(ex.inputs) + 1          # every input, plus the answer
    text = ex.render()
    assert text.isascii() and "revenue_per_share" in text and "cpi_yoy" in text
    binding = ex.binding_input()
    assert binding is not None and binding.available_at == d.get("ps_ratio").available_at
    with pytest.raises(KeyError):
        d.explain("no_such_field")


def test_the_dossier_renders_and_produces_a_result_card_provenance_block():
    d = _clean_dossier()
    text = d.render()
    assert text.isascii()
    assert "APPLE DOSSIER" in text and "as of 2024-04-01" in text
    for name in ("close", "revenue", "risk_factor", "ps_ratio", "cpi_yoy@series_id"):
        assert name[:18] in text
    block = d.provenance_block()
    assert block and all(isinstance(s, DataSource) for s in block)
    names = {s.name for s in block}
    assert {"A", "B", "edgar", "fred", "edgar-10k"} <= names
    assert all(s.retrieved_at and s.adjustment for s in block)
    assert d.data_sources() == block
    assert "cpi_yoy@series_id:CPIAUCSL" in d.fields()      # context keeps its entity
    assert d.value("revenue") == pytest.approx(1.19e11)


def test_the_dossier_hands_the_whole_thing_to_check():
    d = _clean_dossier()
    bundle = d.to_bundle()
    assert bundle.has("dossier", "timeline", "as_of")
    report = api.check(bundle)
    assert "synthesis_integrity" in report.ran
    assert next(r for r in report if r.guard == "synthesis_integrity").passed
    # extra slots reach the other guards through the same call
    closes = d.timeline.panel("close", pd.bdate_range("2024-01-02", periods=60))
    wide = closes.rename(columns={APPLE.key: "AAPL"})
    assert api.check(d.to_bundle(prices=wide), guards=["survivorship_audit"]).ran


# ============================================================================= 4. the guard
def test_the_guard_passes_on_a_clean_dossier():
    d = _clean_dossier()
    r = api.get("synthesis_integrity").run(timeline=d.timeline, dossier=d)
    assert r.passed and r.errors == []
    assert r.evidence["n_availability_violations"] == 0
    assert r.evidence["n_silent_disagreements"] == 0
    assert r.evidence["n_incomplete_fields"] == 0
    assert r.summary().startswith("PASS") and r.summary().isascii()
    # a Dossier alone is accepted in the timeline slot
    assert api.get("synthesis_integrity").run(timeline=d).passed


def test_the_guard_fails_a_backdated_combined_fact(price, filing):
    backdated = Fact(field="earnings_yield", entity=APPLE, value=0.062,
                     period_end="2024-02-02", filed_at="2024-02-02",
                     available_at="2024-02-02", kind="derived", source="combined",
                     inputs=(price, filing))
    r = api.get("synthesis_integrity").run(timeline=Timeline([price, filing, backdated]))
    assert not r.passed
    assert any("LAST input" in f.message for f in r.errors)
    assert r.evidence["n_availability_violations"] == 1


def test_the_guard_fails_a_merge_that_resolved_silently():
    a, b, _ = _sources()
    res = merge_series([a, b], MergePolicy(("A", "B"), tol_bps=10.0))
    facts = res.to_facts()
    stripped = [Fact(field=f.field, entity=f.entity, value=f.value,
                     period_end=f.period_end, filed_at=f.filed_at,
                     available_at=f.available_at, kind=f.kind, source=f.source,
                     inputs=f.inputs) for f in facts]        # the record thrown away
    r = api.get("synthesis_integrity").run(timeline=Timeline(stripped), tol_bps=10.0)
    assert not r.passed
    assert r.evidence["n_silent_disagreements"] == len(res.disagreements) == 2
    assert any("no Disagreement recorded" in f.message for f in r.errors)
    # raising the threshold above the spread is a DECLARED tolerance, not a silent pick
    loose = api.get("synthesis_integrity").run(timeline=Timeline(stripped), tol_bps=1e6)
    assert loose.passed


def test_the_guard_fails_an_incomplete_provenance_chain(price, filing):
    anonymous = Fact(field="revenue", entity=APPLE, value=1.19e11,
                     period_end="2023-12-30", filed_at="2024-02-01",
                     available_at="2024-02-14", kind="fundamental")     # no source
    derived = combine("earnings_yield", (price, anonymous), 0.062)
    tl = Timeline([price, anonymous, derived])
    d = Dossier(entity=APPLE, as_of="2024-03-01", timeline=tl)
    r = api.get("synthesis_integrity").run(timeline=tl, dossier=d)
    assert not r.passed
    assert r.evidence["n_rootless_facts"] == 1
    assert r.evidence["n_incomplete_fields"] >= 1
    assert any("names no source" in f.message for f in r.errors)
    assert not d.explain("earnings_yield").complete


def test_the_guard_rejects_the_wrong_kind_of_input():
    d = _clean_dossier()
    with pytest.raises(TypeError, match="must be a fin_skills.synthesis.Timeline"):
        api.get("synthesis_integrity").run(timeline=pd.Series([1.0]))
    with pytest.raises(TypeError, match="pass the Dossier as"):
        api.get("synthesis_integrity").run(timeline=d, dossier=d)
    with pytest.raises(TypeError, match="must be a fin_skills.synthesis.Dossier"):
        api.get("synthesis_integrity").run(timeline=d.timeline, dossier="a dossier")


# ============================================================================ determinism
def test_everything_is_deterministic():
    one, two = _clean_dossier(), _clean_dossier()
    assert one.render() == two.render()
    assert one.explain("ps_ratio").render() == two.explain("ps_ratio").render()
    assert [(s.name, s.adjustment) for s in one.provenance_block()] == \
           [(s.name, s.adjustment) for s in two.provenance_block()]
    a, b, _ = _sources()
    r1 = merge_series([a, b], MergePolicy(("A", "B")))
    r2 = merge_series([b, a], MergePolicy(("A", "B")))       # argument order must not matter
    pd.testing.assert_series_equal(r1.values, r2.values)
    pd.testing.assert_series_equal(r1.chosen, r2.chosen)
    assert [str(d) for d in r1.disagreements] == [str(d) for d in r2.disagreements]
    assert list(one.fields()) == sorted(one.fields())


def test_combine_values_and_entity_helpers():
    p = price_fact("close", APPLE, 200.0, "2024-02-02", source="v")
    f = Fact(field="eps", entity=APPLE, value=5.0, period_end="2023-12-30",
             filed_at="2024-02-01", available_at="2024-02-14", kind="fundamental",
             source="edgar")
    pe = combine_values("pe", (p, f), lambda px, eps: px / eps, kind="derived")
    assert pe.value == pytest.approx(40.0)
    assert pe.available_at == pd.Timestamp("2024-02-14")
    assert as_entity(APPLE) is APPLE
    assert as_entity("cik:0000320193") == Entity("0000320193", "cik")
    assert as_entity("AAPL") == Entity("AAPL", "ticker")
    assert str(APPLE) == "cik:0000320193"
    with pytest.raises(ValueError, match="non-empty entity_id"):
        Entity("")
    with pytest.raises(TypeError, match="must be Facts"):
        combine("x", (p, "not a fact"), 1.0)
    with pytest.raises(TypeError, match="holds Facts"):
        Timeline().add("nope")
