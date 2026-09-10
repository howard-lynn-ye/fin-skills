"""fin_skills.data.sample - the seeded SYNTHETIC world every offline example runs on.

Two properties are load-bearing and both are asserted mechanically:

  * it is NOT market data, and it says so in every field that travels with it. The library
    ships no vendor data and must not start: FRED requires each user's own key and forbids
    redistributing copyright-flagged series, Tiingo's Starter terms forbid persistent
    storage, akshare restricts its DATA to academic research, and Polygon prohibits
    non-display use. A sample cut from any of them would breach one of those.
  * it is generated on demand and never committed - no file of it exists in the package.

Beyond that: the same seed must produce the same bytes, and the world must contain the
defects the guards exist to find, or an example built on it proves nothing.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import fin_skills.data as D
from fin_skills.data.sample import (NOT_MARKET_DATA, RESTATED_PERIOD_END, SEED, SOURCE,
                                    VINTAGE_OBS_DATE, bars, fundamentals, macro, sample)
from fin_skills.data.schema import Adjustment, Bars, Fundamentals, Macro


# ------------------------------------------------------- it says SYNTHETIC on its face
def test_every_provenance_says_synthetic_in_three_independent_places():
    s = sample(n_names=4, years=2, n_dead=1, quarters=8)
    for obj in (s.bars, s.fundamentals, s.macro):
        prov = obj.provenance
        assert prov.source == SOURCE == "synthetic"
        assert prov.library_version == f"synthetic-{SEED}"
        assert prov.terms_url == NOT_MARKET_DATA
        assert "SYNTHETIC" in prov.terms_url and "not market data" in prov.terms_url
        assert prov.request["generator"] == "fin_skills.data.sample"
        assert prov.request["seed"] == SEED
        # the one dataset in this layer that MAY be passed on, because we made it
        assert prov.redistributable is True

    card = s.bars.as_data_source()
    assert card.name == "synthetic"
    assert "SYNTHETIC" in s.describe() and s.describe().isascii()


def test_no_sample_data_file_ships_in_the_package():
    """The generator is code; the data is not. `test_the_package_contains_code_only` in
    test_data_adapters.py asserts the whole package is .py - this pins the reason."""
    pkg = Path(D.__file__).resolve().parent
    assert (pkg / "sample.py").is_file()
    assert not list(pkg.rglob("*.csv")) and not list(pkg.rglob("*.parquet"))
    text = (pkg / "sample.py").read_text(encoding="utf-8")
    for host in ("http://", "https://"):
        assert host not in text, "the generator must not fetch anything"


# ------------------------------------------------------------------------ determinism
def test_the_same_seed_produces_the_same_bytes():
    a = sample(n_names=4, years=2, n_dead=1, quarters=8)
    b = sample(n_names=4, years=2, n_dead=1, quarters=8)
    assert a.bars.fingerprint() == b.bars.fingerprint()
    assert a.fundamentals.fingerprint() == b.fundamentals.fingerprint()
    assert a.macro.fingerprint() == b.macro.fingerprint()

    other = sample(seed=SEED + 1, n_names=4, years=2, n_dead=1, quarters=8)
    assert other.bars.fingerprint() != a.bars.fingerprint()
    assert other.bars.provenance.request["seed"] == SEED + 1


# ------------------------------------------------------------- what the world contains
def test_the_panel_carries_splits_dividends_and_names_that_actually_stop():
    b = bars(n_names=6, years=4, n_dead=2)
    assert isinstance(b, Bars) and b.adjustment is Adjustment.RAW
    assert b.calendar == "XNYS" and b.interval == "1d" and b.half_open

    kinds = set(b.actions["kind"])
    assert kinds == {"split", "dividend"}, "both event types, so a check has both"
    splits = b.actions[b.actions["kind"] == "split"]
    assert list(splits["ratio"]) == [4.0, 2.0]

    alive = b.alive()
    assert int(alive["ends_early"].sum()) == 2, "the dead names STOP; they are not dropped"
    dead = b.listings[b.listings["end_date"].notna()]
    assert len(dead) == 2 and set(dead["reason"]) == {"delisted"}
    # a stopped name is NaN after its last session, never forward-filled
    t = str(dead["ticker"].iloc[0])
    close = b.field("close")[t]
    assert close.isna().any() and close.dropna().index.max() < close.index.max()


def test_the_split_is_a_real_jump_in_the_raw_series_so_a_convention_check_can_see_it():
    b = bars(n_names=4, years=4, n_dead=0)
    t = str(b.actions[b.actions["kind"] == "split"]["ticker"].iloc[0])
    when = b.actions[b.actions["kind"] == "split"]["date"].iloc[0]
    close = b.field("close")[t]
    before = float(close[close.index < when].iloc[-1])
    after = float(close[close.index >= when].iloc[0])
    assert before / after == pytest.approx(4.0, rel=0.15), "the RAW quote took the drop"


def test_readjusting_the_sample_moves_the_prices_and_keeps_the_declaration_honest():
    raw = bars(n_names=4, years=3, n_dead=1)
    anchored = sample(n_names=4, years=3, n_dead=1).bars
    assert raw.adjustment is Adjustment.RAW, "the vendor shape is what bars() builds"
    assert anchored.adjustment is Adjustment.ANCHORED_START, "the set's default is clean"
    assert not anchored.adjustment.rewrites_history
    t = str(raw.actions[raw.actions["kind"] == "split"]["ticker"].iloc[0])
    assert not raw.field("close")[t].equals(anchored.field("close")[t])


def test_the_fundamentals_have_one_restatement_and_one_post_close_filing():
    f = fundamentals(n_entities=2, quarters=12)
    assert isinstance(f, Fundamentals)

    rest = f.restatements()
    assert len(rest) == 1, "exactly one period where PIT and naive disagree"
    row = rest.iloc[0]
    assert row["period_end"] == RESTATED_PERIOD_END
    assert row["restated_val"] < row["original_val"]
    assert row["n_vintages"] == 2

    # the first filing of every entity is accepted after the close, so available_at is
    # the NEXT session - the one-day gap a filed_at join swallows
    first = f.frame.sort_values(["entity_id", "period_end"]).groupby("entity_id").head(1)
    assert (first["available_at"] > first["filed_at"]).all()
    assert (f.frame["available_at"] >= f.frame["filed_at"]).all()
    assert f.frame["available_at"].notna().all()


def test_the_point_in_time_answer_differs_from_the_naive_one():
    """The whole reason a restatement is in the world: as of the date between the original
    filing and the amendment, the number ON FILE is the original one - and
    drop_duplicates(keep='last') returns the restatement instead."""
    f = fundamentals(n_entities=1, quarters=12)
    eid = str(f.frame["entity_id"].iloc[0])
    vints = f.vintages("Revenues", RESTATED_PERIOD_END, entity=eid)
    assert len(vints) == 2
    as_of = vints["available_at"].iloc[1] - pd.Timedelta(days=1)

    pit = D.pit_used(f, as_of, tag="Revenues", entity=eid)
    naive = D.pit_used(f, as_of, tag="Revenues", entity=eid, naive=True)
    key = [k for k in pit.index if k.endswith(RESTATED_PERIOD_END.strftime("%Y-%m-%d"))]
    assert key, key
    assert pit[key[0]] != naive[key[0]], "the vintage filter is doing work"
    assert pit[key[0]] == vints["value"].iloc[0], "the number that was on file"


def test_the_macro_series_has_three_vintages_of_one_observation():
    m = macro(quarters=12)
    assert isinstance(m, Macro)
    v = m.vintages(VINTAGE_OBS_DATE)
    assert len(v) == 3, "FRED's own GDP example has three; so does this"
    assert v["realtime_start"].is_monotonic_increasing

    early = m.as_of(v["realtime_start"].iloc[0], series_id=m.series_ids[0])
    late = m.as_of(v["realtime_start"].iloc[-1], series_id=m.series_ids[0])
    assert early[VINTAGE_OBS_DATE] != late[VINTAGE_OBS_DATE]
    # as_of deduplicates: one value per observation date, never one row per revision
    assert not early.index.has_duplicates and not late.index.has_duplicates
    assert m.latest().attrs["research_use"] == "DISPLAY ONLY"


# -------------------------------------------------------------- it drops into the API
def test_to_bundle_works_on_the_sample_directly_and_every_data_guard_runs_green():
    from fin_skills.api import check

    s = sample(n_names=6, years=4, n_dead=2, quarters=12)
    bundle = s.to_bundle(ticker=s.split_ticker)
    for slot in ("prices", "close", "bars", "actions", "listings", "liquidity",
                 "periods_per_year", "facts", "as_of"):
        assert bundle.get(slot) is not None, slot
    assert bundle.get("periods_per_year") == 252

    report = check(bundle)
    assert {r.guard for r in report} == {"adjustment_check", "pit_fundamentals",
                                         "survivorship_audit"}
    assert not report.failed, [str(f) for r in report for f in r.errors]


def test_the_same_world_in_RAW_makes_the_adjustment_guard_fail_on_purpose():
    """The clean default is not the only thing this generator is for: the same seed with
    adjustment=RAW puts the split back into the series, and `adjustment_check` calls it.
    A fixture that can only produce green runs cannot demonstrate a trap."""
    from fin_skills.api import check

    s = sample(n_names=6, years=4, n_dead=2, quarters=12, adjustment=Adjustment.RAW)
    report = check(s.to_bundle(ticker=s.split_ticker))
    assert {r.guard for r in report.failed} == {"adjustment_check"}
    assert any("RAW" in str(f) for r in report.failed for f in r.errors)


def test_the_helpers_name_the_ticker_each_kind_of_event_belongs_to():
    s = sample(n_names=5, years=3, n_dead=1, quarters=8)
    acts = s.bars.actions
    assert set(acts[acts["ticker"] == s.split_ticker]["kind"]) == {"split"}
    assert set(acts[acts["ticker"] == s.dividend_ticker]["kind"]) == {"dividend"}
    assert s.dead_tickers and all(t in s.bars.tickers for t in s.dead_tickers)
    assert "Sample(seed=" in repr(s)


def test_the_package_exports_it_under_one_name():
    import sys

    assert D.sample is sys.modules["fin_skills.data.sample"].sample
    assert isinstance(D.sample(n_names=3, years=1, n_dead=0, quarters=4), D.Sample)


def test_the_generator_refuses_a_world_with_no_survivors():
    with pytest.raises(ValueError, match="at least one name alive"):
        bars(n_names=3, n_dead=3)
