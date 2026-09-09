"""fin_skills.data - the three schemas, provenance, and what to_bundle() fills.

Every fixture is seeded synthetic data. The autouse fixture in conftest.py refuses every
socket, so nothing here can reach a network even by accident.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _data_fixtures import (GDP_VINTAGES, ORIGINAL_VAL, RESTATED_VAL, RESTATEMENT_AS_OF,
                            clean_bars, clean_fundamentals, clean_macro,
                            fundamentals_frame, macro_frame, raw_panel, split_ticker)
from fin_skills.data import (Adjustment, Bars, Fundamentals, Macro, content_hash,
                             guard_convention, pit_used, to_bundle)
from fin_skills.data.convert import FILLS, periods_per_year, to_long
from fin_skills.data.provenance import Provenance, make as make_provenance, scrub
from fin_skills.data.schema import FUNDAMENTAL_COLUMNS, MACRO_COLUMNS


@pytest.fixture(scope="module")
def bars() -> Bars:
    return clean_bars(n_names=8, years=6, n_dead=3)


# ------------------------------------------------------------------- D1 declarations
def test_bars_rejects_a_construction_missing_a_required_declaration():
    frame, _, _ = raw_panel(n_names=2, years=1, n_dead=0)
    prov = make_provenance("t", library_version="0", request={}, content=frame)
    kw = dict(frame=frame, adjustment=Adjustment.RAW, calendar="XNYS",
              tz="America/New_York", interval="1d", bar_label="close", currency="USD",
              provenance=prov)
    Bars(**kw)                                        # complete: fine
    for missing in ("adjustment", "calendar", "tz", "interval", "bar_label", "currency"):
        with pytest.raises(TypeError):
            Bars(**{k: v for k, v in kw.items() if k != missing})


@pytest.mark.parametrize("column", FUNDAMENTAL_COLUMNS)
def test_fundamentals_rejects_a_frame_missing_a_required_column(column):
    frame = fundamentals_frame().drop(columns=[column])
    prov = make_provenance("edgar", library_version="0", request={}, content=frame)
    with pytest.raises(ValueError, match=column):
        Fundamentals(frame=frame, provenance=prov)


@pytest.mark.parametrize("column", MACRO_COLUMNS)
def test_macro_rejects_a_frame_missing_a_required_column(column):
    frame = macro_frame().drop(columns=[column])
    prov = make_provenance("fred", library_version="0", request={}, content=frame)
    with pytest.raises(ValueError, match=column):
        Macro(frame=frame, provenance=prov)


def test_unknown_adjustment_is_legal_and_poisons_the_run(bars):
    poisoned = Bars(frame=bars.frame, adjustment=Adjustment.UNKNOWN,
                    calendar=bars.calendar, tz=bars.tz, interval=bars.interval,
                    bar_label=bars.bar_label, currency=bars.currency,
                    provenance=bars.provenance, actions=bars.actions)
    assert not poisoned.adjustment.is_declared
    errors = [f for f in poisoned.validate() if f.severity == "error"]
    assert errors and "UNKNOWN" in errors[0].message
    with pytest.raises(ValueError, match="no basis to convert"):
        poisoned.readjust(Adjustment.ANCHORED_START)


def test_rewrites_history_is_the_behavioural_question_not_the_label():
    assert Adjustment.ANCHORED_PRESENT.rewrites_history          # anchored at the present (qfq)
    assert not Adjustment.ANCHORED_START.rewrites_history         # anchored at the start (hfq)
    assert not Adjustment.RAW.rewrites_history


# ------------------------------------------------------------------------ D2 readjust
def test_readjust_raises_without_an_actions_table(bars):
    naked = Bars(frame=bars.frame, adjustment=bars.adjustment, calendar=bars.calendar,
                 tz=bars.tz, interval=bars.interval, bar_label=bars.bar_label,
                 currency=bars.currency, provenance=bars.provenance, actions=None)
    with pytest.raises(ValueError, match="actions table"):
        naked.readjust(Adjustment.ANCHORED_PRESENT)


def test_readjust_round_trips_and_differs_by_the_cumulative_factor(bars):
    t = split_ticker(bars)
    total = float(bars.actions["ratio"].prod())
    fwd = bars.readjust(Adjustment.ANCHORED_PRESENT)
    ratio = (bars.close(t) / fwd.close(t)).dropna().round(9).unique()
    assert ratio.tolist() == [total], "back/forward differ by ONE constant, the cum factor"
    back_again = fwd.readjust(Adjustment.ANCHORED_START)
    assert np.allclose(back_again.close(t), bars.close(t), rtol=1e-12)
    assert fwd.adjustment is Adjustment.ANCHORED_PRESENT
    assert fwd.provenance.content_sha256 != bars.provenance.content_sha256


def test_the_guard_vocabulary_is_inverted_and_the_table_says_so():
    # ANCHORED_START is anchored at the START, which detect_convention calls "forward-adjusted"
    assert guard_convention(Adjustment.ANCHORED_START) == "forward-adjusted"
    assert guard_convention(Adjustment.ANCHORED_PRESENT) == "back-adjusted"
    assert guard_convention(Adjustment.RAW) == "raw"
    assert guard_convention(Adjustment.UNKNOWN) == "unknown"


# ------------------------------------------------------------------------ D3 calendar
def test_a_bar_on_a_non_session_raises_and_a_missing_session_stays_nan(bars):
    sessions = bars.frame.index
    with pytest.raises(ValueError, match="non-sessions"):
        bars.align(sessions[:-5])                       # five bars now have no session

    holes = sessions.delete([10, 11, 12])               # three sessions with no bar
    thinned = Bars(frame=bars.frame.drop(index=sessions[[10, 11, 12]]),
                   adjustment=bars.adjustment, calendar=bars.calendar, tz=bars.tz,
                   interval=bars.interval, bar_label=bars.bar_label,
                   currency=bars.currency, provenance=bars.provenance)
    aligned = thinned.align(sessions)
    assert len(aligned.frame) == len(sessions)
    gap = aligned.frame.loc[sessions[[10, 11, 12]]]
    assert gap.isna().all().all(), "a missing session is NaN and is never filled"
    assert not holes.equals(sessions)


def test_index_tz_must_match_the_declaration(bars):
    with pytest.raises(ValueError, match="contradicts the declared"):
        Bars(frame=bars.frame, adjustment=bars.adjustment, calendar=bars.calendar,
             tz="Asia/Shanghai", interval=bars.interval, bar_label=bars.bar_label,
             currency=bars.currency, provenance=bars.provenance)


def test_an_unsorted_or_duplicated_index_is_refused(bars):
    kw = dict(adjustment=bars.adjustment, calendar=bars.calendar, tz=bars.tz,
              interval=bars.interval, bar_label=bars.bar_label,
              currency=bars.currency, provenance=bars.provenance)
    with pytest.raises(ValueError, match="unsorted"):
        Bars(frame=bars.frame.iloc[::-1], **kw)
    with pytest.raises(ValueError, match="duplicate"):
        Bars(frame=pd.concat([bars.frame.iloc[:5], bars.frame.iloc[:5]]).sort_index(), **kw)


# -------------------------------------------------------------------------- D5 PIT
def test_as_of_selects_the_vintage_on_file_and_never_the_restatement():
    f = clean_fundamentals()
    on_file = f.as_of(RESTATEMENT_AS_OF)
    q3 = on_file[on_file["period_end"] == pd.Timestamp("2022-09-30")]
    assert len(q3) == 1
    assert float(q3["value"].iloc[0]) == ORIGINAL_VAL

    # what drop_duplicates(keep="last") picks instead: the amendment filed 2023-02-14
    naive = pit_used(f, RESTATEMENT_AS_OF, tag="Revenues", naive=True)
    pit = pit_used(f, RESTATEMENT_AS_OF, tag="Revenues")
    key = "2022-07-01..2022-09-30"
    assert naive[key] == RESTATED_VAL and pit[key] == ORIGINAL_VAL
    assert naive[key] != pit[key], "the fixture must make the two paths disagree"

    # and it invents a period that was not filed until 2023-02-03
    assert "2022-10-01..2022-12-31" in naive.index
    assert "2022-10-01..2022-12-31" not in pit.index


def test_as_of_is_a_tail_not_a_groupby_last():
    """GroupBy.last() takes the last NON-NULL value of each column independently, which can
    assemble a row that was never filed in that shape."""
    frame = fundamentals_frame()
    frame.loc[frame["accn"] == "0000000-23-001", "form"] = None
    f = Fundamentals(frame=frame,
                     provenance=make_provenance("edgar", library_version="0", request={},
                                                content=frame))
    row = f.as_of("2023-06-01")
    q3 = row[row["period_end"] == pd.Timestamp("2022-09-30")].iloc[0]
    assert q3["value"] == RESTATED_VAL
    assert q3["form"] is None or pd.isna(q3["form"]), \
        "the latest vintage's own null must survive, not be back-filled from an older one"


def test_vintages_shows_every_version_of_one_number():
    v = clean_fundamentals().vintages("Revenues", "2022-09-30")
    assert list(v["value"]) == [ORIGINAL_VAL, RESTATED_VAL]
    assert v["available_at"].is_monotonic_increasing


def test_to_daily_is_an_asof_join_not_a_fill():
    sessions = pd.bdate_range("2022-07-01", "2023-03-01")
    daily = clean_fundamentals().to_daily(sessions, "Revenues")
    col = daily.columns[0]
    assert daily.loc[:pd.Timestamp("2022-08-03"), col].isna().all(), \
        "nothing was knowable before the first filing, and nothing is invented"
    assert daily.loc[pd.Timestamp("2022-08-04"), col] == 880_000_000.0
    assert daily.loc[pd.Timestamp("2022-11-03"), col] == 880_000_000.0, \
        "the Q3 figure was accepted after the close on 2022-11-03, so not that session"
    assert daily.loc[pd.Timestamp("2022-11-04"), col] == ORIGINAL_VAL
    assert daily.loc[pd.Timestamp("2023-02-15"), col] == 1_100_000_000.0


# ------------------------------------------------------------------------ D6 vintages
def test_macro_as_of_resolves_each_published_vintage_and_deduplicates():
    m = clean_macro()
    q4 = pd.Timestamp("2013-10-01")
    for (realtime, value), ask in zip(GDP_VINTAGES, ("2014-02-01", "2014-03-01",
                                                     "2014-04-01")):
        s = m.as_of(ask)
        assert s.index.is_unique, "one row per obs_date - fredapi returns one per revision"
        assert s.loc[q4] == value, (ask, realtime)

    # the deduplication fredapi's get_series_as_of_date() omits: five rows are known by
    # 2014-06-01 and they describe two observation dates
    late = m.as_of("2014-06-01")
    known = m.frame[m.frame["realtime_start"] <= pd.Timestamp("2014-06-01")]
    assert len(known) == 5 and len(late) == 2
    assert late.loc[q4] == GDP_VINTAGES[-1][1]

    early = m.as_of("2014-01-01")
    assert q4 not in early.index, "no row may have realtime_start after the as-of date"


def test_macro_latest_is_marked_display_only():
    s = clean_macro().latest()
    assert s.attrs["research_use"] == "DISPLAY ONLY"
    assert s.loc[pd.Timestamp("2013-10-01")] == GDP_VINTAGES[-1][1]


def test_macro_refuses_a_row_without_a_realtime_start():
    frame = macro_frame()
    frame.loc[0, "realtime_start"] = None
    with pytest.raises(ValueError, match="realtime_start"):
        Macro(frame=frame,
              provenance=make_provenance("fred", library_version="0", request={},
                                         content=frame))


# ------------------------------------------------------------ D7 content addressing
def test_hash_is_content_addressed_not_path_or_time_addressed(bars):
    frame = bars.frame
    shuffled = frame[list(frame.columns)[::-1]]
    assert content_hash(frame) == content_hash(shuffled), "column order is not content"

    changed = frame.copy()
    col = changed.columns[0]
    changed.iloc[7, changed.columns.get_loc(col)] += 0.01
    assert content_hash(frame) != content_hash(changed), "one cell must move the hash"

    a = make_provenance("yfinance", library_version="1.7.0", request={"s": ["X"]},
                        content=frame, retrieved_at="2020-01-01T00:00:00+00:00")
    b = make_provenance("yfinance", library_version="1.7.0", request={"s": ["X"]},
                        content=frame, retrieved_at="2026-09-09T00:00:00+00:00")
    assert a.content_sha256 == b.content_sha256, "the clock is not content"


def test_fingerprint_covers_the_declarations_too(bars):
    relabelled = Bars(frame=bars.frame, adjustment=Adjustment.UNKNOWN,
                      calendar=bars.calendar, tz=bars.tz, interval=bars.interval,
                      bar_label=bars.bar_label, currency=bars.currency,
                      provenance=bars.provenance, actions=bars.actions)
    assert relabelled.fingerprint() != bars.fingerprint()
    assert content_hash(relabelled.frame) == content_hash(bars.frame)


# ------------------------------------------------------------------- credentials
def test_provenance_never_carries_a_credential(monkeypatch):
    monkeypatch.setenv("SOME_VENDOR_TOKEN", "PLAINTEXTSECRETVALUE")
    prov = make_provenance(
        "fred", library_version="0.5.2", content=pd.Series([1.0, 2.0]),
        request={"api_key": "0123456789abcdef0123456789abcdef",     # FRED key shape
                 "token": "another-secret", "series_ids": ["GDP"],
                 "headers": {"Authorization": "Bearer abc"},
                 "innocent_name": "PLAINTEXTSECRETVALUE",           # caught by VALUE
                 "nested": [{"password": "hunter2"}]})
    blob = repr(prov) + prov.to_json() + str(prov.to_dict()) + str(prov.request)
    for secret in ("0123456789abcdef0123456789abcdef", "another-secret", "hunter2",
                   "Bearer abc", "PLAINTEXTSECRETVALUE"):
        assert secret not in blob, secret
    assert "GDP" in blob, "scrubbing must not eat the request it is describing"


def test_scrub_is_applied_at_construction_not_at_print_time():
    p = Provenance(source="x", adapter_version="0", library_version="0",
                   retrieved_at="2026-09-09T00:00:00+00:00",
                   request={"api_key": "abc"}, content_sha256="0" * 64)
    assert p.request == {"api_key": "<redacted>"}
    assert scrub({"nested": {"secret": "s"}})["nested"]["secret"] == "<redacted>"


# ------------------------------------------------------------------------ D11 dtypes
def test_dtypes_are_exact(bars):
    for col in bars.frame.columns:
        want = "int64" if col[0] == "volume" else "float64"
        assert str(bars.frame[col].dtype) == want, col
    # 16,777,217 = 2**24 + 1: the first integer float32 cannot represent
    v = pd.Series([16_777_217], dtype="int64")
    assert int(v.astype("float32").iloc[0]) == 16_777_216
    assert int(v.iloc[0]) == 16_777_217


def test_float32_volume_is_an_error_finding(bars):
    frame = bars.frame.copy()
    col = [c for c in frame.columns if c[0] == "volume"][0]
    frame[col] = frame[col].astype("float32")
    poor = Bars(frame=frame, adjustment=bars.adjustment, calendar=bars.calendar,
                tz=bars.tz, interval=bars.interval, bar_label=bars.bar_label,
                currency=bars.currency, provenance=bars.provenance, actions=bars.actions)
    assert any(f.severity == "error" and "float32" in f.message for f in poor.validate())


# ------------------------------------------------------------------- D12 to_bundle
def test_to_bundle_fills_exactly_the_documented_slots(bars):
    """The design lists nine slots. FILLS is that list, and to_bundle fills it exactly -
    no more (an undocumented slot is an unreviewed one) and no fewer."""
    assert set(FILLS) == {"prices", "close", "bars", "actions", "listings", "liquidity",
                          "periods_per_year", "facts", "as_of"}
    b = to_bundle(bars, fundamentals=clean_fundamentals(), ticker=split_ticker(bars),
                  as_of=RESTATEMENT_AS_OF)
    assert set(b.slots()) == set(FILLS)


def test_to_bundle_from_bars_alone_fills_only_the_bars_slots(bars):
    b = to_bundle(bars, ticker=split_ticker(bars))
    assert set(b.slots()) == {"prices", "close", "bars", "actions", "listings",
                              "liquidity", "periods_per_year"}
    assert list(b.bars.columns) == ["open", "high", "low", "close", "volume"]
    assert b.periods_per_year == 252


def test_to_bundle_from_fundamentals_alone_fills_only_facts_and_as_of():
    b = to_bundle(fundamentals=clean_fundamentals(), as_of=RESTATEMENT_AS_OF)
    assert set(b.slots()) == {"facts", "as_of"}
    assert b.as_of == RESTATEMENT_AS_OF
    assert all({"start", "end", "val", "accn", "form", "filed"} <= set(f) for f in b.facts)


def test_a_macro_object_fills_no_slot_today():
    b = to_bundle(macro=clean_macro())
    assert set(b.slots()) == set()
    with pytest.raises(TypeError):
        to_bundle(macro=clean_macro().frame)


def test_periods_per_year_is_stated_only_where_it_is_unambiguous():
    assert periods_per_year("1d", "XNYS") == 252
    assert periods_per_year("1d", "24/7") == 365
    assert periods_per_year("1h", "XNYS") is None, "an intraday factor is not invented"
    assert periods_per_year("1m", "24/7") is None


def test_a_multi_ticker_panel_needs_a_ticker_for_the_single_name_slots(bars):
    b = to_bundle(bars)
    assert "close" not in b and "bars" not in b
    assert "prices" in b and "liquidity" in b
    with pytest.raises(KeyError, match="NOPE"):
        to_bundle(bars, ticker="NOPE")


def test_long_and_wide_round_trip(bars):
    long = to_long(bars)
    assert list(long.columns) == ["date", "ticker", "field", "value"]
    assert long["field"].nunique() == 5
    assert len(long) <= bars.frame.size


def test_liquidity_is_dollar_volume(bars):
    b = to_bundle(bars, ticker=split_ticker(bars))
    t = split_ticker(bars)
    expected = bars.close(t) * bars.frame[("volume", t)]
    assert np.allclose(b.liquidity[t].dropna(), expected.dropna())


# ---------------------------------------------------- the two adjustment vocabularies
# `Adjustment` names its members after the ANCHOR because "back" and "forward" are
# inverted between vocabularies that are all in use: `core.adjustment_check` calls the
# present-anchored series "back-adjusted", A-share qfq is present-anchored, and English
# futures usage anchors "back-adjusted" at the newest contract. These tests pin the
# translation to the guard, so a future rename cannot quietly invert it.
def test_the_enum_is_named_after_the_anchor_not_after_back_or_forward():
    from fin_skills.data.convert import guard_convention
    assert {m.name for m in Adjustment} == {
        "RAW", "ANCHORED_START", "ANCHORED_PRESENT", "RAW_PLUS_FACTORS", "UNKNOWN"}
    # the words themselves must not reappear as member names
    assert not {"BACK", "FORWARD"} & {m.name for m in Adjustment}
    # only the present-anchored convention rewrites history - that is the behavioural test
    assert Adjustment.ANCHORED_PRESENT.rewrites_history
    assert not any(m.rewrites_history for m in Adjustment if m is not Adjustment.ANCHORED_PRESENT)
    # and the translation to the guard's vocabulary is the INVERSE of the naive reading
    assert guard_convention(Adjustment.ANCHORED_START) == "forward-adjusted"
    assert guard_convention(Adjustment.ANCHORED_PRESENT) == "back-adjusted"


def test_the_guards_own_definition_is_the_one_the_translation_targets():
    # Not a restatement of the table: read the guard module's own docstring, so this test
    # fails if adjustment_check ever changes what it means by the two words.
    import fin_skills.core.adjustment_check as ac
    doc = ac.__doc__ or ""
    assert "back-adjusted    -- anchored at the PRESENT" in doc
    assert "forward-adjusted -- anchored at the START" in doc


def test_a_cache_written_before_the_rename_still_loads():
    assert Adjustment("back") is Adjustment.ANCHORED_START
    assert Adjustment("forward") is Adjustment.ANCHORED_PRESENT
    with pytest.raises(ValueError):
        Adjustment("sideways")
