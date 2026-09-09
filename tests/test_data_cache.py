"""fin_skills.data.cache - refusal, round-trip fidelity, and divergence as evidence."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from _data_fixtures import (StubAdapter, bad_tick_vintage, clean_bars, clean_fundamentals,
                            clean_macro, rewritten_vintage, split_ticker)
from fin_skills.api import check
from fin_skills.data import Adjustment, Bars, Cache, Unpublished, content_hash, declare
from fin_skills.data.cache import META, CachePolicyError, Divergence
from fin_skills.data.declare import Declaration
from fin_skills.data.provenance import make as make_provenance


@pytest.fixture(scope="module")
def bars() -> Bars:
    return clean_bars(n_names=4, years=5, n_dead=1)


@pytest.fixture
def cache(tmp_path) -> Cache:
    return Cache(tmp_path / "store")


# ---------------------------------------------------------------- D13 cache policy
@pytest.fixture
def no_persist_adapter():
    """A stub declaring what Tiingo's Starter plan declares: no durable storage."""
    decl = Declaration(
        name="stubvendor", library="stubvendor", library_license="MIT",
        licence_source="pypi", adjustment_default=Adjustment.FORWARD,
        adjustment_supported=(Adjustment.FORWARD,), calendar="XNYS",
        tz="America/New_York", bar_label="close", interval_support=("1d",),
        includes_delisted=False, point_in_time=False, rate_limit=Unpublished(),
        free_tier="Starter $0", requires_key=True, key_env_var="STUBVENDOR_TOKEN",
        key_sharing="byok-blessed", cache_policy="no-persist",
        non_display_use="unstated", terms_url="https://example.invalid/terms",
        redistribution="prohibited", verified_on="2026-09-09",
        notes="ToS 1.6(a) forbids retaining the data in any persistent storage")
    declare.register(None, decl)
    try:
        yield decl
    finally:
        declare.unregister(decl.name)


def test_put_refuses_a_no_persist_provider(cache, bars, no_persist_adapter):
    prov = make_provenance("stubvendor", library_version="1.0",
                           request={"method": "bars", "symbols": ["X"]},
                           content=bars.frame)
    with pytest.raises(CachePolicyError, match="no-persist"):
        cache.put(bars, prov)
    assert not any(cache.root.iterdir()), "the refusal must write nothing at all"

    key = cache.put(bars, prov, acknowledge_paid_tier=True)
    assert cache.get(key) is not None
    assert cache.manifest()["cache_policy"].tolist() == ["no-persist"]


def test_a_persist_ok_provider_needs_no_acknowledgement(cache, bars):
    key = cache.put(bars, bars.provenance)
    assert cache.get(key) is not None


def test_declaration_warnings_name_the_no_persist_risk(no_persist_adapter):
    warnings = no_persist_adapter.warnings()
    assert any("no-persist" in w for w in warnings)
    assert any("byok" in w or "own" in w for w in warnings)
    assert not no_persist_adapter.may_persist


# ------------------------------------------------------------------ round-tripping
def test_the_timezone_comes_back_from_the_sidecar_not_the_file(cache, bars):
    key = cache.put(bars, bars.provenance)
    folder = cache.vintages(key)[0]
    meta = json.loads((folder / META).read_text(encoding="utf-8"))
    assert meta["frame"]["index_tz"] == "America/New_York"

    csv = (folder / "frame.csv").read_text(encoding="utf-8").splitlines()[2]
    assert "-05:00" in csv or "-04:00" in csv, "CSV keeps only a fixed offset"
    assert "America/New_York" not in csv, "the named zone is NOT in the data file"

    back, _ = cache.get(key)
    assert str(back.frame.index.tz) == "America/New_York"
    assert back.tz == bars.tz and back.adjustment is bars.adjustment
    assert back.frame.index.equals(bars.frame.index)


def test_round_trip_is_bit_exact_including_dtypes(cache, bars):
    key = cache.put(bars, bars.provenance)
    back, prov = cache.get(key)
    assert content_hash(back.frame) == content_hash(bars.frame)
    assert prov.content_sha256 == bars.provenance.content_sha256
    for col in bars.frame.columns:
        assert back.frame[col].dtype == bars.frame[col].dtype, col
    assert back.actions is not None and len(back.actions) == len(bars.actions)
    assert back.listings is not None


def test_fundamentals_and_macro_round_trip(cache):
    for obj in (clean_fundamentals(), clean_macro()):
        key = cache.put(obj, obj.provenance)
        back, _ = cache.get(key)
        assert type(back) is type(obj)
        assert content_hash(back.frame) == content_hash(obj.frame)


def test_a_zero_padded_identifier_survives_the_round_trip(cache):
    """A CIK is text that looks like a number. Read back as a number, "0000320193"
    becomes 320193 and stops matching anything; the sidecar says it was text."""
    f = clean_fundamentals()
    assert f.frame["entity_id"].iloc[0] == "0000320193"
    key = cache.put(f, f.provenance)
    back, _ = cache.get(key)
    assert back.frame["entity_id"].iloc[0] == "0000320193"
    assert content_hash(back.frame) == content_hash(f.frame)


def test_the_key_is_the_request_and_carries_no_credential(cache, bars):
    decl = declare.lookup("fred")
    k1 = cache.key({"series_ids": ["GDP"], "api_key": "0" * 32}, decl)
    k2 = cache.key({"api_key": "f" * 32, "series_ids": ["GDP"]}, decl)
    assert k1 == k2, "two different keys, one request - the credential is not part of it"
    assert cache.key({"series_ids": ["CPI"]}, decl) != k1


def test_verify_catches_a_file_changed_underneath(cache, bars):
    key = cache.put(bars, bars.provenance)
    assert all(f.severity == "info" for f in cache.verify())

    path = cache.vintages(key)[0] / "frame.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    head, cell = lines[5].split(",", 1)
    lines[5] = f"{head},{float(cell.split(',')[0]) + 1.0}," + cell.split(",", 1)[1]
    path.write_text("\n".join(lines), encoding="utf-8")

    findings = cache.verify()
    assert any(f.severity == "error" and "hash mismatch" in f.message for f in findings)


def test_manifest_has_one_row_per_vintage(cache, bars):
    key = cache.put(bars, bars.provenance)
    cache.put(bars, bars.provenance.with_content(bars.frame * 1.0),
              decl=declare.lookup("yfinance"))
    man = cache.manifest()
    assert list(man["vintage"]) == [0, 1]
    assert set(man["key"]) == {key}
    assert man["source"].unique().tolist() == ["yfinance"]
    assert man["adjustment"].unique().tolist() == ["back"]
    assert man["includes_delisted"].unique().tolist() == [False]
    assert man["span"].iloc[0].count("..") == 1


def test_keep_vintages_false_replaces_instead_of_accumulating(tmp_path, bars):
    c = Cache(tmp_path / "one", keep_vintages=False)
    key = c.put(bars, bars.provenance)
    c.put(bars, bars.provenance)
    assert len(c.vintages(key)) == 1


# ------------------------------------------------------- D8 refetch and divergence
def test_refetch_records_a_divergence_and_keeps_both_vintages(cache, bars):
    t = split_ticker(bars)
    key = cache.put(bars, bars.provenance)
    stub = StubAdapter(rewritten_vintage(bars, t))

    div = cache.refetch(key, stub)

    assert isinstance(div, Divergence) and div.changed
    assert len(cache.vintages(key)) == 2, "the old vintage must still be on disk"
    old, _ = cache.get_vintage(key, 0)
    assert content_hash(old.frame) == content_hash(bars.frame), "nothing was overwritten"
    assert div.n_changed > 0 and div.max_abs_bps > 0
    assert stub.requests[0]["method"] == "bars", "the stored request drove the refetch"
    assert "both vintages retained" in div.report()


def test_a_divergence_bundle_is_accepted_by_reconcile_sources(cache, bars):
    t = split_ticker(bars)
    key = cache.put(bars, bars.provenance)
    div = cache.refetch(key, StubAdapter(rewritten_vintage(bars, t)))

    bundle = div.to_bundle()
    assert set(bundle.slots()) == {"close", "other", "actions"}
    report = check(bundle, guards=["reconcile_sources"])
    assert len(report) == 1 and report[0].passed
    verdict = report[0].evidence["verdict"]
    assert verdict.startswith("ADJUSTMENT CONVENTION ONLY"), verdict


def test_a_fat_finger_print_is_reported_as_a_data_error(cache, bars):
    t = split_ticker(bars)
    key = cache.put(bars, bars.provenance)
    div = cache.refetch(key, StubAdapter(bad_tick_vintage(bars, t)))

    report = check(div.to_bundle(), guards=["reconcile_sources"])
    assert not report[0].passed
    assert report[0].evidence["verdict"].startswith("DATA ERROR")


def test_an_unchanged_refetch_reports_no_divergence(cache, bars):
    key = cache.put(bars, bars.provenance)
    div = cache.refetch(key, StubAdapter(bars))
    assert not div.changed and div.n_changed == 0 and div.first_changed is None
    assert "UNCHANGED" in div.report()


def test_refetch_on_an_unknown_key_says_so(cache):
    with pytest.raises(KeyError, match="nothing cached"):
        cache.refetch("0" * 32, StubAdapter())
