"""The five adapters: what they declare, what they refuse, and what they never import.

Three properties are asserted mechanically rather than reviewed by eye, because all three
have been got wrong in the wild:

  * no vendor library is imported until it is used (proved by blocking all five on
    sys.meta_path and importing the package anyway);
  * the three sources with no usable published rate limit contain no rate constant;
  * no credential can enter this library except through the declared environment variable.
"""
from __future__ import annotations

import ast
import importlib
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fin_skills.data as D
from fin_skills.data import declare
from fin_skills.data.adapters import MODULES, Base, get, library_version, require
from fin_skills.data.ratelimit import (PerInstanceDelay, PerSecond, RateLimit, SHAPES,
                                       Unpublished)
from fin_skills.data.schema import MACRO_COLUMNS

VENDORS = ("yfinance", "akshare", "ccxt", "fredapi", "edgar", "edgartools")
PKG = Path(D.__file__).resolve().parent
ADAPTER_DIR = PKG / "adapters"

#: the three whose vendors publish no usable number - see ratelimit.Unpublished
UNPUBLISHED_MODULES = ("yfinance.py", "akshare.py", "fredapi.py")


def _source(name: str) -> str:
    return (ADAPTER_DIR / name).read_text(encoding="utf-8")


def _tree(name: str) -> ast.Module:
    return ast.parse(_source(name))


# ------------------------------------------------------------------- lazy imports
class _Blocker:
    """A meta-path finder that makes the five vendor libraries un-importable."""

    def __init__(self, names) -> None:
        self.names = set(names)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.names:
            raise ImportError(f"blocked in this test: {fullname}")
        return None


def test_importing_the_package_imports_no_vendor_library():
    blocker = _Blocker(VENDORS)
    saved_modules = {k: v for k, v in sys.modules.items()
                     if k.split(".")[0] in VENDORS or k.startswith("fin_skills.data")}
    for k in saved_modules:
        del sys.modules[k]
    sys.meta_path.insert(0, blocker)
    try:
        data = importlib.import_module("fin_skills.data")
        assert not [m for m in sys.modules if m.split(".")[0] in VENDORS]
        assert len(data.declarations()) == 5, "the whole table is available regardless"
        assert not data.adapters().empty
    finally:
        sys.meta_path.remove(blocker)
        for k in [m for m in sys.modules if m.startswith("fin_skills.data")]:
            del sys.modules[k]
        sys.modules.update(saved_modules)
        importlib.import_module("fin_skills.data")


@pytest.mark.parametrize("name", sorted(f"{m}.py" for m in MODULES.values()))
def test_no_adapter_module_imports_its_vendor_at_module_scope(name):
    tree = _tree(name)
    for node in tree.body:                       # module scope only
        if isinstance(node, ast.Import):
            roots = {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots = {(node.module or "").split(".")[0]}
        else:
            continue
        assert not roots & set(VENDORS), f"{name} imports {roots} at module scope"


def test_require_names_the_pip_install():
    with pytest.raises(ImportError, match="pip install edgartools"):
        require("edgar_definitely_not_installed", pip_name="edgartools")


def test_library_version_reads_the_object_it_is_given():
    class Fake:
        __version__ = "1.2.3"

    assert library_version(Fake()) == "1.2.3"


# ------------------------------------------------------------------ the declarations
def test_five_adapters_are_registered_with_the_expected_shape():
    decls = {d.name: d for d in declare.declarations()}
    assert set(decls) == {"yfinance", "akshare", "ccxt", "fred", "edgar"}
    assert set(MODULES) == set(decls)
    for d in decls.values():
        assert d.verified_on == "2026-09-09"
        assert isinstance(d.rate_limit, RateLimit)
        assert type(d.rate_limit) in SHAPES
        assert d.terms_url.startswith("https://")


def test_no_free_source_here_claims_to_include_delisted_names():
    for d in declare.declarations():
        if d.name == "edgar":
            assert d.includes_delisted, "EDGAR filings survive: Lehman is still on file"
            continue
        assert not d.includes_delisted, d.name
        assert any("UPPER BOUND" in w for w in d.warnings())


def test_point_in_time_is_true_only_where_a_vintage_exists():
    pit = {d.name for d in declare.declarations() if d.point_in_time}
    assert pit == {"fred", "edgar"}


def test_akshare_declares_raw_because_that_is_what_akshare_returns():
    """The ecosystem default is qfq; akshare's own `adjust=""` is unadjusted. The
    declaration states what the vendor does, not what its users usually ask for."""
    d = declare.lookup("akshare")
    assert d.adjustment_default is D.Adjustment.RAW
    assert not d.rewrites_history
    assert "academic research" in d.notes


def test_yfinance_declares_the_convention_that_rewrites_history():
    d = declare.lookup("yfinance")
    assert d.adjustment_default is D.Adjustment.FORWARD    # auto_adjust=True since 1.0
    assert d.rewrites_history
    assert any("rewrites history" in w for w in d.warnings())


# --------------------------------------------------------------- D14 licence provenance
def test_licence_provenance_is_resolved_not_assumed():
    """Verified against PyPI's JSON API on 2026-09-09: fredapi declares no `license`, no
    `license_expression` and no License classifier, so its SPDX string can only come from
    the repository LICENSE. edgartools, akshare, ccxt and yfinance all declare one."""
    by_name = {d.name: d for d in declare.declarations()}
    assert by_name["fred"].licence_source == "repo-LICENSE"
    assert by_name["fred"].library_license == "Apache-2.0"
    assert any("will not see it" in w for w in by_name["fred"].warnings())
    for name in ("yfinance", "akshare", "ccxt", "edgar"):
        assert by_name[name].licence_source == "pypi", name
    assert {d.library_license for d in declare.declarations()} == {"MIT", "Apache-2.0"}
    for d in declare.declarations():
        assert d.licence_source in declare.LICENCE_SOURCE


def test_redistribution_is_prohibited_unless_the_source_is_public():
    by_name = {d.name: d for d in declare.declarations()}
    assert by_name["edgar"].redistribution == "public-domain"
    assert by_name["fred"].redistribution == "attribution"
    assert by_name["fred"].attribution.startswith("This product uses the FRED(R) API")
    for name in ("yfinance", "akshare", "ccxt"):
        assert by_name[name].redistribution == "prohibited", name


# -------------------------------------------------- D9 no invented rate limits
_SHAPE_NAMES = {"PerSecond", "PerMinute", "PerIP", "PerAccount", "PerInstanceDelay",
                "WeightedDaily", "PerHourDayMonth"}
_RATE_ISH = re.compile(r"(?i)rate|limit|rpm|rps|per_second|per_minute|per_hour|per_day|"
                       r"throttl|qps|quota|budget|cooldown|interval|delay")


def _rate_constants(tree: ast.Module) -> list[str]:
    """Names bound to a NUMBER whose name says it is a rate."""
    out = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        else:
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value,
                                                               (int, float)):
            out += [t for t in targets if _RATE_ISH.search(t)]
    return out


def _shape_calls(tree: ast.Module) -> list[str]:
    return [n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id in _SHAPE_NAMES]


@pytest.mark.parametrize("name", UNPUBLISHED_MODULES)
def test_an_unpublished_source_carries_no_hardcoded_rate_constant(name):
    tree = _tree(name)
    assert _rate_constants(tree) == [], f"{name} invented a rate constant"
    assert _shape_calls(tree) == [], f"{name} declared a numeric rate shape"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Unpublished":
            assert not node.args and not node.keywords, \
                f"{name} passed a number to Unpublished(); the pace is the caller's"


@pytest.mark.parametrize("adapter", ["yfinance", "akshare", "fred"])
def test_those_three_declare_the_unpublished_shape(adapter):
    lim = declare.lookup(adapter).rate_limit
    assert isinstance(lim, Unpublished)
    assert lim.courtesy_per_s is None
    assert "no vendor number" in lim.describe()


def test_the_detector_finds_a_real_rate_constant_where_one_belongs():
    """A positive control: edgar.py DOES carry a number, so the scan above can fail."""
    tree = _tree("edgar.py")
    assert "EDGARTOOLS_SELF_THROTTLE" in _rate_constants(tree)
    assert _shape_calls(tree) == ["PerSecond"]


def test_freds_two_documented_shapes_are_recorded_as_prose_not_as_a_constant():
    d = declare.lookup("fred")
    assert "120 requests per minute" in d.free_tier      # v1 errors page
    assert "2 requests per second" in d.free_tier        # v2 errors page
    assert "courtesy_per_s=2.0" in d.free_tier           # what satisfies both
    assert isinstance(d.rate_limit, Unpublished)


def test_edgar_throttles_below_the_sec_ceiling():
    from fin_skills.data.adapters.edgar import EDGARTOOLS_SELF_THROTTLE

    assert EDGARTOOLS_SELF_THROTTLE == 8, "edgar/httprequests.py, verified 2026-09-09"
    assert EDGARTOOLS_SELF_THROTTLE < 10, "the SEC's ceiling, regardless of machine count"
    lim = declare.lookup("edgar").rate_limit
    assert isinstance(lim, PerSecond) and lim.n == 8
    assert PerSecond.COOLDOWN_S == 600.0


def test_ccxt_declares_the_per_instance_shape():
    lim = declare.lookup("ccxt").rate_limit
    assert isinstance(lim, PerInstanceDelay)
    assert "per exchange instance" in lim.describe()


# ------------------------------------------------------------------- no shared key
_CRED = re.compile(r"(?i)(api[_-]?key|apikey|secret|password|passwd|token|bearer|"
                   r"credential|private[_-]?key)")
_KEY_SHAPED = re.compile(r"['\"][0-9a-f]{32}['\"]")


@pytest.mark.parametrize("path", sorted(PKG.rglob("*.py")), ids=lambda p: p.name)
def test_no_function_in_the_data_layer_accepts_a_credential(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a = node.args
        names = [x.arg for x in (a.posonlyargs + a.args + a.kwonlyargs)]
        names += [x.arg for x in (a.vararg, a.kwarg) if x]
        bad = [n for n in names if _CRED.search(n)]
        assert not bad, f"{path.name}:{node.lineno} {node.name}() takes {bad}"


@pytest.mark.parametrize("path", sorted(PKG.rglob("*.py")), ids=lambda p: p.name)
def test_no_source_file_contains_a_key_shaped_literal(path):
    text = path.read_text(encoding="utf-8")
    assert not _KEY_SHAPED.search(text), f"{path.name} contains a key-shaped literal"


def test_a_credential_can_only_come_from_the_declared_environment_variable(monkeypatch):
    decl = declare.lookup("fred")
    assert decl.key_sharing == "byok-required"
    monkeypatch.delenv(decl.key_env_var, raising=False)
    with pytest.raises(RuntimeError, match=decl.key_env_var):
        declare.credential(decl)

    monkeypatch.setenv(decl.key_env_var, "0123456789abcdef0123456789abcdef")
    assert declare.credential(decl) == "0123456789abcdef0123456789abcdef"

    # and it is not kept anywhere on the adapter object
    adapter = get("fred")
    assert "0123456789abcdef0123456789abcdef" not in repr(vars(adapter))


def test_an_adapter_refuses_a_credential_passed_as_an_argument():
    for kw in ({"api_key": "x"}, {"token": "x"}, {"password": "x"}, {"secret": "x"}):
        with pytest.raises(TypeError, match="cannot be passed"):
            get("fred", **kw)
    assert declare.credential(declare.lookup("yfinance")) is None


def test_a_declaration_cannot_hold_a_key():
    assert "key_env_var" in declare.field_names()
    assert not any(_CRED.search(f) and f != "key_env_var" and f != "key_sharing"
                   for f in declare.field_names())


# ------------------------------------------------------------------ pure normalisers
def test_yfinance_normalises_all_three_column_layouts():
    from fin_skills.data.adapters.yfinance import _normalise

    idx = pd.bdate_range("2024-01-02", periods=3)
    flat = pd.DataFrame({"Open": [1.0, 2, 3], "High": [1.0, 2, 3], "Low": [1.0, 2, 3],
                         "Close": [1.0, 2, 3], "Volume": [10, 20, 30]}, index=idx)
    one = _normalise(flat, ["AAPL"])
    assert list(one.columns.names) == ["field", "ticker"]
    assert one[("volume", "AAPL")].dtype == np.dtype("int64")
    assert ("adj_close", "AAPL") not in one.columns

    wide = pd.concat({"Close": flat[["Close"]].rename(columns={"Close": "AAPL"}),
                      "Volume": flat[["Volume"]].rename(columns={"Volume": "AAPL"})},
                     axis=1)
    assert ("close", "AAPL") in _normalise(wide, ["AAPL"]).columns

    swapped = pd.concat({"AAPL": flat}, axis=1)          # group_by="ticker"
    assert ("close", "AAPL") in _normalise(swapped, ["AAPL"]).columns

    with pytest.raises(ValueError, match="cannot be attributed"):
        _normalise(flat, ["AAPL", "MSFT"])
    with pytest.raises(ValueError, match="no rows"):
        _normalise(pd.DataFrame(), ["AAPL"])


def test_yfinance_refuses_a_caching_session_and_an_unstated_intraday_zone():
    with pytest.raises(TypeError, match="requests_cache"):
        get("yfinance", session=object())
    a = get("yfinance")
    with pytest.raises(ValueError, match="explicit tz"):
        a.bars("AAPL", "2024-01-01", "2024-01-05", interval="5m")
    with pytest.raises(ValueError, match="not one of"):
        a.bars("AAPL", "2024-01-01", "2024-01-05", interval="7s")
    with pytest.raises(ValueError, match="readjust"):
        a.bars("AAPL", "2024-01-01", "2024-01-05", adjustment=D.Adjustment.BACK)


def test_akshare_normalises_the_chinese_headers_and_refuses_what_it_cannot_do():
    from fin_skills.data.adapters.akshare import _COLUMNS, _normalise

    raw = pd.DataFrame({"日期": ["2024-01-02", "2024-01-03"], "开盘": [10.0, 11.0],
                        "最高": [10.5, 11.5], "最低": [9.5, 10.5],
                        "收盘": [10.2, 11.2], "成交量": [100, 200]})
    out = _normalise(raw, "600519")
    assert ("close", "600519") in out.columns
    assert out[("volume", "600519")].dtype == np.dtype("int64")
    assert "日期" in _COLUMNS

    a = get("akshare")
    with pytest.raises(NotImplementedError, match="no date parameter"):
        a.universe("HS300", "2020-01-01")
    with pytest.raises(NotImplementedError, match="announcement date"):
        a.fundamentals(["600519"])
    # the academic-research-only framing is surfaced the first time the adapter is used
    from fin_skills.data.adapters import akshare as ak_mod
    ak_mod._notified = False
    with pytest.warns(UserWarning, match="academic research"):
        with pytest.raises(ValueError, match="daily endpoint"):
            a.bars("600519", "2024-01-01", "2024-02-01", interval="1h")


def test_ccxt_holds_one_instance_drops_the_unclosed_bar_and_uses_milliseconds():
    from fin_skills.data.adapters.ccxt import _quote, _to_frame, drop_unclosed

    rows = [[1704153600000, 1.0, 2.0, 0.5, 1.5, 100.0],
            [1704240000000, 1.5, 2.5, 1.0, 2.0, 200.0]]
    frame = _to_frame(rows, "BTC/USDT")
    assert str(frame.index.tz) == "UTC"
    assert frame.index[0] == pd.Timestamp("2024-01-02", tz="UTC"), "ms, not seconds"

    closed = drop_unclosed(frame, "1d", pd.Timestamp("2024-01-05", tz="UTC"))
    assert len(closed) == 2
    open_now = drop_unclosed(frame, "1d", pd.Timestamp("2024-01-03 12:00", tz="UTC"))
    assert len(open_now) == 1, "the final candle had not closed yet"

    assert _quote("BTC/USDT") == "USDT" and _quote("BTCUSDT") == "unknown"

    a = get("ccxt:kraken")
    assert a.venue == "kraken" and a._exchange is None
    with pytest.raises(ValueError, match="RAW is the only convention"):
        a.bars("BTC/USDT", "2024-01-01", "2024-02-01", adjustment=D.Adjustment.BACK)


def test_fred_flags_copyrighted_series_and_never_calls_the_broken_method():
    from fin_skills.data.adapters import fredapi as mod

    assert mod.copyright_flagged("Copyright, 2016, Chicago Board Options Exchange")
    assert not mod.copyright_flagged("Source: U.S. Bureau of Economic Analysis")
    assert not mod.copyright_flagged(None)
    called = {n.func.attr for n in ast.walk(_tree("fredapi.py"))
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "get_series_as_of_date" not in called, \
        "the method whose docstring lies is documented here and never called"
    assert "get_series_all_releases" in called

    releases = pd.DataFrame({"date": ["2013-10-01", "2013-10-01"],
                             "realtime_start": ["2014-01-30", "2014-02-28"],
                             "value": [17102.5, 17080.7]})
    frame = mod.releases_to_frame(releases, "GDP", sa_flag="SAAR")
    assert list(frame.columns) == list(MACRO_COLUMNS)
    assert frame["realtime_end"].isna().all(), "0.5.2 drops realtime_end; it is not invented"

    with pytest.raises(ValueError, match="DISPLAY ONLY"):
        get("fred").macro("GDP", vintages=False)


def test_edgar_requires_a_declared_identity_and_rolls_post_close_filings(monkeypatch):
    from fin_skills.data.adapters.edgar import IDENTITY_ENV, facts_to_frame

    monkeypatch.delenv(IDENTITY_ENV, raising=False)
    with pytest.raises(ValueError, match="User-Agent"):
        get("edgar")
    monkeypatch.setenv(IDENTITY_ENV, "A Name a@example.com")
    assert get("edgar").identity == "A Name a@example.com"

    units = [{"start": "2022-07-01", "end": "2022-09-30", "val": 1.0,
              "accn": "x-22-1", "form": "10-Q", "filed": "2022-11-03",
              "acceptanceDateTime": "2022-11-03T21:05:00Z"},          # 17:05 ET, post-close
             {"start": "2022-04-01", "end": "2022-06-30", "val": 2.0,
              "accn": "x-22-0", "form": "10-Q", "filed": "2022-08-04",
              "acceptanceDateTime": "2022-08-04T14:30:00Z"}]          # 10:30 ET, intraday
    frame = facts_to_frame(units, entity_id="0000320193", tag="Revenues")
    assert frame.loc[0, "available_at"] == pd.Timestamp("2022-11-04"), "post-close rolls"
    assert frame.loc[1, "available_at"] == pd.Timestamp("2022-08-04"), "intraday does not"
    assert not frame["is_amendment"].any()

    no_stamp = facts_to_frame([{**units[0], "acceptanceDateTime": None}],
                              entity_id="c", tag="t")
    assert pd.isna(no_stamp.loc[0, "acceptance_at"])
    assert no_stamp.loc[0, "available_at"] == pd.Timestamp("2022-11-03")


def test_edgar_refuses_the_current_snapshot_ticker_file(monkeypatch):
    monkeypatch.setenv("EDGAR_IDENTITY", "A Name a@example.com")
    with pytest.raises(NotImplementedError, match="CURRENT snapshot"):
        get("edgar").universe("us", "2018-01-01")


def test_an_unserved_method_says_why_instead_of_returning_an_empty_frame():
    a = get("yfinance")
    with pytest.raises(NotImplementedError, match="point-in-time"):
        a.fundamentals(["AAPL"])
    with pytest.raises(NotImplementedError, match="macro"):
        a.macro(["GDP"])
    with pytest.raises(KeyError, match="no adapter"):
        get("bloomberg")


# ------------------------------------------------------------------- no data ships
def test_the_package_contains_code_only():
    """Adapters ship code. No CSV in the wheel, no bundled parquet, no mirror this
    project controls, and no "sample data" that is really a redistribution."""
    shipped = [p for p in PKG.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    assert shipped, "the package directory must exist"
    assert all(p.suffix == ".py" for p in shipped), \
        [str(p.relative_to(PKG)) for p in shipped if p.suffix != ".py"]

    pyproject = (PKG.parent.parent / "pyproject.toml")
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8")
        assert "data/**" not in text and "fin_skills/data" not in text, \
            "nothing under data/ may be declared as package-data"


def test_no_module_reads_from_a_mirror_this_project_controls():
    for path in PKG.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for host in ("raw.githubusercontent.com", "huggingface.co", "s3.amazonaws.com"):
            assert host not in text, f"{path.name} reaches for {host}"


# --------------------------------------------------------------------------- the CLI
def test_the_adapters_command_prints_the_table_and_the_warnings(capsys):
    from fin_skills.data.__main__ import main

    assert main(["adapters"]) == 0
    out = capsys.readouterr().out
    assert out.isascii(), "the CLI must survive a stock Windows console"
    for name in ("yfinance", "akshare", "ccxt", "fred", "edgar"):
        assert name in out
    assert "byok-required" in out and "UPPER BOUND" in out
    assert "FRED_API_KEY" in out

    assert main(["adapters", "--name", "fred"]) == 0
    one = capsys.readouterr().out
    assert "yfinance" not in one and "2 requests per second" in one


def test_the_manifest_command_reads_a_cache(capsys, tmp_path, monkeypatch):
    from _data_fixtures import clean_bars
    from fin_skills.data.__main__ import main

    root = tmp_path / "cache"
    bars = clean_bars(n_names=2, years=2, n_dead=0)
    D.Cache(root).put(bars, bars.provenance)

    assert main(["manifest", "--root", str(root), "--verify"]) == 0
    out = capsys.readouterr().out
    assert "yfinance" in out and "verified" in out and out.isascii()

    assert main(["manifest", "--root", str(tmp_path / "nope")]) == 2
    monkeypatch.delenv("FIN_SKILLS_DATA_CACHE", raising=False)
    assert main(["manifest"]) == 2, "no root and no env var: say so, do not guess one"
