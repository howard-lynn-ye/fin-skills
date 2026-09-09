"""fin_skills.bridges: the registry, the licence audit, and the lazy-import contract.

These are the tests that run with NONE of the fourteen bridged libraries installed. The
per-bridge enforcement lives in test_bridges_enforcement.py and the read-only proof for
the execution bridge in test_bridges_execution.py.
"""
from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
import warnings

import pandas as pd
import pytest

from _helpers import PACKAGE_ROOT, REPO_ROOT

import fin_skills.bridges as bridges
from fin_skills.bridges import _lazy

#: the import names of every bridged library, for the "we imported none of them" test.
VENDOR_MODULES = tuple(lib.module for lib in _lazy.LIBRARIES)
BRIDGE_MODULES = ("vectorbt", "backtesting_py", "zipline", "qlib", "optimizers",
                  "quantlib", "execution", "reporting")


def _module(name: str):
    return importlib.import_module(f"fin_skills.bridges.{name}")


# ------------------------------------------------------------------ importing nothing
def test_bridges_import_nothing():
    """`import fin_skills.bridges` must not drag in one of the fourteen.

    Run in a subprocess: this process has almost certainly imported quantstats and
    QuantLib by now for the round-trip tests, so an in-process sys.modules check would
    pass or fail for the wrong reason.
    """
    code = (
        "import sys, json\n"
        "import fin_skills.bridges as b\n"
        f"vendors = {list(VENDOR_MODULES)!r}\n"
        "print(json.dumps([m for m in vendors if m in sys.modules]))\n"
    )
    env = {"PYTHONPATH": str(PACKAGE_ROOT.parent), "SYSTEMROOT": "",
           "PATH": "", "PYTHONIOENCODING": "utf-8"}
    import os
    full = dict(os.environ)
    full.update({k: v for k, v in env.items() if v})
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=full, timeout=180)
    assert proc.returncode == 0, proc.stderr
    leaked = proc.stdout.strip().splitlines()[-1]
    assert leaked == "[]", f"importing fin_skills.bridges pulled in {leaked}"


def test_available_imports_nothing_and_covers_every_library():
    got = bridges.available()
    assert set(got) == {lib.pip for lib in _lazy.LIBRARIES}
    assert all(isinstance(v, bool) for v in got.values())


def test_available_and_audit_work_with_nothing_installed(monkeypatch):
    """The two functions that have to work on a bare numpy/pandas machine."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda *a, **k: None)
    assert set(bridges.available().values()) == {False}
    audit = bridges.licence_audit()
    assert len(audit) == len(_lazy.LIBRARIES)
    assert not audit["installed"].any()
    assert isinstance(bridges.audit_text(), str)


def test_find_spec_failure_is_not_an_import_error(monkeypatch):
    def boom(*_a, **_k):
        raise ValueError("__spec__ is None")

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    assert set(bridges.available().values()) == {False}


# ------------------------------------------------------------------- the licence audit
def test_licence_audit_complete():
    audit = bridges.licence_audit()
    assert list(audit.columns) == ["bridge", "pip", "import", "licence", "verdict",
                                   "installed", "verified_at", "verified_on", "note"]
    assert len(audit) == 14
    for col in ("licence", "verdict", "verified_at", "verified_on"):
        assert audit[col].astype(str).str.strip().ne("").all(), f"{col} has a blank"
        assert not audit[col].astype(str).str.contains("UNKNOWN", case=False).any()
    assert set(audit["verdict"]) == {"extra", "runtime-only"}
    # every verification carries a real date, not a placeholder
    dates = pd.to_datetime(audit["verified_on"])
    assert dates.notna().all()
    assert (dates >= pd.Timestamp("2026-01-01")).all()


@pytest.mark.parametrize("name", BRIDGE_MODULES)
def test_every_bridge_declares_its_library(name):
    mod = _module(name)
    for attr in ("LIBRARY", "LICENCE", "VERIFIED_ON"):
        assert getattr(mod, attr, ""), f"{name} does not declare {attr}"
    row = _lazy.info(mod.LIBRARY)
    assert mod.LICENCE == row.licence
    assert mod.VERIFIED_ON == row.verified_on
    audit = bridges.licence_audit()
    assert row.pip in set(audit["pip"]), f"{mod.LIBRARY} has no audit row"


def test_every_audited_library_belongs_to_a_bridge():
    assert {lib.bridge for lib in _lazy.LIBRARIES} <= set(BRIDGE_MODULES)


@pytest.mark.skipif(not (REPO_ROOT / "pyproject.toml").is_file(),
                    reason="pyproject.toml not present in this checkout")
def test_copyleft_libraries_are_not_dependencies():
    """backtesting.py (AGPL-3.0) and vectorbt (Commons Clause) may appear in NO list.

    The licence audit says 'runtime-only' for both. This is the test that makes the
    verdict binding: they are optional imports with a printed warning, never
    install_requires and never an extra, exactly as this repo treats openbb.
    """
    import tomllib

    cfg = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = cfg.get("project", {})
    lists = {"dependencies": project.get("dependencies", [])}
    for name, extra in (project.get("optional-dependencies") or {}).items():
        lists[f"optional-dependencies.{name}"] = extra
    banned = [lib.pip.lower() for lib in _lazy.LIBRARIES if lib.verdict == "runtime-only"]
    assert banned, "the audit lists no runtime-only library - the test would be vacuous"
    for where, items in lists.items():
        for item in items:
            head = str(item).split(">")[0].split("<")[0].split("=")[0].split("[")[0]
            assert head.strip().lower() not in banned, \
                f"{item!r} is in [project.{where}] and its licence forbids it"


@pytest.mark.skipif(not (REPO_ROOT / "ci").is_dir(), reason="ci/ not present")
def test_copyleft_libraries_are_not_installed_by_ci():
    banned = [lib.pip.lower() for lib in _lazy.LIBRARIES if lib.verdict == "runtime-only"]
    for path in sorted((REPO_ROOT / "ci").glob("*.yml")):
        text = path.read_text(encoding="utf-8").lower()
        for line in text.splitlines():
            if "pip install" not in line:
                continue
            for pip in banned:
                assert f" {pip}" not in line, f"{path.name} installs {pip}: {line.strip()}"


# ------------------------------------------------------------------ the import helper
@pytest.mark.parametrize("lib", _lazy.LIBRARIES, ids=[x.pip for x in _lazy.LIBRARIES])
def test_need_names_the_pip_install_when_absent(lib, monkeypatch):
    monkeypatch.setattr(_lazy, "find", lambda _name: False)
    with pytest.raises(_lazy.MissingLibrary) as exc:
        _lazy.need(lib.pip)
    msg = str(exc.value)
    assert f"pip install {lib.pip}" in msg
    assert lib.module in msg
    assert lib.licence in msg


def test_need_rejects_a_library_that_is_not_bridged():
    with pytest.raises(KeyError):
        _lazy.need("openbb")


def test_runtime_only_import_warns_once(monkeypatch):
    """The printed warning is the whole reason these two are importable at all."""
    monkeypatch.setattr(_lazy, "_WARNED", set())
    monkeypatch.setattr(_lazy, "find", lambda _name: True)
    monkeypatch.setattr(importlib, "import_module", lambda _name: object())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _lazy.need("backtesting")
        _lazy.need("backtesting")
    assert len(caught) == 1, "the warning should fire once per process, not per call"
    text = str(caught[0].message)
    assert caught[0].category is _lazy.LicenceWarning
    assert "AGPL-3.0" in text and "lists it NOWHERE" in text


def test_permissive_import_does_not_warn(monkeypatch):
    monkeypatch.setattr(_lazy, "_WARNED", set())
    monkeypatch.setattr(_lazy, "find", lambda _name: True)
    monkeypatch.setattr(importlib, "import_module", lambda _name: object())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _lazy.need("quantstats")
    assert [w for w in caught if w.category is _lazy.LicenceWarning] == []


def test_version_is_read_at_call_time(monkeypatch):
    class Fake:
        __version__ = "9.9.9"
        __name__ = "fake"

    assert _lazy.version_of(Fake(), "fake") == "9.9.9"
    prov = _lazy.provenance("quantstats", Fake())
    assert prov.library_version == "9.9.9"
    assert prov.licence == _lazy.info("quantstats").licence
    assert prov.licence_verified_on == _lazy.info("quantstats").verified_on


def test_provenance_never_carries_a_credential():
    prov = _lazy.provenance("ccxt", request={"symbols": ["BTC/USDT"], "since": "2026-01-01"})
    text = repr(prov.as_dict()).lower()
    for word in ("apikey", "api_key", "secret", "password", "token"):
        assert word not in text


# --------------------------------------------------------------- periods_per_year
def test_periods_per_year_never_guesses():
    assert _lazy.periods_per_year(None) is None
    assert _lazy.periods_per_year(None, default=252) == 252
    assert _lazy.periods_per_year(365) == 365
    assert _lazy.periods_per_year("crypto") == 365

    class Sessions:
        periods_per_year = 260

    assert _lazy.periods_per_year(Sessions()) == 260
    with pytest.raises(TypeError):
        _lazy.periods_per_year(object())


def test_audit_text_is_ascii_and_names_both_runtime_only_libraries():
    text = bridges.audit_text()
    assert all(ord(ch) < 128 for ch in text)
    assert "runtime-only" in text
    for lib in _lazy.LIBRARIES:
        if lib.verdict == "runtime-only":
            assert lib.pip in text and lib.licence in text
