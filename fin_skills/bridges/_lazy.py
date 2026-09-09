"""The single import helper, the audited LICENCE table, and the enforcement helpers.

Nothing in `fin_skills.bridges` imports a bridged library at module scope. Every vendor
import goes through `need()`, which:

  * raises `MissingLibrary` naming the exact `pip install` when the library is absent -
    one message, not an ImportError from three frames down;
  * warns once per process for a library this MIT repo may NOT list as a dependency
    (`backtesting.py` is AGPL-3.0, `vectorbt` is Apache-2.0 + Commons Clause). Both are
    optional RUNTIME imports and appear in no dependency list anywhere - see
    `licence_audit()` and `tests/test_bridges.py::test_copyleft_libraries_are_not_dependencies`;
  * reads the library's own `__version__` at CALL time and records it, with the audited
    licence, into the `Provenance` of whatever the bridge returns. akshare-style history
    purges are why a pin is not evidence and the version is read live.

Two enforcement helpers live here because three bridges need the same proof:

  * `prove_causal`  - the repo's own `assert_causal` guard: output before k must not move
    when only rows >= k are perturbed.
  * `prove_lagged`  - STRICTER, and the one an `already_lagged=True` claim is tested with:
    the value at bar t must be a function of bars strictly BEFORE t. It is `assert_causal`
    run on `fn(df).shift(-1)`, so the guard's own check of "rows before k" becomes "the
    value AT k does not depend on bar k". A caller who cannot supply the function that
    produced the signal cannot prove the claim, and the claim is refused: "could not prove
    it is lagged" and "it is not lagged" have the same consequence, which is the posture
    `fin_skills.core.paper_account_guard` already takes on live accounts.
"""
from __future__ import annotations

import importlib
import importlib.util
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

import pandas as pd

from fin_skills.api import get


class MissingLibrary(ImportError):
    """A bridged library is not installed. The message names the pip install."""


class LicenceWarning(UserWarning):
    """A library this MIT repo may not depend on was imported at runtime anyway."""


class NotLaggedError(ValueError):
    """An `already_lagged=True` claim that could not be proven, or was proven false."""


class NonCausalError(ValueError):
    """A function whose output before k moved when only rows >= k were perturbed."""


# --------------------------------------------------------------------- the audit table
@dataclass(frozen=True)
class Library:
    """One bridged library, and whether an MIT repo may depend on it.

    `verdict` is one of:
      'extra'        - permissive (MIT/BSD/Apache); may be a pyproject extra.
      'runtime-only' - copyleft or a non-OSI addendum; optional runtime import with a
                       printed warning, never install_requires, never an extra.
    """

    bridge: str          # the module in this package that imports it
    pip: str             # what you type after `pip install`
    module: str          # what you type after `import` (they differ 7 times out of 14)
    licence: str         # SPDX, or SPDX plus the addendum that is not SPDX
    verified_at: str     # where it was read
    verified_on: str     # ISO date it was read
    verdict: str         # 'extra' | 'runtime-only'
    note: str = ""


LIBRARIES: tuple[Library, ...] = (
    Library("vectorbt", "vectorbt", "vectorbt", "Apache-2.0 + Commons Clause",
            "repo LICENSE.md", "2026-09-03", "runtime-only",
            "not OSI open source; Commons Clause removes the right to SELL a product or "
            "service whose value derives substantially from it. PyPI declares no licence "
            "at all, so a metadata-only audit sees nothing"),
    Library("backtesting_py", "backtesting", "backtesting", "AGPL-3.0",
            "repo LICENSE", "2026-09-03", "runtime-only",
            "network copyleft: serving this over a dashboard, API or MCP owes your users "
            "the source of the combined work"),
    Library("zipline", "zipline-reloaded", "zipline", "Apache-2.0",
            "repo LICENSE", "2026-09-03", "extra",
            "pip name != import name; wheels cp310-cp313 only, no Linux aarch64"),
    Library("qlib", "pyqlib", "qlib", "MIT",
            "PyPI classifier + GitHub SPDX", "2026-09-04", "extra",
            "`pip install qlib` is the WRONG package (an abandoned 2018 upload); PyPI "
            "0.9.7 lags the repo by ~11 months"),
    Library("optimizers", "PyPortfolioOpt", "pypfopt", "MIT",
            "repo LICENSE", "2026-09-04", "extra",
            "requires_python is unset on PyPI - pin a floor yourself"),
    Library("optimizers", "skfolio", "skfolio", "BSD-3-Clause",
            "repo LICENSE", "2026-09-04", "extra",
            "1.0.0 landed 2026-08-23; every pre-1.0 snippet predates the stability commitment"),
    Library("optimizers", "Riskfolio-Lib", "riskfolio", "BSD-3-Clause",
            "repo LICENSE", "2026-09-04", "extra", "pip name != import name"),
    Library("quantlib", "QuantLib", "QuantLib", "BSD-3-Clause",
            "repo LICENSE (GitHub reports NOASSERTION - the custom BSD-derived text "
            "confuses the detector)", "2026-09-03", "extra",
            "1.43 ships cp39-abi3 wheels but NO sdist: an unsupported platform cannot build it"),
    Library("execution", "ccxt", "ccxt", "MIT",
            "GitHub license.spdx_id", "2026-09-04", "extra",
            "PyPI info.license is null with no classifier - do not read PyPI metadata as "
            "the answer here. CCXT Pro was merged into the MIT package at v1.95 (2022)"),
    Library("execution", "ib_async", "ib_async", "BSD-2-Clause",
            "repo LICENSE", "2026-09-03", "extra",
            "main's last commit is 2025-12-06; ib_insync is archived and must not be forked"),
    Library("execution", "alpaca-py", "alpaca", "Apache-2.0",
            "repo LICENSE", "2026-09-03", "extra",
            "the legacy alpaca-trade-api is the dangerous one (get_base_url() returns the "
            "LIVE host by default) and is deliberately not bridged"),
    Library("reporting", "alphalens-reloaded", "alphalens", "Apache-2.0",
            "repo LICENSE", "2026-09-04", "extra",
            "pins pandas<3.0,>=1.5.0 - a hard conflict on a pandas 3.x environment, which "
            "is a packaging reason to keep it optional on top of the licence reason"),
    Library("reporting", "quantstats", "quantstats", "Apache-2.0",
            "repo LICENSE", "2026-09-04", "extra", ""),
    Library("reporting", "pyfolio-reloaded", "pyfolio", "Apache-2.0",
            "PyPI info.license + classifier, 0.9.9 uploaded 2025-06-02", "2026-09-09",
            "extra",
            "low velocity; it brings empyrical-reloaded, which carries the PER-PERIOD "
            "risk_free trap this bridge converts for"),
)

_BY_PIP: dict[str, Library] = {lib.pip.lower(): lib for lib in LIBRARIES}
_BY_MODULE: dict[str, Library] = {lib.module.lower(): lib for lib in LIBRARIES}


def info(name: str) -> Library:
    """The audit row for a pip name or an import name."""
    key = str(name).strip().lower()
    lib = _BY_PIP.get(key) or _BY_MODULE.get(key)
    if lib is None:
        raise KeyError(f"{name!r} is not a bridged library; "
                       f"known: {sorted(x.pip for x in LIBRARIES)}")
    return lib


# ------------------------------------------------------------------------- importing
def find(name: str) -> bool:
    """Is the library importable? Uses find_spec, so it imports NOTHING."""
    lib = info(name)
    try:
        return importlib.util.find_spec(lib.module) is not None
    except (ImportError, ValueError):
        return False


_WARNED: set[str] = set()


def need(name: str, *, why: str = "") -> Any:
    """Import a bridged library or raise `MissingLibrary` naming the pip install.

    Warns once per process for a 'runtime-only' licence. The warning is the whole
    reason these two are importable at all and absent from every dependency list.
    """
    lib = info(name)
    if not find(name):
        extra = f" ({why})" if why else ""
        raise MissingLibrary(
            f"{lib.pip} is not installed but this bridge needs it{extra}: "
            f"pip install {lib.pip}   [then `import {lib.module}` - "
            f"licence {lib.licence}, verdict '{lib.verdict}']")
    if lib.verdict == "runtime-only" and lib.pip not in _WARNED:
        _WARNED.add(lib.pip)
        warnings.warn(
            f"{lib.pip} is {lib.licence}. fin-skills is MIT and lists it NOWHERE - not in "
            f"install_requires, not as an extra. You imported it yourself and its terms "
            f"are yours: {lib.note}. fin_skills.bridges.licence_audit() has the row.",
            LicenceWarning, stacklevel=3)
    return importlib.import_module(lib.module)


def version_of(module: Any, name: str = "") -> str:
    """The library's own __version__, read at call time; metadata as the fallback."""
    v = getattr(module, "__version__", "") or getattr(module, "VERSION", "")
    if not v:
        try:
            import importlib.metadata as md
            v = md.version(name or getattr(module, "__name__", ""))
        except Exception:                                     # noqa: BLE001 - best effort
            v = "unknown"
    return str(v)


# ------------------------------------------------------------------------ provenance
@dataclass(frozen=True)
class Provenance:
    """What came back, from which library version, under which audited licence.

    Field names match `fin_skills.data.provenance.Provenance` where they overlap, so the
    two can be merged rather than reconciled once the data layer lands. Nothing here ever
    holds a credential: `request` carries the NORMALISED request only.
    """

    source: str                       # 'vectorbt' | 'ccxt:binance' | 'quantstats' | ...
    library_version: str = ""
    licence: str = ""
    licence_verified_on: str = ""
    retrieved_at: str = field(default_factory=lambda:
                              datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    request: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "library_version": self.library_version,
                "licence": self.licence, "licence_verified_on": self.licence_verified_on,
                "retrieved_at": self.retrieved_at, "request": dict(self.request),
                "notes": self.notes}


def provenance(name: str, module: Any = None, *, source: str = "", request: Any = None,
               notes: str = "") -> Provenance:
    """A Provenance row for a bridged library, with its version read live."""
    lib = info(name)
    return Provenance(source=source or lib.pip,
                      library_version=version_of(module, lib.module) if module else "",
                      licence=lib.licence, licence_verified_on=lib.verified_on,
                      request=dict(request or {}), notes=notes or lib.note)


# ------------------------------------------------------------------- causality proofs
def _causal_result(fn: Callable[[pd.DataFrame], Any], df: pd.DataFrame, k: int | None,
                   name: str):
    """Run the repo's assert_causal guard and return its GuardResult."""
    kw: dict[str, Any] = {"fn": fn, "df": df, "name": name}
    if k is not None:
        kw["k"] = int(k)
    return get("assert_causal").run(**kw)


def _raised(result) -> str:
    """The guard reports a function that blew up as a warning, not a leak. That is not
    a pass either: an untested claim is a refused claim."""
    for f in result.findings:
        if f.severity == "warning" and "raised" in f.message:
            return f.message
    return ""


def prove_causal(fn: Callable[[pd.DataFrame], Any], df: pd.DataFrame, *,
                 k: int | None = None, name: str = "indicator") -> None:
    """Raise `NonCausalError` unless fn's output before k is independent of rows >= k.

    The guard is `fin_skills.api.get("assert_causal")` - this is the check
    `backtesting.py` cannot do for you, because `Strategy.I` computes the indicator over
    the FULL series in init() and then slices it (lib-backtesting-py, "The trap that
    costs you money").
    """
    res = _causal_result(fn, df, k, name)
    why = _raised(res)
    if why:
        raise NonCausalError(f"{name}: causality could not be tested - {why}")
    if not res.passed:
        raise NonCausalError(
            f"{name} is NOT causal: " + "; ".join(str(f) for f in res.errors))


def prove_lagged(fn: Callable[[pd.DataFrame], Any], df: pd.DataFrame, *,
                 k: int | None = None, name: str = "signal") -> None:
    """Raise `NotLaggedError` unless fn's value at bar t uses only bars STRICTLY before t.

    Stricter than `prove_causal`, and the difference is the entire vectorbt/alphalens
    trap: a signal computed from close[t] IS causal, and filling it at close[t] is still
    same-bar execution. Implemented as assert_causal on `fn(df).shift(-1)`, so the guard's
    "no cell before k moved" becomes "the value AT k does not depend on bar k".
    """
    def shifted(d: pd.DataFrame):
        return fn(d).shift(-1)

    shifted.__name__ = f"{name}.shift(-1)"
    res = _causal_result(shifted, df, k, f"{name} (strict lag test)")
    why = _raised(res)
    if why:
        raise NotLaggedError(f"{name}: the already_lagged claim could not be tested - {why}")
    if not res.passed:
        raise NotLaggedError(
            f"already_lagged=True is FALSE for {name}: its value at bar t moves when bar t "
            "itself is perturbed, so it was computed from the bar it trades on. "
            + "; ".join(str(f) for f in res.errors))


def refuse_untestable_claim(what: str, how: str) -> None:
    """One message for 'you claimed it and gave me nothing to test the claim with'."""
    raise NotLaggedError(
        f"already_lagged=True was claimed for {what} but nothing was supplied to test it "
        f"with. {how} 'Could not prove it' and 'it is false' get the same verdict here, "
        f"the way fin_skills.core.paper_account_guard treats a live account.")


# -------------------------------------------------------------------------- misc
def periods_per_year(sessions: Any, default: int | None = None) -> int | None:
    """Rows per year from an engine `Sessions`, an int, a calendar name, or None.

    There is no silent 252. A bridge that cannot learn the annualisation leaves
    `periods_per_year` out of the Bundle rather than guessing it, because every guard
    downstream would then annualise against a number nobody declared.
    """
    if sessions is None:
        return default
    if isinstance(sessions, bool):
        raise TypeError("sessions must not be a bool")
    if isinstance(sessions, int):
        return int(sessions)
    ppy = getattr(sessions, "periods_per_year", None)
    if ppy is not None:
        return int(ppy)
    if isinstance(sessions, str):
        from fin_skills.api import conventions
        return int(conventions.annualization_factor(sessions))
    raise TypeError(f"cannot read periods_per_year from {type(sessions).__name__}; pass an "
                    f"int, an engine Sessions, or a calendar name")


__all__ = ["LIBRARIES", "LicenceWarning", "Library", "MissingLibrary", "NonCausalError",
           "NotLaggedError", "Provenance", "find", "info", "need", "periods_per_year",
           "prove_causal", "prove_lagged", "provenance", "refuse_untestable_claim",
           "version_of"]
