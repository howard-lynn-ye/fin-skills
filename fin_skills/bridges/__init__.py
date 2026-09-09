"""fin_skills.bridges - adapters from the guards to the tools people already use.

A bridge is a place to put a trap, not a place to hide one. Each module does three things
and nothing else:

  1. import its library LAZILY - `import fin_skills.bridges` imports none of the fourteen;
  2. convert between `fin_skills.api.Bundle` and that library's native object;
  3. enforce that library's one documented failure on the way through.

Where enforcement would change a number you asked for, the bridge REFUSES rather than
correcting silently. A bridge that quietly fixes a lie makes the lie unfalsifiable.

    from fin_skills.bridges import available, licence_audit
    available()                       # {'vectorbt': False, 'quantstats': True, ...}
    print(licence_audit().to_string(index=False))
    python -m fin_skills.bridges audit

The traps, each documented in this repo's own skill for that library:

| bridge            | what it refuses to let past                                        |
|-------------------|--------------------------------------------------------------------|
| `vectorbt`        | `Order.price` defaults to `np.inf` = the signal bar's own close     |
| `backtesting_py`  | `Strategy.I` computes over the FULL series; `trade_on_close`=Close[-2] |
| `zipline`         | asset lifetimes are worthless on a survivor-only universe; $0 min   |
| `qlib`            | the default normalizer is fit over a span containing the test set   |
| `optimizers`      | prices where returns are required; HRP linkage; Riskfolio state; rf |
| `quantlib`        | `evaluationDate` is a global and returns NPV exactly 0.0 past expiry |
| `execution`       | READ-ONLY by construction; the account must PROVE it is paper       |
| `reporting`       | alphalens does not lag your factor; `cagr(rf=)` discards rf         |

`licence_audit()` is a function rather than a paragraph because two of the fourteen fall on
the wrong side of an MIT repo's dependency line: `backtesting.py` is AGPL-3.0 and
`vectorbt` is Apache-2.0 + Commons Clause (not OSI, and PyPI declares no licence at all).
Both are optional RUNTIME imports with a printed warning; neither appears in
`pyproject.toml` under `dependencies` or any extra, and a test asserts it.

numpy and pandas are the only hard dependencies here.
"""
from __future__ import annotations

import importlib.util

import pandas as pd

from fin_skills.bridges import (backtesting_py, execution, optimizers, qlib, quantlib,
                                reporting, vectorbt, zipline)
from fin_skills.bridges._lazy import (LIBRARIES, LicenceWarning, Library, MissingLibrary,
                                      NonCausalError, NotLaggedError, Provenance)
from fin_skills.bridges.backtesting_py import from_backtesting, to_backtesting
from fin_skills.bridges.execution import ExecutionRecord, read_execution, read_ohlcv
from fin_skills.bridges.optimizers import OptimizerInput, optimize, weights_to_bundle
from fin_skills.bridges.qlib import from_qlib, to_qlib_handler
from fin_skills.bridges.quantlib import PricingResult, evaluation_date, price
from fin_skills.bridges.reporting import Tearsheet, tearsheet, to_alphalens
from fin_skills.bridges.vectorbt import from_vectorbt, to_vectorbt
from fin_skills.bridges.zipline import (from_zipline, to_zipline_assets,
                                        to_zipline_commission)

#: bridge module -> the library it needs, in the order licence_audit() prints them.
BRIDGES: tuple[str, ...] = ("vectorbt", "backtesting_py", "zipline", "qlib", "optimizers",
                            "quantlib", "execution", "reporting")


def _importable(module: str) -> bool:
    """find_spec only. This function must never import a bridged library."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def available() -> dict[str, bool]:
    """{pip name: is it importable}. Imports nothing; uses importlib.util.find_spec.

    Keyed by the PIP name, because SEVEN of the fourteen differ from the import name
    (`zipline-reloaded`->`zipline`, `pyqlib`->`qlib`, `PyPortfolioOpt`->`pypfopt`,
    `Riskfolio-Lib`->`riskfolio`, `alpaca-py`->`alpaca`,
    `alphalens-reloaded`->`alphalens`, `pyfolio-reloaded`->`pyfolio`) and the pip name is
    the one you have to type.
    """
    return {lib.pip: _importable(lib.module) for lib in LIBRARIES}


def licence_audit() -> pd.DataFrame:
    """One row per bridged library: can an MIT repo depend on it, and who says so.

    Columns: bridge, pip, import, licence, verdict, installed, verified_at, verified_on,
    note. `verdict` is 'extra' (permissive - MIT/BSD/Apache, may be a pyproject extra) or
    'runtime-only' (copyleft or a non-OSI addendum - optional runtime import with a printed
    warning, never install_requires, never an extra).

    Imports nothing: `installed` comes from find_spec.
    """
    inst = {lib.pip: _importable(lib.module) for lib in LIBRARIES}
    rows = [{"bridge": lib.bridge, "pip": lib.pip, "import": lib.module,
             "licence": lib.licence, "verdict": lib.verdict, "installed": inst[lib.pip],
             "verified_at": lib.verified_at, "verified_on": lib.verified_on,
             "note": lib.note}
            for lib in LIBRARIES]
    return pd.DataFrame(rows, columns=["bridge", "pip", "import", "licence", "verdict",
                                       "installed", "verified_at", "verified_on", "note"])


def audit_text(width: int = 96) -> str:
    """The audit as a printable ASCII table plus the two verdicts that constrain packaging.

    ASCII only and wrapped, not truncated: a note that says why a library may not be a
    dependency is not worth printing with the reason cut off.
    """
    import textwrap

    df = licence_audit()
    cols = ["bridge", "pip", "licence", "verdict", "installed"]
    lines = [df[cols].to_string(index=False), "",
             "runtime-only = optional import with a printed warning; NEVER a dependency "
             "or an extra:"]
    body: list[tuple[str, str]] = [(f"{lib.pip} ({lib.licence})", lib.note)
                                   for lib in LIBRARIES if lib.verdict == "runtime-only"]
    notes: list[tuple[str, str]] = [(lib.pip, lib.note) for lib in LIBRARIES
                                    if lib.note and lib.verdict != "runtime-only"]
    for group, header in ((body, None), (notes, "notes:")):
        if header:
            lines += ["", header]
        for head, note in group:
            lines += textwrap.wrap(f"{head} - {note}", width=width, initial_indent="  ",
                                   subsequent_indent="      ")
    return "\n".join(lines).encode("ascii", "replace").decode("ascii")


__all__ = [
    "BRIDGES", "ExecutionRecord", "LIBRARIES", "LicenceWarning", "Library",
    "MissingLibrary", "NonCausalError", "NotLaggedError", "OptimizerInput", "PricingResult",
    "Provenance", "Tearsheet", "audit_text", "available", "backtesting_py",
    "evaluation_date", "execution", "from_backtesting", "from_qlib", "from_vectorbt",
    "from_zipline", "licence_audit", "optimize", "optimizers", "price", "qlib", "quantlib",
    "read_execution", "read_ohlcv", "reporting", "tearsheet", "to_alphalens",
    "to_backtesting", "to_qlib_handler", "to_vectorbt", "to_zipline_assets",
    "to_zipline_commission", "vectorbt", "weights_to_bundle", "zipline",
]
