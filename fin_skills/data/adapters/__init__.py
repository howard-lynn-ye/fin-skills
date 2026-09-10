"""Eight thin adapters. An adapter normalises and DECLARES; it does not clean, fill,
resample or repair.

`import fin_skills.data` imports all eight of these modules, so `adapters()` can print the
whole table, and imports **none** of yfinance, akshare, ccxt, fredapi, edgartools, tiingo,
alpha-vantage or requests. Every vendor import happens inside the method that needs it,
through `require()`, which names the pip install in its error. The tests prove it by
blocking those names on `sys.meta_path` and importing the package anyway.

The other rule is about credentials: no constructor, no method and no helper in this
package accepts an API key, a token or a password. `get()` rejects such a keyword by name.
A credential is read from the environment variable the Declaration names, at the moment it
is used, and is never stored, logged, printed or written into a Provenance. FRED's own
API-key page is why: "All users of an application shall use their own API key."
"""
from __future__ import annotations

import importlib
import inspect
import re
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from fin_skills.data import declare
from fin_skills.data.schema import Adjustment, Bars, Fundamentals, Macro

#: adapter name -> the module inside this package that defines it
MODULES: dict[str, str] = {
    "yfinance": "yfinance",
    "akshare": "akshare",
    "ccxt": "ccxt",
    "fred": "fredapi",
    "edgar": "edgar",
    "tiingo": "tiingo",
    "alphavantage": "alphavantage",
    "stooq": "stooq",
}

_CREDENTIAL_KW = re.compile(
    r"(?i)(api[_-]?key|apikey|secret|password|passwd|token|bearer|credential|"
    r"authorization|private[_-]?key)")


@runtime_checkable
class Adapter(Protocol):
    """What every adapter offers. Methods a source cannot serve raise NotImplementedError
    with the reason, rather than returning an empty frame that looks like an answer."""

    decl: declare.Declaration

    def bars(self, symbols, start, end, *, interval: str = "1d",
             adjustment: Adjustment | None = None, **kw) -> Bars: ...
    def fundamentals(self, entities, tags=(), **kw) -> Fundamentals: ...
    def macro(self, series_ids, *, vintages: bool = True, **kw) -> Macro: ...
    def universe(self, market: str, as_of, *,
                 include_delisted: bool = False) -> pd.DataFrame: ...
    def actions(self, symbols, start, end) -> pd.DataFrame: ...


def require(module: str, pip_name: str | None = None):
    """Import a vendor library HERE, with a message naming the pip install if it is gone.

    Never called at module import. `import fin_skills.data` must work with none of the
    eight installed, and a test asserts it does.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        pip = pip_name or module
        raise ImportError(
            f"this adapter needs {module!r}, which is not installed:\n"
            f"    pip install {pip}\n"
            f"fin_skills.data itself depends only on numpy and pandas; every vendor "
            f"client is optional and imported at the moment it is used.") from exc


def library_version(mod: Any, default: str = "unknown") -> str:
    """The vendor client's own __version__, read at CALL time.

    Not from a lockfile: akshare deletes its own PyPI history (219 releases listed on
    2026-09-09, oldest 1.16.72 of 2025-04-05, against 1,437 in the upstream changelog),
    so a pin is not evidence of what actually ran.
    """
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(mod, attr, None)
        if isinstance(v, str) and v:
            return v
    try:
        from importlib.metadata import version                        # noqa: PLC0415
        return version(getattr(mod, "__name__", str(mod)))
    except Exception:                                                 # pragma: no cover
        return default


class Base:
    """Shared plumbing: the declaration, the limiter, and `replay()` for `Cache.refetch`.

    Every unimplemented method raises with the reason the SOURCE cannot serve it, which is
    information. Returning an empty frame instead is how "no data" ends up in a parquet
    file indistinguishable from "no such thing".
    """

    decl: declare.Declaration

    def __init__(self, **client_kw: Any) -> None:
        bad = [k for k in client_kw if _CREDENTIAL_KW.search(k)]
        if bad:
            raise TypeError(
                f"{sorted(bad)} cannot be passed to an adapter. Credentials are read from "
                f"the environment variable the Declaration names "
                f"(${self.decl.key_env_var or 'none'}) at the moment they are used, and "
                f"this library never stores, logs or serialises one. There is deliberately "
                f"no argument through which a key could travel.")
        self.client_kw = dict(client_kw)
        self.limiter = self.decl.rate_limit

    # ------------------------------------------------------------------ not served
    def _unsupported(self, what: str, why: str) -> "NotImplementedError":
        return NotImplementedError(f"{self.decl.name} does not provide {what}: {why}")

    def bars(self, symbols, start, end, *, interval: str = "1d",
             adjustment: Adjustment | None = None, **kw) -> Bars:
        raise self._unsupported("bars", "this source is not a price source")

    def fundamentals(self, entities, tags=(), **kw) -> Fundamentals:
        raise self._unsupported("fundamentals",
                                "no point-in-time statement data is available")

    def macro(self, series_ids, *, vintages: bool = True, **kw) -> Macro:
        raise self._unsupported("macro", "no macro series are available")

    def universe(self, market: str, as_of, *,
                 include_delisted: bool = False) -> pd.DataFrame:
        raise self._unsupported("universe", "no membership table is available")

    def actions(self, symbols, start, end) -> pd.DataFrame:
        raise self._unsupported("actions", "no corporate-action table is available")

    # --------------------------------------------------------------------- replay
    def replay(self, request: dict):
        """Re-issue a normalised request - what `Cache.refetch()` calls.

        The request records the method it came from, so a refetch reproduces the original
        call rather than approximating it. Keys the method does not accept are dropped:
        a request also records what the adapter DECIDED (`end_is_exclusive`,
        `dropped_unclosed_final_bar`, the pinned vendor flags), and those are there to be
        read by a human, not fed back in as arguments.
        """
        req = dict(request)
        method = req.pop("method", "bars")
        fn = getattr(self, method, None)
        if fn is None:
            raise KeyError(f"{self.decl.name} has no method {method!r} to replay")

        params = inspect.signature(fn).parameters
        positional = {"bars": ("symbols", "start", "end"), "macro": ("series_ids",),
                      "fundamentals": ("entities",), "universe": ("market", "as_of"),
                      "actions": ("symbols", "start", "end")}.get(method, ())
        args = [req.pop(name) for name in positional if name in req]
        if "adjustment" in req:
            adj = req.pop("adjustment")
            req["adjustment"] = Adjustment(adj) if adj else None
        return fn(*args, **{k: v for k, v in req.items() if k in params})

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.decl.name} {self.decl.rate_limit.describe()}>"


# --------------------------------------------------------------------------------- get
def get(name: str, **client_kw: Any) -> Adapter:
    """Construct an adapter by name, importing its vendor library only when it is used."""
    base = str(name).split(":", 1)[0]
    if base not in MODULES:
        raise KeyError(f"no adapter {name!r}; available: {sorted(MODULES)}")
    importlib.import_module(f"{__name__}.{MODULES[base]}")
    cls = declare.adapter_class(base)
    if ":" in str(name):
        client_kw.setdefault("venue", str(name).split(":", 1)[1])
    return cls(**client_kw)


def names() -> list[str]:
    return sorted(MODULES)


__all__ = ["Adapter", "Base", "MODULES", "get", "library_version", "names", "require"]
