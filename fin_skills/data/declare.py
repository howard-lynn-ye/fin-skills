"""What an adapter must state before `register()` will accept it.

No field has a default except `notes`, and the reason is the same one that runs through
the whole layer: the axes that decide whether a result is real - survivorship, point-in-
time, the adjustment convention, whether the data may be cached at all - are exactly the
axes a normalised vendor schema hides. An adapter that cannot answer `includes_delisted`
states False. The library never guesses on that axis.

Four of the fields exist because a vendor's terms, not a vendor's API, decide what the
code may do:

    key_sharing       FRED's API-key page: "All users of an application shall use their
                      own API key." A shared key cannot ship, so the code path that would
                      hold one does not exist (see `credential()` below).
    cache_policy      EODHD permits storing the data; Tiingo's Starter plan forbids
                      retaining it in "any persistent or durable storage" (ToS 1.6a).
                      One global cache would put Starter users in breach, so
                      `Cache.put()` REFUSES on "no-persist".
    non_display_use   Polygon/Massive ToS 5(d) prohibits non-display use, which is
                      arguably what a backtest is. The layer surfaces it and does not
                      opine on whether your use qualifies.
    licence_source    fredapi and pyqlib declare NO licence in PyPI metadata; tushare
                      declares a bare ambiguous 'BSD'. An audit that reads metadata sees
                      nothing or the wrong thing, so every declaration says where its
                      SPDX string was resolved FROM.

All values in the shipped adapters were re-verified on 2026-09-09 against PyPI's JSON API,
the repositories' own LICENSE files and the vendors' documentation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields as dataclass_fields
from datetime import date
from typing import Any

import pandas as pd

from fin_skills.data.ratelimit import RateLimit
from fin_skills.data.schema import Adjustment

KEY_SHARING = ("byok-required", "byok-blessed", "unstated")
CACHE_POLICY = ("persist-ok", "no-persist", "unstated")
NON_DISPLAY_USE = ("permitted", "prohibited", "unstated")
REDISTRIBUTION = ("prohibited", "attribution", "public-domain")
LICENCE_SOURCE = ("pypi", "repo-LICENSE", "classifier-only")


@dataclass(frozen=True)
class Declaration:
    """Everything an adapter states about itself, dated. `adapters()` prints the table."""

    name: str
    library: str                             # the pip name
    library_license: str                     # SPDX string, resolved - see licence_source
    adjustment_default: Adjustment | None    # what the vendor returns when asked nothing;
                                             # None where the concept does not apply
    adjustment_supported: tuple[Adjustment, ...]
    calendar: str
    tz: str
    bar_label: str
    interval_support: tuple[str, ...]
    includes_delisted: bool                  # THE decisive axis for survivorship
    point_in_time: bool                      # can it answer "what was known at t?"
    rate_limit: RateLimit                    # machine-readable; drives ratelimit.py
    free_tier: str                           # human-readable, dated
    requires_key: bool
    key_env_var: str
    key_sharing: str
    cache_policy: str
    non_display_use: str
    terms_url: str
    redistribution: str
    licence_source: str
    verified_on: str                         # a dated claim is a recoverable one
    attribution: str = ""                    # the exact string the terms require, if any
    notes: str = ""

    def __post_init__(self) -> None:
        _check(self.key_sharing, KEY_SHARING, "key_sharing", self.name)
        _check(self.cache_policy, CACHE_POLICY, "cache_policy", self.name)
        _check(self.non_display_use, NON_DISPLAY_USE, "non_display_use", self.name)
        _check(self.redistribution, REDISTRIBUTION, "redistribution", self.name)
        _check(self.licence_source, LICENCE_SOURCE, "licence_source", self.name)
        if self.bar_label not in ("open", "close", "n/a"):
            raise ValueError(f"{self.name}: bar_label must be 'open', 'close' or 'n/a'")
        if self.requires_key and not self.key_env_var:
            raise ValueError(f"{self.name}: requires_key=True needs a key_env_var naming "
                             f"the environment variable the USER sets")
        if not self.requires_key and self.key_env_var:
            raise ValueError(f"{self.name}: key_env_var is set but requires_key is False")
        try:
            date.fromisoformat(self.verified_on)
        except ValueError as exc:
            raise ValueError(f"{self.name}: verified_on must be an ISO date "
                             f"(YYYY-MM-DD), got {self.verified_on!r}") from exc
        if self.adjustment_default is not None:
            adj = Adjustment(self.adjustment_default)
            object.__setattr__(self, "adjustment_default", adj)
            if adj not in self.adjustment_supported:
                raise ValueError(f"{self.name}: adjustment_default {adj.value!r} is not "
                                 f"in adjustment_supported")
        object.__setattr__(self, "adjustment_supported",
                           tuple(Adjustment(a) for a in self.adjustment_supported))

    # ---------------------------------------------------------------------- views
    @property
    def rewrites_history(self) -> bool:
        """True when the vendor's default convention is anchored at the present, so a
        cached copy drifts from the live series on every new corporate action."""
        return (self.adjustment_default is not None
                and self.adjustment_default.rewrites_history)

    @property
    def may_persist(self) -> bool:
        return self.cache_policy != "no-persist"

    def row(self) -> dict[str, Any]:
        """One row of the table `adapters()` returns."""
        return {
            "adapter": self.name, "library": self.library,
            "licence": f"{self.library_license} ({self.licence_source})",
            "adjustment_default": (self.adjustment_default.value
                                   if self.adjustment_default else "n/a"),
            "delisted": self.includes_delisted, "PIT": self.point_in_time,
            "rate_limit": self.rate_limit.describe(),
            "key": self.key_env_var if self.requires_key else "-",
            "key_sharing": self.key_sharing, "cache_policy": self.cache_policy,
            "non_display_use": self.non_display_use,
            "redistribution": self.redistribution, "verified_on": self.verified_on,
        }

    def warnings(self) -> list[str]:
        """The things a user should read BEFORE choosing this adapter."""
        out: list[str] = []
        if not self.includes_delisted:
            out.append("includes_delisted=False - every result from this source is an "
                       "UPPER BOUND, not an estimate (survivorship)")
        if self.rewrites_history:
            out.append(f"adjustment_default={self.adjustment_default.value} rewrites "
                       f"history: the same query next month returns different numbers")
        if not self.point_in_time:
            out.append("point_in_time=False - it cannot answer 'what was known at t?'")
        if self.cache_policy == "no-persist":
            out.append("cache_policy=no-persist - the terms forbid retaining this data; "
                       "Cache.put() refuses without an explicit paid-tier acknowledgement")
        if self.non_display_use == "prohibited":
            out.append("non_display_use=prohibited - the terms forbid non-display use, "
                       "which arguably describes a backtest")
        if self.key_sharing == "byok-required":
            out.append(f"key_sharing=byok-required - every user must set their own "
                       f"${self.key_env_var}; a shared key cannot be shipped")
        if self.licence_source != "pypi":
            out.append(f"licence resolved from {self.licence_source}, not from PyPI "
                       f"metadata - dependency auditing tools will not see it")
        if self.redistribution == "prohibited":
            out.append("redistribution=prohibited - installing this adapter grants you "
                       "nothing; the code licence is not the data licence")
        return out

    def __repr__(self) -> str:
        return (f"Declaration({self.name!r}, {self.library}=={self.library_license}, "
                f"delisted={self.includes_delisted}, pit={self.point_in_time}, "
                f"{self.rate_limit.describe()}, verified {self.verified_on})")


def _check(value: str, allowed: tuple[str, ...], field: str, name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name}: {field} must be one of {allowed}, got {value!r}")


# ------------------------------------------------------------------------- the registry
_REGISTRY: dict[str, tuple[type, Declaration]] = {}


def register(adapter: type, decl: Declaration) -> None:
    """Accept an adapter into the registry, or refuse it.

    Registration is the only way into `adapters()`, so nothing reaches a user without a
    complete, dated declaration. Registering the same name twice replaces it, which is
    what a module reload does.
    """
    if not isinstance(decl, Declaration):
        raise TypeError("register() needs a Declaration; an adapter without one cannot "
                        "state its survivorship, PIT or licence position")
    if adapter is not None and not isinstance(adapter, type):
        raise TypeError("register() takes the adapter CLASS, not an instance")
    _REGISTRY[decl.name] = (adapter, decl)


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def declarations() -> list[Declaration]:
    """Every registered Declaration, by name. Importing this never imports a vendor."""
    return [d for _, d in sorted(_REGISTRY.values(), key=lambda kv: kv[1].name)]


def adapter_class(name: str) -> type:
    return _REGISTRY[_base(name)][0]


def lookup(name: str) -> Declaration:
    """The Declaration for an adapter or a Provenance source ("ccxt:binance" -> ccxt)."""
    key = _base(name)
    if key not in _REGISTRY:
        raise KeyError(f"no adapter {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key][1]


def registered(name: str) -> bool:
    return _base(name) in _REGISTRY


def _base(name: str) -> str:
    return str(name).split(":", 1)[0]


def adapters() -> pd.DataFrame:
    """Every registered adapter and what it declares - the table to read BEFORE choosing.

    Importing this never imports yfinance, akshare, ccxt, fredapi or edgartools.
    """
    return pd.DataFrame([d.row() for d in declarations()])


def describe(name: str | None = None) -> str:
    """The declaration table plus every warning, as ASCII text."""
    decls = declarations() if name is None else [lookup(name)]
    lines: list[str] = []
    for d in decls:
        lines.append(f"{d.name}  ({d.library}, {d.library_license} via "
                     f"{d.licence_source}, verified {d.verified_on})")
        lines.append(f"  adjustment default : "
                     f"{d.adjustment_default.value if d.adjustment_default else 'n/a'}"
                     f"   supported: "
                     f"{[a.value for a in d.adjustment_supported] or 'n/a'}")
        lines.append(f"  calendar / tz      : {d.calendar} / {d.tz}   "
                     f"bar_label={d.bar_label}   intervals="
                     f"{list(d.interval_support) or 'n/a'}")
        lines.append(f"  delisted / PIT     : {d.includes_delisted} / {d.point_in_time}")
        lines.append(f"  rate limit         : {d.rate_limit.describe()}")
        lines.append(f"  free tier          : {d.free_tier}")
        lines.append(f"  credential         : "
                     f"{('$' + d.key_env_var) if d.requires_key else 'none required'}"
                     f"   ({d.key_sharing})")
        lines.append(f"  cache / non-display: {d.cache_policy} / {d.non_display_use}")
        lines.append(f"  redistribution     : {d.redistribution}   terms: {d.terms_url}")
        if d.attribution:
            lines.append(f"  attribution        : {d.attribution}")
        for w in d.warnings():
            lines.append(f"  ! {w}")
        if d.notes:
            for chunk in str(d.notes).splitlines():
                lines.append(f"  . {chunk}")
        lines.append("")
    return "\n".join(lines).encode("ascii", "replace").decode("ascii")


# ---------------------------------------------------------------------- credentials
def credential(decl: Declaration) -> str | None:
    """Read the credential from the environment variable the Declaration names.

    This is the ONLY way a credential enters this library, and it is a read: nothing here
    stores, caches, logs, prints or serialises the value, no function in this package
    accepts a key as an argument, and no Declaration field can hold one. That is not
    politeness - FRED's API-key page says "All users of an application shall use their own
    API key", so a code path capable of shipping or proxying one must not exist.
    """
    if not decl.requires_key:
        return None
    value = os.environ.get(decl.key_env_var) or None
    if not value:
        raise RuntimeError(
            f"{decl.name} needs a credential and none is set. Get your own key at "
            f"{decl.terms_url or 'the vendor'} and export it:\n"
            f"    set {decl.key_env_var}=...        (Windows)\n"
            f"    export {decl.key_env_var}=...     (POSIX)\n"
            f"This library never ships, embeds or proxies a key ({decl.key_sharing}).")
    return value


def field_names() -> tuple[str, ...]:
    return tuple(f.name for f in dataclass_fields(Declaration))


__all__ = ["CACHE_POLICY", "Declaration", "KEY_SHARING", "LICENCE_SOURCE",
           "NON_DISPLAY_USE", "REDISTRIBUTION", "adapter_class", "adapters",
           "credential", "declarations", "describe", "field_names", "lookup",
           "register", "registered", "unregister"]
