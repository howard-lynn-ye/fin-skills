"""fin_skills.api.bundle - one container for a research run, one call for every guard.

PyOD gets a uniform interface from a uniform data container: every detector is fit(X).
Finance guards need different things - a look-ahead check needs the signal function and
the bars, a cost model needs returns and turnover - so the uniformity has to live in the
container. A Bundle holds the artefacts of one research run under a fixed vocabulary;
check() runs every guard whose inputs are present and says what each skipped guard still
needs. That is the fit(X) of this library.

    from fin_skills.api import Bundle, check

    b = Bundle(returns=strategy, turnover=turn, rf=0.05,
               close=aapl_close, actions=aapl_actions,
               bars=aapl_bars, signal_fn=lambda d: d.close.rolling(20).mean())
    print(b.coverage().summary())   # which guards are ready, what the others still need
    report = check(b)               # RunReport: ran / passed / failed / skipped (why)
    print(report.summary())

    check(b, guards=["cost_curve", "rf_convention"])       # a subset
    Suite("assert_causal", "warmup_probe").check(b)        # a reusable subset

The vocabulary is the union of every guard's input names, plus a few aliases so ONE slot
feeds every guard that means the same thing by it:

    returns             per-period strategy returns  -> cost_curve, rf_convention,
                                                        regime_coverage (its `strategy`)
    close               one instrument's closes      -> adjustment_check, reconcile_sources,
                                                        warmup_probe (its `closes`)
    bars                one instrument's OHLCV frame -> assert_causal (its `df`)
    signal_fn           df -> Series / DataFrame     -> assert_causal (its `fn`)
    regime_labels       per-period regime label      -> regime_coverage (its `labels`),
                                                        regime_lookahead (its `regime`)
    benchmark_returns   benchmark per-period returns -> spa_test, cost_curve (its `benchmark`)
    underlying_returns  the traded asset's returns   -> regime_coverage (its `asset`)
    features / target   ML matrix and labels         -> purge_effect (its `x` / `y`)

Everything else keeps the guard's own keyword. `slots()` lists every slot with the guards
it reaches; `Bundle(...)` rejects a name that is not in the vocabulary and suggests the
closest one. Construction also checks the obvious shape errors early (a DataFrame where a
Series is expected, an unsorted DatetimeIndex) with a message naming the slot, because
every guard downstream assumes sorted time series.
"""
from __future__ import annotations

import difflib
import numbers
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from fin_skills.api.base import Guard, RunReport, ascii_only, get, registry

# ----------------------------------------------------------------------------- aliases
# guard name -> {guard keyword: bundle slot}. A guard keyword not listed here IS its slot.
ALIASES: dict[str, dict[str, str]] = {
    "assert_causal": {"fn": "signal_fn", "df": "bars"},
    "warmup_probe": {"closes": "close"},
    "regime_coverage": {"strategy": "returns", "asset": "underlying_returns",
                        "labels": "regime_labels"},
    "regime_lookahead": {"regime": "regime_labels"},
    "cost_curve": {"benchmark": "benchmark_returns"},
    "purge_effect": {"x": "features", "y": "target"},
}

# slots whose value is a time series that every guard assumes is sorted
_TIME_SERIES_SLOTS = frozenset({
    "returns", "close", "other", "prices", "bars", "asset_returns", "underlying_returns",
    "position", "p_calm", "stitched", "index_returns", "benchmark_returns", "model_returns",
})


@dataclass(frozen=True)
class Slot:
    """One name in the vocabulary: what kind of value it holds and what it means."""

    name: str
    kind: str  # series | frame | callable | scalar | text | date | sequence | mapping | any
    doc: str


# The curated data slots. Tuning knobs (tol, k, seed, alpha, ...) are accepted too - they
# are documented by the guard that owns them - but these are the ones a research run has.
DATA_SLOTS: tuple[Slot, ...] = (
    Slot("returns", "series", "per-period GROSS strategy returns (Series, DatetimeIndex)"),
    Slot("turnover", "any", "one-way traded notional per period as a fraction of the book, or a scalar"),
    Slot("book", "scalar", "book size in dollars - the capital the strategy actually deploys"),
    Slot("adv", "any", "dollar volume per traded name: a scalar, a per-name Series, or a dates x tickers panel"),
    Slot("benchmark_returns", "any", "per-period benchmark returns (Series), or a scalar hurdle"),
    Slot("model_returns", "frame", "(T, k) per-period returns of EVERY candidate tried, abandoned ones included"),
    Slot("rf", "scalar", "annual risk-free rate as a decimal (0.05 = 5%)"),
    Slot("periods_per_year", "scalar", "252 equities, 365 crypto, 260 FX, 12 monthly"),
    Slot("prices", "frame", "wide close panel, dates x tickers, NaN after a name stops trading"),
    Slot("close", "series", "one instrument's close (Series, DatetimeIndex)"),
    Slot("other", "series", "the same instrument's close from a second source"),
    Slot("actions", "any", "corporate actions: DataFrame(date, ratio[, kind]) or (date, ratio) pairs"),
    Slot("bars", "frame", "one instrument's OHLCV frame - the input a signal function sees"),
    Slot("signal_fn", "callable", "df -> Series/DataFrame aligned to df, or {name: callable}"),
    Slot("indicator", "callable", "ndarray -> ndarray of the same length, or {name: callable}"),
    Slot("asset_returns", "frame", "wide returns panel, dates x assets - the optimizer input"),
    Slot("underlying_returns", "series", "per-period returns of the asset the strategy trades"),
    Slot("position", "series", "per-period position, 0 = flat"),
    Slot("dates", "any", "DatetimeIndex (or dates) of the test window"),
    Slot("regime_labels", "any", "per-period regime label, truth or a proxy such as a drawdown state"),
    Slot("p_calm", "series", "P(regime 0) per period - the series the strategy trades on"),
    Slot("universe", "mapping", "{rebalance date: [tickers]} as produced at each rebalance"),
    Slot("members", "frame", "membership table: ticker, start_date, end_date"),
    Slot("rebalance_dates", "sequence", "rebalance dates"),
    Slot("listings", "frame", "listing / delisting table"),
    Slot("liquidity", "frame", "dollar-volume panel for the trailing-ADV screen"),
    Slot("facts", "sequence", "SEC companyfacts units list: dicts with start, end, val, accn, fy, fp, form, filed"),
    Slot("as_of", "date", "the decision date for point-in-time checks"),
    Slot("left", "frame", "signals frame for an as-of join"),
    Slot("right", "frame", "quotes frame for an as-of join"),
    Slot("on", "text", "timestamp column present in both join frames"),
    Slot("by", "any", "group key column(s) for the join"),
    Slot("tolerance", "any", "staleness budget for the join ('5min', '3D', or a number)"),
    Slot("features", "any", "ML feature matrix (n, p)"),
    Slot("target", "any", "labels (n,), where target[i] resolves at i + horizon"),
    Slot("horizon", "scalar", "label horizon in bars"),
    Slot("folds", "sequence", "the CV folds handed to run_fold (at least 2)"),
    Slot("run_fold", "callable", "(fold, config) -> result"),
    Slot("cutoff", "date", "the LLM's training-data cutoff"),
    Slot("test_start", "date", "backtest window start"),
    Slot("test_end", "date", "backtest window end"),
    Slot("stitched", "series", "a continuous futures series from continuous_contract.stitch"),
    Slot("contracts", "frame", "near -> far contract frame"),
    Slot("roll_dates", "sequence", "roll dates"),
    Slot("index_returns", "series", "daily simple returns of a leveraged product's index"),
    Slot("lev", "scalar", "the product's leverage (3, -1, -3, ...)"),
    Slot("greeks", "mapping", "{price, delta, gamma, vega, theta, rho} (any subset)"),
    Slot("flag", "text", "'c' or 'p'"),
    Slot("pair", "any", "'EURUSD' or a sequence of pairs"),
    Slot("card", "any", "a result_manifest.ResultCard"),
    Slot("timeline", "any", "a fin_skills.synthesis.Timeline - facts of mixed kinds, each with its clock"),
    Slot("dossier", "any", "a fin_skills.synthesis.Dossier - one entity as of one date, fields with provenance"),
    Slot("best_sharpe", "scalar", "the best in-sample Sharpe found"),
    Slot("n_obs", "scalar", "observations behind that Sharpe"),
    Slot("sharpes", "sequence", "the Sharpe of every configuration tried"),
    Slot("ledger", "any", "path to trials.jsonl, or a TrialLedger"),
    Slot("broker", "text", "'ib', 'alpaca', 'schwab', 'ccxt:<venue>'"),
    Slot("bar", "mapping", "one A-share bar: prev_close, open, high, low, close, volume"),
    Slot("code", "text", "6-digit A-share code, optional exchange prefix/suffix"),
    Slot("date", "date", "the bar's date"),
    Slot("eval_date", "any", "QuantLib evaluation date"),
    Slot("curve_ref", "any", "the term structure's reference date"),
    Slot("expiry", "any", "the instrument's exercise date"),
    Slot("panel", "frame", "Brinson panel: sector index with wp, wb, rp, rb"),
    Slot("panels", "sequence", "a sequence of Brinson panels, Carino-linked"),
)
_CURATED: dict[str, Slot] = {s.name: s for s in DATA_SLOTS}


# --------------------------------------------------------------------------- vocabulary
def _slot_of(guard: str, keyword: str) -> str:
    return ALIASES.get(guard, {}).get(keyword, keyword)


def _reach() -> dict[str, list[str]]:
    """{slot: [guard names it reaches]} over the whole registry."""
    out: dict[str, list[str]] = {}
    for cls in registry():
        for k in cls.required + cls.optional:
            out.setdefault(_slot_of(cls.name, k), []).append(cls.name)
    return out


def slots() -> list[Slot]:
    """Every accepted slot name, curated ones first, with what it means and who reads it."""
    reach = _reach()
    out: list[Slot] = []
    for s in DATA_SLOTS:
        who = ", ".join(reach.get(s.name, []))
        out.append(Slot(s.name, s.kind, f"{s.doc} -> {who}" if who else s.doc))
    for name in sorted(set(reach) - set(_CURATED)):
        out.append(Slot(name, "any", f"parameter of {', '.join(reach[name])} (see its describe())"))
    return out


def vocabulary() -> frozenset[str]:
    """The set of slot names Bundle accepts."""
    return frozenset(_reach()) | frozenset(_CURATED)


# ---------------------------------------------------------------------------- checking
def _is_frame(v: Any) -> bool:
    return type(v).__name__ == "DataFrame"


def _is_series(v: Any) -> bool:
    return type(v).__name__ == "Series"


def _kind_error(slot: Slot, value: Any) -> str | None:
    """A message when `value` plainly cannot be what `slot` expects, else None. Loose on
    purpose: guards validate their own inputs; this catches the shape mistakes that would
    otherwise surface as a confusing error deep inside a wrapped script."""
    k, name = slot.kind, type(value).__name__
    if value is None:
        return None
    if k == "series":
        if _is_frame(value):
            return f"{slot.name!r} expects a Series (one column), got a DataFrame"
        if isinstance(value, (str, bytes, numbers.Number)) or callable(value):
            return f"{slot.name!r} expects a Series or 1-D array, got {name}"
    elif k == "frame":
        if _is_series(value):
            return f"{slot.name!r} expects a DataFrame, got a Series"
        if isinstance(value, (str, bytes, numbers.Number)) or callable(value):
            return f"{slot.name!r} expects a DataFrame, got {name}"
    elif k == "callable":
        ok = callable(value) or (isinstance(value, Mapping) and value
                                 and all(callable(f) for f in value.values()))
        if not ok:
            return f"{slot.name!r} expects a callable or a {{name: callable}} mapping, got {name}"
    elif k == "scalar":
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return f"{slot.name!r} expects a number, got {name}"
    elif k == "text":
        if not isinstance(value, str):
            return f"{slot.name!r} expects a str, got {name}"
    elif k == "mapping":
        if not isinstance(value, Mapping):
            return f"{slot.name!r} expects a mapping, got {name}"
    elif k == "sequence":
        if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
            return f"{slot.name!r} expects a sequence, got {name}"
    return None


def _order_error(name: str, value: Any) -> str | None:
    """Time-series slots must be sorted: every guard downstream assumes it."""
    if name not in _TIME_SERIES_SLOTS or not (_is_series(value) or _is_frame(value)):
        return None
    idx = value.index
    if type(idx).__name__ == "DatetimeIndex" and not idx.is_monotonic_increasing:
        return (f"{name!r} has an unsorted DatetimeIndex; every guard assumes sorted time "
                f"series - sort_index() it first")
    return None


# -------------------------------------------------------------------------------- Bundle
class Bundle:
    """The artefacts of one research run, under the shared vocabulary. Immutable: use
    with_() / without() to derive a new one."""

    __slots__ = ("_slots",)

    def __init__(self, **slots: Any) -> None:
        vocab = vocabulary()
        bad = sorted(set(slots) - vocab)
        if bad:
            hints = []
            for b in bad:
                close = difflib.get_close_matches(b, sorted(vocab), n=2, cutoff=0.6)
                hints.append(f"{b!r}" + (f" (did you mean {close}?)" if close else ""))
            raise TypeError(f"unknown slot(s) {', '.join(hints)}; "
                            f"fin_skills.api.slots() lists the vocabulary")
        problems = []
        for name, value in slots.items():
            spec = _CURATED.get(name)
            if spec is not None:
                msg = _kind_error(spec, value)
                if msg:
                    problems.append(msg)
            msg = _order_error(name, value)
            if msg:
                problems.append(msg)
        if problems:
            raise TypeError("; ".join(problems))
        self._slots: dict[str, Any] = {k: v for k, v in slots.items() if v is not None}

    # ---------------------------------------------------------------- mapping-like
    def __getattr__(self, name: str) -> Any:
        try:
            return self._slots[name]
        except KeyError:
            raise AttributeError(f"Bundle has no slot {name!r}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._slots

    def __len__(self) -> int:
        return len(self._slots)

    def __iter__(self) -> Iterator[str]:
        return iter(self._slots)

    def __repr__(self) -> str:
        parts = []
        for k, v in self._slots.items():
            shape = getattr(v, "shape", None)
            parts.append(f"{k}={type(v).__name__}{list(shape) if shape is not None else ''}")
        return f"Bundle({', '.join(parts)})"

    def has(self, *names: str) -> bool:
        return all(n in self._slots for n in names)

    def get(self, name: str, default: Any = None) -> Any:
        return self._slots.get(name, default)

    def slots(self) -> dict[str, Any]:
        """A copy of the slot mapping."""
        return dict(self._slots)

    def with_(self, **more: Any) -> "Bundle":
        """A new Bundle with these slots added or replaced."""
        merged = dict(self._slots)
        merged.update(more)
        return Bundle(**merged)

    def without(self, *names: str) -> "Bundle":
        return Bundle(**{k: v for k, v in self._slots.items() if k not in names})

    # ---------------------------------------------------------------- per guard
    def missing_for(self, guard: Guard | str) -> list[str]:
        """Required slots this bundle lacks for `guard`, in slot names."""
        g = get(guard) if isinstance(guard, str) else guard
        return [_slot_of(g.name, k) for k in g.required if _slot_of(g.name, k) not in self._slots]

    def inputs_for(self, guard: Guard | str) -> dict[str, Any]:
        """The keyword arguments guard.run() takes from this bundle (aliases resolved)."""
        g = get(guard) if isinstance(guard, str) else guard
        out: dict[str, Any] = {}
        for k in g.required + g.optional:
            s = _slot_of(g.name, k)
            if s in self._slots:
                out[k] = self._slots[s]
        return out

    def coverage(self, guards: Iterable[str] | None = None) -> "Coverage":
        return coverage(self, guards)

    def check(self, guards: Iterable[str] | None = None, strict: bool = False) -> RunReport:
        return check(self, guards=guards, strict=strict)


# ------------------------------------------------------------------------------ coverage
@dataclass
class Coverage:
    """Which guards a bundle can run, and what each of the others still needs."""

    ready: list[str]
    missing: dict[str, list[str]]

    def unlocks(self) -> dict[str, list[str]]:
        """{slot: guards that become ready if ONLY that slot is added} - the next thing
        to put in the bundle."""
        out: dict[str, list[str]] = {}
        for g, miss in self.missing.items():
            if len(miss) == 1:
                out.setdefault(miss[0], []).append(g)
        return dict(sorted(out.items(), key=lambda kv: (-len(kv[1]), kv[0])))

    def summary(self) -> str:
        lines = [f"ready ({len(self.ready)}): " + (", ".join(self.ready) or "-")]
        if self.missing:
            lines.append(f"not ready ({len(self.missing)}):")
            for g, miss in self.missing.items():
                lines.append(f"  {g:<24} needs {', '.join(miss)}")
        unl = self.unlocks()
        if unl:
            lines.append("one slot away:")
            for s, gs in unl.items():
                lines.append(f"  + {s:<22} unlocks {', '.join(gs)}")
        return ascii_only("\n".join(lines))

    def __str__(self) -> str:
        return self.summary()


def _resolve(guards: Iterable[str] | None) -> list[Guard]:
    if guards is None:
        return [cls() for cls in registry()]
    return [get(n) if isinstance(n, str) else n for n in guards]


def coverage(bundle: Bundle | Mapping[str, Any], guards: Iterable[str] | None = None) -> Coverage:
    b = bundle if isinstance(bundle, Bundle) else Bundle(**dict(bundle))
    ready, missing = [], {}
    for g in _resolve(guards):
        miss = b.missing_for(g)
        (missing.__setitem__(g.name, miss) if miss else ready.append(g.name))
    return Coverage(ready=ready, missing=missing)


# --------------------------------------------------------------------------------- check
def check(bundle: Bundle | Mapping[str, Any] | None = None, *,
          guards: Iterable[str] | None = None, strict: bool = False,
          **slots: Any) -> RunReport:
    """Run every guard (or the named subset) whose required slots are present.

    Skipped guards are listed in report.skipped with the slots they still need. A guard
    whose inputs are present but rejected (its run() raised TypeError - wrong shape, bad
    value) is recorded in report.rejected with the message instead of aborting the
    run, unless strict=True.
    """
    if bundle is None:
        b = Bundle(**slots)
    else:
        b = bundle if isinstance(bundle, Bundle) else Bundle(**dict(bundle))
        if slots:
            b = b.with_(**slots)
    report = RunReport()
    for g in _resolve(guards):
        miss = b.missing_for(g)
        if miss:
            report.skipped[g.name] = miss
            continue
        try:
            report.append(g.run(**b.inputs_for(g)))
        except TypeError as exc:
            if strict:
                raise
            report.rejected[g.name] = ascii_only(str(exc))
    return report


class Suite:
    """A named, reusable subset of guards: Suite("assert_causal", "warmup_probe")."""

    def __init__(self, *names: str) -> None:
        self._guards = [get(n) for n in names]  # KeyError early for a bad name
        if not self._guards:
            raise ValueError("Suite needs at least one guard name")

    @property
    def names(self) -> list[str]:
        return [g.name for g in self._guards]

    def coverage(self, bundle: Bundle | Mapping[str, Any]) -> Coverage:
        return coverage(bundle, self.names)

    def check(self, bundle: Bundle | Mapping[str, Any] | None = None, *,
              strict: bool = False, **slots: Any) -> RunReport:
        return check(bundle, guards=self.names, strict=strict, **slots)

    def describe(self) -> str:
        return ascii_only("\n".join(g.describe() for g in self._guards))

    def __repr__(self) -> str:
        return f"Suite({', '.join(self.names)})"


__all__ = ["ALIASES", "Bundle", "Coverage", "DATA_SLOTS", "Slot", "Suite", "check",
           "coverage", "slots", "vocabulary"]
