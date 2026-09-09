"""alphalens, quantstats and pyfolio - with the two silent conventions taken away from them.

TRAP 1 - **alphalens does not lag your factor.** `compute_forward_returns` does
`prices.pct_change(period).shift(-period)`, so a factor observed at `t` with `period=p` is
scored against `P[t+p]/P[t] - 1` - a return that starts accruing from `t`'s OWN price. If
the factor came from `t`'s close (overwhelmingly the common case), the `1D` number alphalens
shows is the `t`-close -> `t+1`-close return, earnable only by transacting at the very close
that produced the signal (fin_skills.load('lib-alphalens')). `lag_factor` shifts it one bar
per asset unless you claim `already_lagged=True`, and the claim is TESTED with
`assert_causal` rather than believed.

TRAP 2 - **`quantstats.stats.cagr()` accepts `rf` and discards it.** Verified in
`quantstats/utils.py::_prepare_returns`, which dispatches on the CALLER'S FUNCTION NAME:

    function = inspect.stack()[1][3]
    unnecessary_function_calls = ["_prepare_benchmark", "cagr", "gain_to_pain_ratio",
                                  "rolling_volatility"]

`"cagr"` is on the exclusion list, so the parameter is silently dropped - and the same list
disables `rf` for `gain_to_pain_ratio` and `rolling_volatility`. This bridge computes CAGR
itself from the excess series and never passes `rf` to `cagr`.

TRAP 3 - **the same `rf=0.05` is three different numbers.** ANNUAL in quantstats,
PyPortfolioOpt and ffn; **PER-PERIOD (daily), subtracted raw** in empyrical and
pyfolio-reloaded, where it sits right beside `period='daily'`. `ep.sharpe_ratio(r,
risk_free=0.05)` reads 5% PER DAY and returns about -81. This bridge takes `rf_annual`
once, converts per backend (geometric de-annualisation for quantstats, `rf/periods` for the
pyfolio side), records which conversion it used, and runs `rf_convention` on the bundle
before rendering - so the number in the tearsheet and the number the guard checked are the
same number.

Licences: alphalens-reloaded, quantstats and pyfolio-reloaded are all Apache-2.0 and may be
extras. alphalens-reloaded pins `pandas<3.0,>=1.5.0`, a hard conflict on pandas 3.x, which
is a packaging reason to keep it optional on top of the licence one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from fin_skills.api import Bundle, get
from fin_skills.api.conventions import annualize_sharpe
from fin_skills.bridges import _lazy
from fin_skills.libraries.rf_convention import _sharpe_annual_rf, _sharpe_per_period_rf

LIBRARY = "quantstats"
LICENCE = _lazy.info(LIBRARY).licence
VERIFIED_ON = _lazy.info(LIBRARY).verified_on

BACKENDS: tuple[str, ...] = ("quantstats", "pyfolio")
#: alphalens' own default is 0.35 - it silently drops up to 35% of the names.
DEFAULT_MAX_LOSS = 0.0


class FactorNotLaggedError(_lazy.NotLaggedError):
    """An `already_lagged=True` claim about a factor that could not be proven."""


# --------------------------------------------------------------------------- alphalens
def lag_factor(factor: pd.Series | pd.DataFrame, *, already_lagged: bool = False,
               signal_fn: Callable[[pd.DataFrame], Any] | None = None,
               bars: pd.DataFrame | None = None, k: int | None = None) -> Any:
    """Shift a factor one bar per asset, or prove it did not need shifting.

    factor          long, MultiIndex (date, asset) - alphalens' own orientation.
    already_lagged  an ASSERTION. Supply `signal_fn` and `bars` and `_lazy.prove_lagged`
                    runs `assert_causal` on the function that produced the factor; without
                    them the claim cannot be tested, and an untested claim is refused.

    Runs with alphalens absent - the enforcement is the bridge's, not the library's.
    """
    if already_lagged:
        try:
            if signal_fn is None or bars is None:
                _lazy.refuse_untestable_claim(
                    "the factor",
                    "Pass signal_fn=<the callable that produced it> and bars=<the OHLCV "
                    "frame it saw>, so assert_causal can perturb bar t and watch the "
                    "factor at t.")
            _lazy.prove_lagged(signal_fn, bars, k=k, name="factor")
        except _lazy.NotLaggedError as exc:
            raise FactorNotLaggedError(str(exc)) from exc
        return factor
    if isinstance(factor.index, pd.MultiIndex):
        return factor.groupby(level=1).shift(1).dropna()
    return factor.shift(1).dropna()


def to_alphalens(factor: pd.Series | pd.DataFrame, prices: pd.DataFrame, *,
                 periods: tuple[int, ...] = (1, 5, 10), already_lagged: bool = False,
                 signal_fn: Callable[[pd.DataFrame], Any] | None = None,
                 bars: pd.DataFrame | None = None, max_loss: float = DEFAULT_MAX_LOSS,
                 k: int | None = None, **kw: Any):
    """`get_clean_factor_and_forward_returns` with the factor lagged and max_loss pinned.

    The orientation asymmetry is alphalens', not this bridge's: `factor` is LONG with a
    (date, asset) MultiIndex and `prices` is WIDE, dates x assets.

    `max_loss` defaults to 0.0 here, not alphalens' 0.35: its default silently drops up to
    a third of your names to forward-return NaNs and reports the loss in a log line nobody
    reads. Raise it deliberately if you mean to.
    """
    lagged = lag_factor(factor, already_lagged=already_lagged, signal_fn=signal_fn,
                        bars=bars, k=k)
    al = _lazy.need("alphalens-reloaded", why="to build the clean factor frame")
    from alphalens.utils import get_clean_factor_and_forward_returns  # noqa: PLC0415

    out = get_clean_factor_and_forward_returns(lagged, prices, periods=tuple(periods),
                                               max_loss=float(max_loss), **kw)
    _ = _lazy.provenance("alphalens-reloaded", al, request={
        "periods": list(periods), "already_lagged": bool(already_lagged),
        "shifted": not already_lagged, "max_loss": float(max_loss)})
    return out


# -------------------------------------------------------------------------- tearsheets
@dataclass(frozen=True)
class Tearsheet:
    """The numbers, the rf conversion that produced them, and the guard that checked it."""

    backend: str
    rf_annual: float
    rf_handed_to_backend: float
    rf_conversion: str
    periods_per_year: int
    stats: Mapping[str, float] = field(default_factory=dict)
    guard: Any = None
    rendered: Any = None
    provenance: _lazy.Provenance | None = None

    def summary(self) -> str:
        head = (f"{self.backend}: rf_annual={self.rf_annual:.4%} -> "
                f"{self.rf_handed_to_backend:.8g} ({self.rf_conversion})")
        body = "\n".join(f"  {k:<22} {v:.6g}" for k, v in self.stats.items())
        return f"{head}\n{body}"


def _returns_of(bundle: Bundle | pd.Series) -> pd.Series:
    if isinstance(bundle, pd.Series):
        return bundle.astype(float)
    r = bundle.get("returns")
    if r is None:
        raise TypeError("bundle has no `returns` slot")
    return pd.Series(r).astype(float)


def _core_stats(r: pd.Series, rf_per_period: float, ppy: int, geometric: bool) -> dict:
    """The four numbers both backends get wrong in different ways, computed here once."""
    excess = r - rf_per_period
    n = len(r)
    growth = float((1.0 + excess).prod())
    cagr = growth ** (ppy / n) - 1.0 if n and growth > 0 else float("nan")
    sharpe = (_sharpe_annual_rf(r, (1.0 + rf_per_period) ** ppy - 1.0, ppy) if geometric
              else _sharpe_per_period_rf(r, rf_per_period, ppy))
    eq = (1.0 + r).cumprod()
    return {"cagr_excess": cagr,
            "sharpe": sharpe,
            "sharpe_rf_zero": annualize_sharpe(r, ppy),
            "volatility_annual": float(r.std(ddof=1) * np.sqrt(ppy)),
            "max_drawdown": float((eq / eq.cummax() - 1.0).min())}


def tearsheet(bundle: Bundle | pd.Series, *, backend: str = "quantstats",
              rf_annual: float = 0.0, periods_per_year: int | None = None,
              render: bool = False, **kw: Any) -> Tearsheet:
    """Report a strategy under ONE stated risk-free convention, converted per backend.

    backend  'quantstats' - `rf` is ANNUAL and de-annualised geometrically. CAGR is
             computed HERE, from the excess series, because `cagr(rf=...)` discards it.
             'pyfolio'    - empyrical's `risk_free=` is PER-PERIOD and subtracted raw, so
             `rf_annual / periods_per_year` is what the backend is handed. That value is on
             the result as `rf_handed_to_backend`.
    render   also call the library's own tear-sheet renderer. Off by default: the numbers
             are this bridge's, and they are computed whether or not the library is
             installed. Turning it on is what makes the library a dependency of the call.

    Runs `rf_convention` on the bundle before anything is rendered.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {list(BACKENDS)}, got {backend!r}")
    r = _returns_of(bundle)
    ppy = int(periods_per_year if periods_per_year is not None
              else (bundle.get("periods_per_year", 252) if isinstance(bundle, Bundle)
                    else 252))
    if abs(float(rf_annual)) >= 0.5:
        raise ValueError(f"rf_annual={rf_annual:g} looks like a percentage, not a decimal "
                         f"rate (0.05 = 5%). empyrical would read it as {rf_annual:g} PER "
                         f"DAY and hand you a Sharpe near -65.")

    b = (bundle if isinstance(bundle, Bundle) else Bundle(returns=r)).with_(
        rf=float(rf_annual), periods_per_year=ppy)
    guard = get("rf_convention").run(**b.inputs_for("rf_convention"))

    if backend == "quantstats":
        rf_used = (1.0 + float(rf_annual)) ** (1.0 / ppy) - 1.0
        conversion = ("annual, de-annualised GEOMETRICALLY ((1+rf)**(1/periods)-1) - "
                      "quantstats' own convention; rf is never passed to cagr(), which "
                      "silently discards it")
        stats = _core_stats(r, rf_used, ppy, geometric=True)
    else:
        rf_used = float(rf_annual) / ppy
        conversion = ("rf_annual / periods_per_year - empyrical's risk_free= is PER-PERIOD "
                      "and subtracted raw; handing it the annual rate gives a Sharpe near "
                      "-65 and no error")
        stats = _core_stats(r, rf_used, ppy, geometric=False)

    rendered = None
    if render:
        rendered = _render(backend, r, rf_used, ppy, kw)
    lib = "quantstats" if backend == "quantstats" else "pyfolio-reloaded"
    prov = _lazy.provenance(lib, source=backend, request={
        "rf_annual": float(rf_annual), "rf_handed_to_backend": rf_used,
        "conversion": conversion, "periods_per_year": ppy, "rendered": bool(render)})
    return Tearsheet(backend=backend, rf_annual=float(rf_annual),
                     rf_handed_to_backend=rf_used, rf_conversion=conversion,
                     periods_per_year=ppy, stats=stats, guard=guard, rendered=rendered,
                     provenance=prov)


def _render(backend: str, r: pd.Series, rf_used: float, ppy: int, kw: dict) -> Any:
    """Hand the library the number this bridge already converted, and nothing else."""
    if backend == "quantstats":
        qs = _lazy.need("quantstats", why="to render a tear sheet")
        annual = (1.0 + rf_used) ** ppy - 1.0
        # rf goes to sharpe/sortino, which honour it. It is NEVER passed to cagr().
        if kw:
            raise TypeError(f"the quantstats path takes no extra keywords, got "
                            f"{sorted(kw)}; **kw is forwarded to pyfolio's renderer only")
        return {"sharpe": float(qs.stats.sharpe(r, rf=annual, periods=ppy,
                                                annualize=True)),
                "sortino": float(qs.stats.sortino(r, rf=annual, periods=ppy,
                                                  annualize=True)),
                "max_drawdown": float(qs.stats.max_drawdown(r)),
                "volatility": float(qs.stats.volatility(r, periods=ppy))}
    pf = _lazy.need("pyfolio-reloaded", why="to render a tear sheet")
    return pf.create_returns_tear_sheet(r, return_fig=True, **kw)


__all__ = ["BACKENDS", "DEFAULT_MAX_LOSS", "FactorNotLaggedError", "LIBRARY", "LICENCE",
           "Tearsheet", "VERIFIED_ON", "lag_factor", "tearsheet", "to_alphalens"]
