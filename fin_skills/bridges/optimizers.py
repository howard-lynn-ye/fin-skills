"""One signature over PyPortfolioOpt, skfolio and Riskfolio-Lib - and five refusals.

The three libraries solve the same problem with three incompatible conventions, and every
one of the disagreements is silent. `OptimizerInput` states each convention once;
`optimize()` refuses the five ways the conversion goes wrong:

1. **PRICES WHERE RETURNS ARE REQUIRED.** `HRPOpt(returns=)`, `rp.HCPortfolio(returns=)`
   and skfolio's `fit(X)` all take RETURNS, and all three accept a price matrix silently -
   PyPortfolioOpt's type check validates the CONTAINER (`returns are not a dataframe`) and
   never the content. Measured in `fin_skills.libraries.weight_traps` (seed 0, 3-year
   6-asset panel): feeding prices puts **85.18%** of the book into the single
   highest-volatility name, an **L1 weight distance of 1.5648** out of a possible 2.0, and
   **+141.4%** realised annualised vol (0.3486 vs 0.1444) - worse than 1/N at 0.1976. The
   mechanism is units: `cov(prices)` is in dollars-squared, so HRP's inverse-variance split
   collapses into inverse-SHARE-PRICE weighting, and the $8 ticker shows 3,654x less
   "variance" than the $620 one. The output still sums to 1.0000 and is still long-only.
   This bridge runs the repo's own `weight_traps` guard and re-raises its message.

2. **HRP LINKAGE.** PyPortfolioOpt defaults to `'single'` (the AFML original), skfolio to
   Ward. They disagree BY DESIGN, so `linkage` is required for any hierarchical model here.

3. **RISKFOLIO STATE.** `rp.Portfolio` is imperative: `assets_stats()` must be called
   before `optimization()` and RE-called after the data changes, or it optimizes against
   stale or missing mu/Sigma with no error and normal-looking weights. This bridge sets the
   returns and calls `assets_stats` in one step and never lets a mutable handle escape.

4. **rf UNITS.** `rf` is ANNUAL in quantstats, PyPortfolioOpt and ffn, PER-PERIOD in
   empyrical, and undocumented in Riskfolio (which works on per-period returns throughout,
   so per-period is the only coherent reading). `OptimizerInput.rf_annual` is taken ONCE as
   an annual decimal; the conversion applied per backend is recorded on the result's
   `.attrs`.

5. **SOLVER CLASS.** EVaR, RLVaR, EDaR and RLDaR need exponential- or power-cone support
   (Clarabel, SCS, MOSEK); cardinality and buy-in constraints need a MIP solver (HiGHS,
   SCIP, Gurobi, MOSEK). The backends report this as *"solver did not converge"*. This
   bridge maps the model and risk measure to the required cone class and raises a named
   error instead.

Sources: fin_skills.load('lib-pyportfolioopt'), 'lib-skfolio', 'lib-riskfolio'. All three
licences are permissive (MIT / BSD-3-Clause) and may be pyproject extras.
"""
from __future__ import annotations

import importlib.util
import warnings
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from fin_skills.api import Bundle, get
from fin_skills.bridges import _lazy

BACKENDS: tuple[str, ...] = ("pypfopt", "skfolio", "riskfolio")
_PIP = {"pypfopt": "PyPortfolioOpt", "skfolio": "skfolio", "riskfolio": "Riskfolio-Lib"}

LIBRARY = "PyPortfolioOpt"                 # the bridge's primary; see LIBRARIES for all three
LICENCE = _lazy.info(LIBRARY).licence
VERIFIED_ON = _lazy.info(LIBRARY).verified_on

#: models whose result depends on a linkage choice the three libraries do not share.
HIERARCHICAL: frozenset[str] = frozenset({"HRP", "HERC", "NCO", "HERC2"})
#: risk measures that need an exponential or power cone.
CONE_MEASURES: frozenset[str] = frozenset({"EVaR", "RLVaR", "EDaR", "RLDaR"})
CONE_SOLVERS: tuple[str, ...] = ("CLARABEL", "SCS", "MOSEK")
MIP_SOLVERS: tuple[str, ...] = ("HIGHS", "SCIP", "GUROBI", "MOSEK", "CPLEX", "CBC")
#: the largest per-period mean a returns column may plausibly have (5% a day, a period).
MAX_PLAUSIBLE_MEAN = 0.05


class PricesWhereReturnsError(ValueError):
    """A price matrix (or percent-scaled returns) was passed where returns are required."""


class LinkageRequiredError(ValueError):
    """A hierarchical model was requested without stating the linkage."""


class SolverClassError(ValueError):
    """The risk measure or constraint needs a solver class that is not installed."""


class BackendError(RuntimeError):
    """The backend refused. Its own message is attached, with the version that produced it."""


@dataclass(frozen=True)
class OptimizerInput:
    """The shared, unambiguous input. `returns` is RETURNS - and the bridge proves it.

    returns          dates x assets, per-period SIMPLE returns.
    mu               expected returns, PER PERIOD, or None to estimate from `returns`.
    sigma            covariance, PER PERIOD, same units as `mu`, or None.
    rf_annual        the risk-free rate ONCE, as an annual decimal (0.05 = 5%).
    periods_per_year 252 equities, 365 crypto, 260 FX, 12 monthly.
    linkage          'single' | 'ward' | 'average' | 'complete' - REQUIRED for any
                     hierarchical model, because PyPortfolioOpt and skfolio disagree by
                     design and neither says so.
    """

    returns: pd.DataFrame
    mu: pd.Series | None = None
    sigma: pd.DataFrame | None = None
    rf_annual: float = 0.0
    periods_per_year: int = 252
    linkage: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.returns, pd.DataFrame):
            raise TypeError("returns must be a DataFrame of per-period returns "
                            "(dates x assets)")
        if abs(float(self.rf_annual)) >= 0.5:
            raise ValueError(f"rf_annual={self.rf_annual:g} looks like a percentage, not a "
                             f"decimal rate (0.05 = 5%)")
        if int(self.periods_per_year) < 1:
            raise ValueError("periods_per_year must be >= 1")

    # -- rf, converted once, per convention ---------------------------------------
    @property
    def rf_per_period_simple(self) -> float:
        """`rf_annual / periods_per_year` - empyrical's convention, done right."""
        return float(self.rf_annual) / int(self.periods_per_year)

    @property
    def rf_per_period_geometric(self) -> float:
        """`(1+rf)**(1/periods) - 1` - quantstats' de-annualisation."""
        return (1.0 + float(self.rf_annual)) ** (1.0 / int(self.periods_per_year)) - 1.0

    def mu_annual(self) -> pd.Series:
        m = self.returns.mean() if self.mu is None else self.mu
        return m.astype(float) * int(self.periods_per_year)

    def sigma_annual(self) -> pd.DataFrame:
        s = self.returns.cov() if self.sigma is None else self.sigma
        return s.astype(float) * int(self.periods_per_year)


# ------------------------------------------------------------------- the five refusals
def prove_returns(returns: pd.DataFrame, *, linkage: str | None = None) -> None:
    """Raise `PricesWhereReturnsError` unless this really is a per-period returns matrix.

    Two tests. The repo's `weight_traps` guard first: a returns matrix contains negative
    values and a price matrix never does. Then a plausibility band on the column means,
    which catches the other half of the same mistake - returns scaled as percent (5.0 for
    5%), which HAS negative values and is still 100x out.
    """
    res = get("weight_traps").run(asset_returns=returns,
                                  linkage_method=linkage or "single")
    if not res.passed:
        raise PricesWhereReturnsError(
            "; ".join(str(f) for f in res.errors)
            + " -- measured on the repo's own 3-year 6-asset panel (seed 0): 85.18% of the "
              "book into the single highest-volatility name, L1 weight distance 1.5648 of a "
              "possible 2.0, realised annualised vol +141.4% (0.3486 vs 0.1444), worse than "
              "1/N at 0.1976. Pass returns.pct_change().dropna(), not the price panel.")
    worst = float(returns.mean().abs().max())
    if worst > MAX_PLAUSIBLE_MEAN:
        col = returns.mean().abs().idxmax()
        raise PricesWhereReturnsError(
            f"column {col!r} has a mean of {worst:.4g} per period, which is not a return - "
            f"the band this bridge accepts is |mean| <= {MAX_PLAUSIBLE_MEAN}. Returns scaled "
            f"as PERCENT (5.0 meaning 5%) pass the negative-value test and are still 100x "
            f"out; every covariance downstream is then 10,000x out. Divide by 100.")


def require_linkage(model: str, linkage: str | None) -> str:
    """Hierarchical models need an explicit linkage; the defaults disagree by design."""
    if model.upper() not in HIERARCHICAL:
        return linkage or ""
    if not linkage:
        raise LinkageRequiredError(
            f"model={model!r} is hierarchical and `linkage` is required. PyPortfolioOpt's "
            f"HRPOpt defaults to 'single' (Lopez de Prado's original) and skfolio's "
            f"HierarchicalRiskParity defaults to Ward: the two libraries give different "
            f"weights BY DESIGN, not by bug. State which one you mean - reproducing AFML is "
            f"'single', reproducing a skfolio notebook is 'ward'.")
    return str(linkage).lower()


def require_solver_class(model: str, rm: str | None, kw: Mapping[str, Any]) -> str:
    """Map model/risk-measure to the cone class it needs and check a solver provides it."""
    need_cone = bool(rm) and str(rm).upper() in {m.upper() for m in CONE_MEASURES}
    need_mip = any(k in kw for k in ("card", "cardinality", "nea", "buy_in"))
    if not (need_cone or need_mip):
        return ""
    wanted = CONE_SOLVERS if need_cone else MIP_SOLVERS
    what = (f"risk measure {rm!r} needs exponential- or power-cone support"
            if need_cone else "a cardinality / buy-in constraint needs a MIP solver")
    if importlib.util.find_spec("cvxpy") is None:
        warnings.warn(f"{what} and cvxpy is not importable, so the solver class could not "
                      f"be checked here; the backend will report 'solver did not converge' "
                      f"if none of {list(wanted)} is installed.", stacklevel=3)
        return ""
    import cvxpy                                                 # noqa: PLC0415 - lazy
    have = {s.upper() for s in cvxpy.installed_solvers()}
    ok = [s for s in wanted if s in have]
    if not ok:
        raise SolverClassError(
            f"{what}, and none of {list(wanted)} is installed (cvxpy has "
            f"{sorted(have)}). The backend would report 'solver did not converge', which "
            f"names the wrong problem: OSQP cannot EXPRESS these cones. "
            f"pip install clarabel  (or scs).")
    return ok[0]


# ------------------------------------------------------------------------- the backends
def _weights(raw: Any, columns: pd.Index, backend: str) -> pd.Series:
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
    if isinstance(raw, Mapping):
        raw = pd.Series(raw, dtype=float)
    w = pd.Series(np.asarray(raw, dtype=float).ravel(), index=columns) \
        if not isinstance(raw, pd.Series) else raw.astype(float)
    w = w.reindex(columns).fillna(0.0)
    w.name = backend
    return w


def _pypfopt(inp: OptimizerInput, model: str, linkage: str, kw: dict) -> tuple[pd.Series, dict]:
    mod = _lazy.need("PyPortfolioOpt", why=f"backend='pypfopt', model={model!r}")
    m = model.upper() if model.upper() in HIERARCHICAL else model.lower()
    if m == "HRP":
        from pypfopt import HRPOpt                                # noqa: PLC0415 - lazy
        raw = HRPOpt(returns=inp.returns).optimize(linkage_method=linkage)
        note = {"linkage": linkage, "rf_used": None,
                "rf_convention": "not used by HRP"}
    elif m in ("min_vol", "max_sharpe"):
        from pypfopt import EfficientFrontier                     # noqa: PLC0415 - lazy
        ef = EfficientFrontier(inp.mu_annual(), inp.sigma_annual(),
                               weight_bounds=kw.pop("weight_bounds", (0, 1)))
        if m == "min_vol":
            ef.min_volatility()
            note = {"rf_used": None, "rf_convention": "not used by min_volatility"}
        else:
            ef.max_sharpe(risk_free_rate=float(inp.rf_annual))
            note = {"rf_used": float(inp.rf_annual),
                    "rf_convention": "ANNUAL, passed through (PyPortfolioOpt's own "
                                     "convention; mu and S are annualised to match)"}
        raw = ef.clean_weights()
    else:
        raise ValueError(f"model={model!r} is not bridged for pypfopt; "
                         f"try 'HRP', 'min_vol' or 'max_sharpe'")
    note["library_version"] = _lazy.version_of(mod, "pypfopt")
    return _weights(raw, inp.returns.columns, "pypfopt"), note


def _skfolio(inp: OptimizerInput, model: str, linkage: str, kw: dict) -> tuple[pd.Series, dict]:
    mod = _lazy.need("skfolio", why=f"backend='skfolio', model={model!r}")
    m = model.upper() if model.upper() in HIERARCHICAL else model.lower()
    rf = inp.rf_per_period_simple
    if m in ("HRP", "HERC"):
        from skfolio.cluster import HierarchicalClustering, LinkageMethod  # noqa: PLC0415
        from skfolio.optimization import (HierarchicalEqualRiskContribution,  # noqa: PLC0415
                                          HierarchicalRiskParity)
        cluster = HierarchicalClustering(linkage_method=LinkageMethod(linkage))
        cls = HierarchicalRiskParity if m == "HRP" else HierarchicalEqualRiskContribution
        est = cls(hierarchical_clustering_estimator=cluster, **kw)
        note = {"linkage": linkage, "rf_used": None, "rf_convention": "not used by HRP/HERC"}
    elif m in ("min_vol", "max_sharpe"):
        from skfolio.optimization import (MeanRisk,               # noqa: PLC0415 - lazy
                                          ObjectiveFunction)
        obj = (ObjectiveFunction.MINIMIZE_RISK if m == "min_vol"
               else ObjectiveFunction.MAXIMIZE_RATIO)
        est = MeanRisk(objective_function=obj, risk_free_rate=rf, **kw)
        note = {"rf_used": rf,
                "rf_convention": "rf_annual / periods_per_year (skfolio works on "
                                 "per-period returns)"}
    else:
        raise ValueError(f"model={model!r} is not bridged for skfolio; "
                         f"try 'HRP', 'HERC', 'min_vol' or 'max_sharpe'")
    est.fit(inp.returns)
    note["library_version"] = _lazy.version_of(mod, "skfolio")
    return _weights(getattr(est, "weights_"), inp.returns.columns, "skfolio"), note


def _riskfolio(inp: OptimizerInput, model: str, linkage: str,
               kw: dict) -> tuple[pd.Series, dict]:
    """Riskfolio, with `assets_stats` bound to the data and no handle left behind."""
    mod = _lazy.need("Riskfolio-Lib", why=f"backend='riskfolio', model={model!r}")
    rf = inp.rf_per_period_simple
    m = model.upper() if model.upper() in HIERARCHICAL else model.lower()
    if m in ("HRP", "HERC", "NCO"):
        port = mod.HCPortfolio(returns=inp.returns)
        raw = port.optimization(model=m, linkage=linkage, rf=rf,
                                rm=kw.pop("rm", "MV"),
                                codependence=kw.pop("codependence", "pearson"), **kw)
        note = {"linkage": linkage, "rf_used": rf}
    else:
        port = mod.Portfolio(returns=inp.returns)
        # THE state trap: assets_stats must precede optimization, and must be re-called
        # after the data changes. Both happen here, in one step, on a handle that is
        # created and destroyed inside this function.
        port.assets_stats(method_mu=kw.pop("method_mu", "hist"),
                          method_cov=kw.pop("method_cov", "hist"))
        obj = {"min_vol": "MinRisk", "max_sharpe": "Sharpe"}.get(m)
        if obj is None:
            raise ValueError(f"model={model!r} is not bridged for riskfolio; try 'HRP', "
                             f"'HERC', 'NCO', 'min_vol' or 'max_sharpe'")
        raw = port.optimization(model=kw.pop("rp_model", "Classic"), rm=kw.pop("rm", "MV"),
                                obj=obj, rf=rf, hist=kw.pop("hist", True), **kw)
        note = {"rf_used": rf}
    note["rf_convention"] = ("rf_annual / periods_per_year - Riskfolio does not document "
                             "the units, and it works on per-period returns throughout")
    note["library_version"] = _lazy.version_of(mod, "riskfolio")
    return _weights(raw, inp.returns.columns, "riskfolio"), note


_DISPATCH = {"pypfopt": _pypfopt, "skfolio": _skfolio, "riskfolio": _riskfolio}


def optimize(inp: OptimizerInput, *, backend: str, model: str, **kw: Any) -> pd.Series:
    """Weights indexed by asset, with the five refusals applied before any backend runs.

    backend  'pypfopt' | 'skfolio' | 'riskfolio'
    model    'HRP' | 'HERC' | 'NCO' | 'min_vol' | 'max_sharpe' (per-backend support varies)
    **kw     forwarded to the backend after the checks. `rm=` selects the risk measure
             where the backend supports one.

    The returned Series carries `.attrs`: the backend, its version, the linkage actually
    used, the risk-free rate handed to the backend and the conversion that produced it.
    That record is the point - `rf_annual=0.05` means three different numbers in three
    libraries and none of them says which.
    """
    if not isinstance(inp, OptimizerInput):
        raise TypeError("inp must be an OptimizerInput")
    if backend not in _DISPATCH:
        raise ValueError(f"backend must be one of {list(BACKENDS)}, got {backend!r}")

    prove_returns(inp.returns, linkage=inp.linkage)                       # (1)
    linkage = require_linkage(model, inp.linkage)                         # (2)
    solver = require_solver_class(model, kw.get("rm"), kw)                # (5)

    try:                                                                  # (3) and (4)
        w, note = _DISPATCH[backend](inp, model, linkage, dict(kw))
    except (_lazy.MissingLibrary, PricesWhereReturnsError, LinkageRequiredError,
            SolverClassError, ValueError, TypeError):
        raise
    except Exception as exc:                                              # noqa: BLE001
        raise BackendError(f"{_PIP[backend]} refused model={model!r}: "
                           f"{type(exc).__name__}: {exc}") from exc

    w.attrs.update({"backend": backend, "pip": _PIP[backend], "model": model,
                    "rf_annual": float(inp.rf_annual),
                    "periods_per_year": int(inp.periods_per_year),
                    "solver_class": solver, **note})
    return w


def weights_to_bundle(w: pd.Series, returns: pd.DataFrame, **extra: Any) -> Bundle:
    """Weights + the returns they were fit on -> Bundle.

    Fills `asset_returns` (which unlocks `weight_traps`, so the guard runs on the way OUT
    as well as on the way in), `returns` (the weighted portfolio, held constant) and
    `turnover` (the one-way cost of getting into the book on day one, and zero after -
    a static allocation does not trade, and pretending it rebalances every period would
    make `cost_curve` price a strategy nobody ran).
    """
    if not isinstance(w, pd.Series):
        raise TypeError("w must be a Series of weights indexed by asset")
    aligned = w.reindex(returns.columns).fillna(0.0)
    port = (returns.astype(float) * aligned).sum(axis=1)
    turn = pd.Series(0.0, index=returns.index)
    if len(turn):
        turn.iloc[0] = float(aligned.abs().sum())
    slots: dict[str, Any] = {"asset_returns": returns, "returns": port, "turnover": turn}
    ppy = w.attrs.get("periods_per_year")
    if ppy:
        slots["periods_per_year"] = int(ppy)
    rf = w.attrs.get("rf_annual")
    if rf is not None:
        slots["rf"] = float(rf)
    slots.update(extra)
    return Bundle(**slots)


__all__ = ["BACKENDS", "BackendError", "CONE_MEASURES", "HIERARCHICAL", "LICENCE",
           "LIBRARY", "LinkageRequiredError", "MAX_PLAUSIBLE_MEAN", "OptimizerInput",
           "PricesWhereReturnsError", "SolverClassError", "VERIFIED_ON", "optimize",
           "prove_returns", "require_linkage", "require_solver_class", "weights_to_bundle"]
