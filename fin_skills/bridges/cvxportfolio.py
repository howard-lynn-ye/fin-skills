"""Native Cvxportfolio decisions over caller-provided data, never a broker connection."""
from pathlib import Path

import numpy as np
import pandas as pd

from fin_skills.algorithms.runtime import integer, number
from fin_skills.model_zoo.upstream_catalog import CVX


def make_market_data(returns, *, cache_dir, prices=None, volumes=None, cash_key="USDOLLAR"):
    """No market downloads. Explicit cash returns and writable cache directory are required."""
    import cvxportfolio as cvx
    if not isinstance(returns, pd.DataFrame) or cash_key not in returns.columns:
        raise ValueError("returns must explicitly include the cash return column")
    if not isinstance(returns.index, pd.DatetimeIndex) or not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError("returns require ordered unique datetimes")
    if returns.columns.has_duplicates or len(returns) < 3 or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("returns require finite observations and unique columns")
    if (returns.to_numpy() < -1).any():
        raise ValueError("returns must be decimal simple returns")
    directory = Path(cache_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return cvx.UserProvidedMarketData(returns=returns, prices=prices, volumes=volumes,
        cash_key=cash_key, min_history=pd.Timedelta(0), base_location=directory)


def native_policy(method_id, **parameters):
    """Construct an allowlisted upstream policy; advanced native objectives stay Python-only."""
    import cvxportfolio as cvx
    if method_id not in CVX:
        raise KeyError(f"unknown Cvxportfolio policy: {method_id}")
    p = dict(parameters)
    cls = getattr(cvx, CVX[method_id])
    if method_id in ("single_period_optimization", "multi_period_optimization"):
        if "objective" not in p:
            if not {"expected_returns", "covariance"} <= set(p):
                raise ValueError("supply an explicit native objective or expected_returns and covariance")
            mu, covariance = p.pop("expected_returns"), p.pop("covariance")
            risk_aversion = number(p.pop("risk_aversion", 1.), "risk_aversion", minimum=0)
            cost = number(p.pop("trading_cost_bps", 0.), "trading_cost_bps", minimum=0)
            p["objective"] = (cvx.ReturnsForecast(mu) - risk_aversion * cvx.FullCovariance(covariance)
                              - cvx.TransactionCost(a=cost / 10000, b=None))
            p.setdefault("constraints", [cvx.LongOnly(), cvx.LeverageLimit(1.)])
            p.setdefault("include_cash_return", False)
        if method_id == "multi_period_optimization":
            p["planning_horizon"] = integer(p.get("planning_horizon", 2), "planning_horizon", maximum=100)
    policy = cls(**p)
    if method_id == "adaptive_rebalance":
        # Cvxportfolio 1.5.1 constructs this target without the cash entry but subtracts
        # it from cash-inclusive holdings. Configure its native DataEstimator explicitly.
        # No replacement decision logic or changes to the installed package.
        policy.target = type(policy.target)(p["target"], data_includes_cash=True)
    return policy
