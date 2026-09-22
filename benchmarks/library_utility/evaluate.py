"""Independent portfolio ledger for the library-utility feasibility experiment.

ECB observations are indicative reference rates, not fills. Results from this dataset
are price-only proxies excluding interest/carry, not executable trading returns.
No fin_skills imports are permitted in this accounting module.
"""
import hashlib
import json
import math

import numpy as np
import pandas as pd


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def allocation(value, n_assets):
    if not isinstance(value, list) or len(value) != n_assets:
        raise ValueError("weights must be a list with one number per asset")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           or not 0 <= x <= 1 for x in value):
        raise ValueError("weights must be finite numbers between zero and one")
    if sum(value) > 1 + 1e-10:
        raise ValueError("weights exceed unlevered long-only budget")
    return np.array(value, dtype=float)


def ledger(prices, decisions, *, start, end, cost_bps=5.0):
    """Decision after observation t executes at reference t+1, never earns t->t+1.

    A target is a fraction of AFTER-fee NAV. Solve the self-financing fee equation
    instead of subtracting a fee and treating the portfolio as still fully funded.
    Invalid decisions are explicit no-rebalance events; terminal liquidation is charged.
    """
    px = np.asarray(prices, dtype=float)
    if px.ndim != 2 or not np.isfinite(px).all() or (px <= 0).any():
        raise ValueError("positive finite two-dimensional prices required")
    if not 0 <= start < end < len(px) or not 0 <= cost_bps <= 100:
        raise ValueError("invalid evaluation bounds or cost")
    fills = {}
    for row in decisions:
        t = row["decision_index"]
        if type(t) is not int or not start <= t < end:
            raise ValueError("decision outside declared evaluation interval")
        if t + 1 in fills:
            raise ValueError("duplicate execution timestamp")
        fills[t + 1] = None if row["weights"] is None else allocation(row["weights"], px.shape[1])
    cash, units, previous = 1.0, np.zeros(px.shape[1]), 1.0
    rates, navs, turnovers, exposures = [], [], [], []
    c = cost_bps / 10000
    for t in range(start + 1, end + 1):
        holdings = units * px[t]
        before = cash + holdings.sum()
        turnover = 0.0
        target = fills.get(t)
        if target is not None:
            lower, upper = 0.0, before
            for _ in range(60):
                after = (lower + upper) / 2
                balance = after + c * np.abs(after * target - holdings).sum() - before
                if balance > 0:
                    upper = after
                else:
                    lower = after
            after = (lower + upper) / 2
            turnover = float(np.abs(after * target - holdings).sum() / before)
            units = after * target / px[t]
            cash = after * (1 - target.sum())
        nav = float(cash + units @ px[t])
        exposure = float(units @ px[t] / nav)
        if t == end:
            exit_value = float(units @ px[t])
            turnover += exit_value / nav
            nav -= c * exit_value
        rates.append(nav / previous - 1)
        navs.append(nav)
        turnovers.append(turnover)
        exposures.append(exposure)
        previous = nav
    r = np.array(rates)
    nav = np.r_[1., navs]
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return {"daily_returns": rates, "nav": navs, "daily_turnover": turnovers,
            "metrics": {"total_return": float(nav[-1] - 1),
                "annualized_vol": float(sd * np.sqrt(252)),
                "sharpe_zero_cash_rate": float(r.mean() / sd * np.sqrt(252)) if sd > 1e-12 else None,
                "max_drawdown": float(np.min(nav / np.maximum.accumulate(nav) - 1)),
                "total_turnover": float(sum(turnovers)),
                "mean_risky_exposure": float(np.mean(exposures)), "observations": len(r)},
            "cost_bps": cost_bps, "execution_lag_observations": 1,
            "return_label": "indicative price-only proxy; not executable total return"}


def paired_blocks(a, b, *, block=21, draws=1000, seed=0):
    """Paired circular date-block interval; descriptive for this short feasibility run."""
    delta = np.asarray(a) - np.asarray(b)
    if delta.ndim != 1 or len(delta) < block or not np.isfinite(delta).all():
        raise ValueError("insufficient paired observations")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        starts = rng.integers(0, len(delta), size=math.ceil(len(delta) / block))
        idx = ((starts[:, None] + np.arange(block)) % len(delta)).ravel()[:len(delta)]
        samples.append(delta[idx].mean())
    return {"mean_daily_return_difference": float(delta.mean()),
            "descriptive_95pct_interval": np.quantile(samples, [.025, .975]).tolist(),
            "block_observations": block, "draws": draws,
            "inference_scope": "one correlated market path, not independent model-seed samples"}
