"""Performance metrics, with the conventions this repo already uses elsewhere.

Two choices here are deliberate and worth stating:

  Sharpe is computed on EXCESS return over the 2% risk-free rate. Reporting a
  raw Sharpe in a market where idle cash pays 2% flatters every fully-invested
  arm and makes the cash arm look like it earned nothing.

  Turnover is ONE-WAY: 0.5 * sum |w_t - w_{t-1}|, matching the convention that
  alpha_combine.py and cost_curve.py consume. Since the broker records the
  gross notional of both legs, one-way turnover is gross/2 divided by average
  equity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import RISK_FREE_ANNUAL, TRADING_DAYS


def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    return float((nav / peak - 1.0).min())


def summarise(nav: pd.Series, account, n_days: int) -> dict:
    nav = nav.astype(float)
    rets = nav.pct_change().dropna()
    years = n_days / TRADING_DAYS

    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol = float(rets.std() * np.sqrt(TRADING_DAYS)) if len(rets) > 1 else 0.0

    rf_daily = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / TRADING_DAYS) - 1.0
    excess = rets - rf_daily
    sharpe = (float(excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)
              if len(excess) > 1 and excess.std() > 0 else 0.0)

    mean_equity = float(nav.mean())
    one_way = account.gross_traded / 2.0
    turnover = one_way / mean_equity if mean_equity > 0 else 0.0

    return {
        "total_return": total,
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe_excess": sharpe,
        "max_drawdown": max_drawdown(nav),
        "final_equity": float(nav.iloc[-1]),
        "turnover_one_way_total": turnover,
        "turnover_one_way_annual": turnover / years if years > 0 else 0.0,
        "friction": {
            "stamp_duty": account.stamp_duty_paid,
            "commission": account.commission_paid,
            "slippage": account.slippage_paid,
            "total": account.friction_paid,
            "pct_of_initial": account.friction_paid / account.initial_cash,
        },
        "trades": {
            "buys": account.n_buys,
            "sells": account.n_sells,
            "blocked_by_price_limit": account.blocked_by_limit,
        },
    }
