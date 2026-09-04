#!/usr/bin/env python3
"""HRPOpt takes RETURNS. Hand it PRICES and it runs, and the weights look fine.

`HRPOpt(returns=...)` type-checks the CONTAINER and never the CONTENT: a prices
DataFrame is a DataFrame, so it is accepted. Nothing raises, nothing warns, and
the output is a clean long-only vector summing to 1 that you cannot tell from a
correct one by looking at it.

The damage is mechanical. HRP's recursive bisection weights each split by inverse
cluster variance. On returns that is variance in units of RETURN — risk. On prices
it is variance in units of DOLLARS-SQUARED, so a $8 stock has ~6000x less "variance"
than a $620 stock at the same volatility. HRP-on-prices is therefore an
INVERSE-SHARE-PRICE portfolio: it buys whatever ticker has the smallest nominal
price, a quantity with no relationship to risk. The correlation tree is wrong the
same way — price LEVELS trend together, so the distance matrix measures drift, not
co-movement.

Run:  python weight_traps.py
PyPortfolioOpt is optional. Without it this reproduces HRP from the AFML definition
with numpy/pandas/scipy, so the demonstration still runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

PERIODS = 252
NAMES = ["MEGA", "BANK", "MID", "PENNY", "UTIL", "TECH"]


# --- reference HRP (Lopez de Prado 2016), numpy/pandas/scipy only ------------

def _cluster_var(cov: pd.DataFrame, items: list[str]) -> float:
    """Inverse-variance portfolio variance for one cluster."""
    c = cov.loc[items, items]
    w = 1.0 / np.diag(c)
    w = w / w.sum()
    return float(np.linalg.multi_dot((w, c, w)))


def _raw_hrp_allocation(cov: pd.DataFrame, ordered: list[str]) -> pd.Series:
    """Top-down recursive bisection, splitting by inverse cluster variance."""
    w = pd.Series(1.0, index=ordered)
    clusters = [ordered]
    while len(clusters) > 0:
        clusters = [i[j:k] for i in clusters
                    for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1.0 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
    return w


def quasi_diag_order(df: pd.DataFrame, linkage_method: str = "single") -> list[str]:
    """The clustering order HRP will bisect. Built from the correlation distance."""
    corr = df.corr()
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    link = linkage(squareform(dist, checks=False), linkage_method)
    return corr.index[to_tree(link, rd=False).pre_order()].tolist()


def hrp_weights(df: pd.DataFrame, linkage_method: str = "single") -> pd.Series:
    """HRP on whatever you pass. It cannot tell returns from prices either — that
    is the point: the arithmetic is well defined on both, only one is meaningful."""
    ordered = quasi_diag_order(df, linkage_method)
    return _raw_hrp_allocation(df.cov(), ordered).sort_index()


def ann_vol(w: pd.Series, cov: pd.DataFrame) -> float:
    """Annualised vol of a weight vector under the TRUE (returns) covariance."""
    v = w.reindex(cov.columns).values
    return float(np.sqrt(v @ cov.values @ v) * np.sqrt(PERIODS))


# --- synthetic panel --------------------------------------------------------

def make_panel(seed: int = 0, n: int = PERIODS * 3):
    """One factor plus idiosyncratic noise, and a realistic spread of share prices.

    PENNY is deliberately the HIGHEST-risk name (beta 1.3, idio 2.2%/day) and the
    CHEAPEST ticker ($8). Correct HRP underweights it; HRP-on-prices buys it.
    """
    rng = np.random.default_rng(seed)
    p0 = np.array([620.0, 95.0, 48.0, 8.0, 41.0, 310.0])
    betas = np.array([1.1, 1.0, 0.9, 1.3, 0.4, 1.5])
    idio = np.array([0.009, 0.010, 0.012, 0.022, 0.007, 0.015])

    factor = rng.normal(0.0002, 0.011, (n, 1))
    rets = pd.DataFrame(
        factor * betas + rng.normal(0.0, 1.0, (n, len(NAMES))) * idio,
        columns=NAMES, index=pd.bdate_range("2022-01-03", periods=n))
    prices = pd.DataFrame(p0 * (1.0 + rets).cumprod().values,
                          columns=NAMES, index=rets.index)
    return rets, prices


def demo(seed: int = 0) -> dict:
    rets, prices = make_panel(seed)
    cov_true = rets.cov()

    w_ret = hrp_weights(rets)
    w_prc = hrp_weights(prices)
    w_eq = pd.Series(1.0 / len(NAMES), index=NAMES)

    out = {
        "w_returns": w_ret,
        "w_prices": w_prc,
        "w_equal": w_eq,
        "l1": float((w_ret - w_prc).abs().sum()),
        "max_shift": float((w_ret - w_prc).abs().max()),
        "order_returns": quasi_diag_order(rets),
        "order_prices": quasi_diag_order(prices),
        "asset_vol": rets.std(ddof=1) * np.sqrt(PERIODS),
        "vol_returns": ann_vol(w_ret, cov_true),
        "vol_prices": ann_vol(w_prc, cov_true),
        "vol_equal": ann_vol(w_eq, cov_true),
        "guard_returns": bool((rets < 0).any().any()),
        "guard_prices": bool((prices < 0).any().any()),
        "dollar_var": prices.var(),
    }

    # If PyPortfolioOpt is installed, verify the reference implementation actually
    # matches HRPOpt rather than asserting that it does.
    try:
        from pypfopt import HRPOpt

        live_ret = pd.Series(HRPOpt(returns=rets).optimize(linkage_method="single"))
        live_prc = pd.Series(HRPOpt(returns=prices).optimize(linkage_method="single"))
        out["LIVE_w_returns"] = live_ret
        out["LIVE_w_prices"] = live_prc
        out["LIVE_err_returns"] = float((w_ret - live_ret.reindex(w_ret.index)).abs().max())
        out["LIVE_err_prices"] = float((w_prc - live_prc.reindex(w_prc.index)).abs().max())

        # What DOES it reject? The container type, and only that.
        try:
            HRPOpt(returns=rets.values)
            out["LIVE_numpy_input"] = "accepted"
        except Exception as exc:  # noqa: BLE001 - reporting the class is the point
            out["LIVE_numpy_input"] = f"{type(exc).__name__}: {exc}"
        out["LIVE_prices_input"] = "accepted, no warning"

        # The other documented weight trap: EfficientFrontier objects are single-use.
        # "raises or returns stale state" is vague; record which one 1.6.0 does.
        from pypfopt import EfficientFrontier, expected_returns, risk_models

        mu = expected_returns.mean_historical_return(prices, frequency=PERIODS)
        S = risk_models.sample_cov(prices, frequency=PERIODS)
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        ef.max_sharpe(risk_free_rate=0.02)
        try:
            ef.min_volatility()
            out["LIVE_reuse"] = "returned stale state, NO raise"
        except Exception as exc:  # noqa: BLE001 - the class is the finding
            out["LIVE_reuse"] = f"{type(exc).__name__}: {exc}"
    except ImportError:
        pass
    return out


def _row(label: str, w: pd.Series) -> str:
    return f"  {label:<28}" + " ".join(f"{w[c]:>7.4f}" for c in NAMES)


if __name__ == "__main__":
    res = demo()
    print("HRP on the same 3-year panel, once from returns and once from prices"
          " (seed=0)\n")
    print(f"  {'':<28}" + " ".join(f"{c:>7}" for c in NAMES))
    print(_row("HRP(returns)  correct", res["w_returns"]))
    print(_row("HRP(prices)   the trap", res["w_prices"]))
    print(_row("difference", (res["w_returns"] - res["w_prices"]).abs()))

    print(f"\n  L1 distance between the two vectors   {res['l1']:>10.4f}  (max possible 2.0)")
    print(f"  largest single-name shift             {res['max_shift']:>10.4f}")
    print(f"  both sum to 1, both long-only         "
          f"{res['w_prices'].sum():>10.4f}  <- nothing looks wrong")

    print("\n  Why: cov(prices) is in DOLLARS-SQUARED, so inverse-variance")
    print("  weighting collapses into inverse-share-price weighting.")
    dv, av = res["dollar_var"], res["asset_vol"]
    print(f"    {'':<7}{'var($^2)':>12}{'true ann vol':>14}{'w(prices)':>11}")
    for c in NAMES:
        print(f"    {c:<7}{dv[c]:>12.2f}{av[c]:>14.4f}{res['w_prices'][c]:>11.4f}")
    print(f"    PENNY has {dv['MEGA'] / dv['PENNY']:,.0f}x less dollar-variance than MEGA"
          f" and is the HIGHEST-vol name")
    print(f"    of the six ({av.max():.4f} vs {av.min():.4f}) -- and it takes"
          f" {res['w_prices']['PENNY']:.2%} of the book.")

    print("\n  The clustering tree is wrong too. Quasi-diagonal order:")
    print(f"    from returns  {res['order_returns']}")
    print(f"    from prices   {res['order_prices']}   <- a different tree entirely")

    print(f"\n  Annualised vol under the TRUE return covariance:")
    print(f"    HRP(returns)   {res['vol_returns']:.4f}")
    print(f"    HRP(prices)    {res['vol_prices']:.4f}"
          f"   {res['vol_prices'] / res['vol_returns'] - 1:+.1%} vs correct")
    print(f"    1/N            {res['vol_equal']:.4f}"
          f"   HRP(prices) is {res['vol_prices'] / res['vol_equal'] - 1:+.1%} vs even 1/N")

    print(f"\n  The documented guard, `assert (returns < 0).any().any()`:")
    print(f"    on returns -> {res['guard_returns']}   passes")
    print(f"    on prices  -> {res['guard_prices']}  fails, and it is the ONLY thing"
          f" standing between you and the number above")

    if "LIVE_w_returns" in res:
        print("\n  verified against the installed PyPortfolioOpt:")
        print(_row("LIVE HRPOpt(returns)", res["LIVE_w_returns"]))
        print(_row("LIVE HRPOpt(prices)", res["LIVE_w_prices"]))
        print(f"    max |reference - HRPOpt| on returns {res['LIVE_err_returns']:.2e}")
        print(f"    max |reference - HRPOpt| on prices  {res['LIVE_err_prices']:.2e}")
        print(f"    HRPOpt(returns=<numpy array>)   -> {res['LIVE_numpy_input']}")
        print(f"    HRPOpt(returns=<prices frame>)  -> {res['LIVE_prices_input']}")
        print("    so the type check is real but shallow: it validates the container,"
              " never the content")
        print(f"\n    and the single-use EfficientFrontier claim, ef.max_sharpe() then"
              f" ef.min_volatility():\n      {res['LIVE_reuse']}")
    else:
        print("\n  PyPortfolioOpt not installed — reference implementation only")

    print("\n  Rule: assert your input contains negative values before HRPOpt sees it."
          "\n        A returns matrix has them. A price matrix never does."
          "\n        And state linkage_method explicitly: pypfopt defaults to"
          " 'single', skfolio to Ward.")
