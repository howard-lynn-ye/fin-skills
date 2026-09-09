#!/usr/bin/env python3
"""Trend-following models, and the two look-aheads that flatter their backtests.

Time-series momentum (TSMOM) as Moskowitz, Ooi and Pedersen (2012) define it, a Donchian
breakout, a moving-average crossover and volatility targeting, all run on one seeded panel
of synthetic futures with persistent trends and clustered volatility. Nothing here is a
claim about real markets: the panel exists so that each trap can be measured against the
same strategy without it.

The two traps, both of which produce a plausible equity curve and no error:

  1. sizing the position that earns r_t with a volatility estimate that already contains
     r_t - sigma_t instead of the paper's sigma_{t-1} (the vol series was never shifted);
  2. crediting r_t to a position computed from close_t (the missing .shift(1)), and its
     milder cousin, filling at the close that produced the signal instead of the next open.

MOP (2012) conventions reproduced here, read in the paper (J. Financial Economics 104
(2012) 228-250; quotations from the author copy at
w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf):

  * eq. (1), section 2.4:  sigma_t^2 = 261 * sum_{i=0..inf} (1 - d) d^i (r_{t-1-i} - rbar_t)^2
    -- "the scalar 261 scales the variance to be annual, the weights (1-d)d^i add up to
    one, and rbar_t is the exponentially weighted average return computed similarly. The
    parameter d is chosen so that the center of mass of the weights is
    sum_i (1-d) d^i i = d/(1-d) = 60 days."
  * section 2.4: "To ensure no look-ahead bias contaminates our results, we use the
    volatility estimates at time t-1 applied to time-t returns throughout the analysis."
    Note that eq. (1) already sums over r_{t-1-i}, so sigma_t excludes r_t by construction.
  * eq. (5), section 4.1:  r^{TSMOM,s}_{t,t+1} = sign(r^s_{t-12,t}) * (40% / sigma^s_t)
    * r^s_{t,t+1}   -- the full past 12 months, NOTHING SKIPPED, with a 1-month holding
    period ("k = 12 and h = 1"). The prose one paragraph above eq. (5) writes the same
    position size as "40%/sigma_{t-1}"; the two agree because of eq. (1)'s indexing.
  * section 4.1 on the 40%: "The choice of 40% is inconsequential, but it makes it easier
    to intuitively compare our portfolios to others in the literature."
  * footnote 10 (p. 240) is the ONLY place a skipped month appears, and it is about the
    CROSS-sectional benchmark: "Asness, Moskowitz, and Pedersen (2010) exclude the most
    recent month when computing 12-month cross-sectional momentum. For consistency, we
    follow that convention here, but our results do not depend on whether the most recent
    month is excluded or not."

Run:  python trend_models.py          (numpy + pandas only, fixed seed, a few seconds)
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

SEED = 20260908
PERIODS_PER_YEAR = 261      # MOP eq. (1): "the scalar 261 scales the variance to be annual"
COM_DAYS = 60.0             # MOP eq. (1): centre of mass d/(1 - d) = 60 days
VOL_TARGET = 0.40           # MOP section 4.1: 40% ex-ante annualised volatility per position
LOOKBACK = 261              # 12 months of daily bars on this 261-day calendar
MONTHS_PER_YEAR = 12


# ----------------------------------------------------------------------- synthetic panel
def simulate_panel(n_assets: int = 10, n_years: int = 20, seed: int = SEED,
                   trend_sd: float = 0.05, phi: float = 0.997,
                   overnight_share: float = 0.30) -> dict:
    """Daily bars for a panel of synthetic futures with persistent trends and GARCH volatility.

    Each asset's daily log return is mu_t + sigma_t * z: mu_t is an AR(1) drift with
    autocorrelation `phi` (half-life about 230 days at 0.997) and stationary standard
    deviation `trend_sd` * sigma; sigma_t^2 follows a GARCH(1,1) with persistence 0.98
    around unconditional levels spread from 10% to 40% annualised across assets. The return
    is split into an overnight (close -> open) and an intraday (open -> close) part so that
    a fill at the next open can be priced against a fill at the signal's own close.
    """
    if n_assets < 1 or n_years < 1:
        raise ValueError("need at least one asset and one year")
    if not 0.0 < overnight_share < 1.0:
        raise ValueError("overnight_share must be strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    n = n_years * PERIODS_PER_YEAR
    ann_vol = np.linspace(0.10, 0.40, n_assets)
    dvol = ann_vol / np.sqrt(PERIODS_PER_YEAR)
    g_a, g_b = 0.06, 0.92
    omega = dvol ** 2 * (1.0 - g_a - g_b)
    var = dvol ** 2
    mu = np.zeros(n_assets)
    on = np.empty((n, n_assets))
    day = np.empty((n, n_assets))
    drift = np.empty((n, n_assets))
    sq = np.sqrt(1.0 - phi ** 2)
    s_on, s_day = np.sqrt(overnight_share), np.sqrt(1.0 - overnight_share)
    for t in range(n):
        mu = phi * mu + sq * trend_sd * dvol * rng.standard_normal(n_assets)
        sig = np.sqrt(var)
        z = rng.standard_normal((2, n_assets))
        on[t] = overnight_share * mu + s_on * sig * z[0]
        day[t] = (1.0 - overnight_share) * mu + s_day * sig * z[1]
        r = on[t] + day[t]
        var = omega + g_a * r ** 2 + g_b * var
        drift[t] = mu
    idx = pd.bdate_range("2006-01-02", periods=n)
    cols = [f"F{i:02d}" for i in range(n_assets)]
    on = pd.DataFrame(on, idx, cols)
    day = pd.DataFrame(day, idx, cols)
    log_cc = on + day
    close = 100.0 * np.exp(log_cc.cumsum())
    return {
        "overnight": on,
        "intraday": day,
        "returns": np.expm1(log_cc),                    # close_{t-1} -> close_t, simple
        "open_to_open": np.expm1(day + on.shift(-1)),   # open_t -> open_{t+1}, simple
        "close": close,
        "open": close / np.exp(day),
        "drift": pd.DataFrame(drift, idx, cols),
        "ann_vol": pd.Series(ann_vol, index=cols),
    }


# ------------------------------------------------------------------ volatility estimators
def ex_ante_vol(returns, com: float = COM_DAYS, periods_per_year: int = PERIODS_PER_YEAR,
                lag: int = 1):
    """MOP eq. (1) as a pandas EWM: annualised sigma at t from returns through t - lag.

    `lag=1` is the paper. Eq. (1) sums (1-d)d^i (r_{t-1-i} - rbar_t)^2 over i >= 0, i.e.
    r_{t-1}, r_{t-2}, ... -- so sigma_t already excludes r_t, and pandas' EWM value at row
    t-1 is exactly that sum. `lag=0` puts r_t inside the sigma that sizes the position
    earning r_t, which is the trap measured below.

    pandas' `com` IS the paper's centre of mass: pandas decays by 1 - alpha = com/(1+com),
    so d = com/(1+com) and d/(1-d) = com. `var(bias=True)` with `adjust=True` is
    sum_i w_i (r_i - rbar_w)^2 / sum_i w_i, the finite-sample form of eq. (1) with the
    weights renormalised to one. `eq1_variance` below checks the two against each other.
    """
    if lag < 0:
        raise ValueError("lag must be >= 0; a negative lag is a look-ahead by construction")
    var = returns.ewm(com=com, adjust=True).var(bias=True) * periods_per_year
    return np.sqrt(var).shift(lag)


def eq1_variance(r, com: float = COM_DAYS, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """MOP eq. (1) written out literally, evaluated at the last observation of a 1-D series.

    This is the independent reference for `ex_ante_vol`: explicit weights (1-d)d^i on
    r_{t-1}, r_{t-2}, ..., an exponentially weighted mean rbar_t on the same weights, and
    the 261 scaling. Used by the demo to show the two agree to machine precision.
    """
    r = np.asarray(r, dtype=float)
    if r.size < 2:
        raise ValueError("need at least two returns")
    delta = com / (1.0 + com)
    past = r[:-1][::-1]                                 # r_{t-1}, r_{t-2}, ..., r_0
    w = (1.0 - delta) * delta ** np.arange(past.size)
    w = w / w.sum()                                     # renormalise the truncated tail
    rbar = float(np.sum(w * past))
    return float(periods_per_year * np.sum(w * (past - rbar) ** 2))


def rolling_vol(returns, window: int = 20, periods_per_year: int = PERIODS_PER_YEAR,
                lag: int = 1):
    """The everyday alternative, a trailing simple standard deviation, same lag convention."""
    if lag < 0:
        raise ValueError("lag must be >= 0; a negative lag is a look-ahead by construction")
    return (returns.rolling(window).std(ddof=0) * np.sqrt(periods_per_year)).shift(lag)


def ewma_center_of_mass(delta: float, n_terms: int = 20_000) -> float:
    """sum_i (1 - delta) delta^i i - the quantity MOP eq. (1) sets to 60 days."""
    i = np.arange(n_terms)
    return float(np.sum((1.0 - delta) * delta ** i * i))


# ------------------------------------------------------------------------------- signals
def tsmom_signal(returns, lookback: int = LOOKBACK, skip: int = 0):
    """sign of the total return over (t - lookback - skip, t - skip]. MOP: skip = 0.

    MOP eq. (5) is sign(r_{t-12,t}) with a one-month holding period; the lookback runs up
    to the rebalance date and no month is dropped. `skip=21` is the cross-sectional "12-1"
    convention of footnote 10, included only so the two can be compared.
    """
    if lookback < 1 or skip < 0:
        raise ValueError("lookback must be >= 1 and skip >= 0")
    total = np.log1p(returns).rolling(lookback).sum().shift(skip)
    return np.sign(total)


def ma_crossover_signal(close, fast: int = 50, slow: int = 200):
    """+1 when the fast simple moving average is above the slow one, -1 below."""
    if fast >= slow or fast < 1:
        raise ValueError("need 1 <= fast < slow")
    return np.sign(close.rolling(fast).mean() - close.rolling(slow).mean())


def donchian_signal(close, entry: int = 20, exit: int = 10):
    """Long on a close above the prior `entry`-day channel high, flat again on a close below
    the prior `exit`-day channel low; mirror image for shorts. Channels are built from
    closes and lagged one bar, so the bar that breaks out is never inside its own channel."""
    if entry < 2 or exit < 2:
        raise ValueError("entry and exit windows must be >= 2")
    c = close.to_numpy(dtype=float)
    hi_e = close.rolling(entry).max().shift(1).to_numpy()
    lo_e = close.rolling(entry).min().shift(1).to_numpy()
    hi_x = close.rolling(exit).max().shift(1).to_numpy()
    lo_x = close.rolling(exit).min().shift(1).to_numpy()
    out = np.zeros_like(c)
    for j in range(c.shape[1]):
        pos = 0.0
        for t in range(c.shape[0]):
            if np.isnan(hi_e[t, j]):
                continue
            if pos > 0 and c[t, j] < lo_x[t, j]:
                pos = 0.0
            elif pos < 0 and c[t, j] > hi_x[t, j]:
                pos = 0.0
            if pos == 0.0:
                if c[t, j] > hi_e[t, j]:
                    pos = 1.0
                elif c[t, j] < lo_e[t, j]:
                    pos = -1.0
            out[t, j] = pos
    return pd.DataFrame(out, index=close.index, columns=close.columns)


# ------------------------------------------------------------------------------ backtest
def backtest(signal, returns, target: float = VOL_TARGET, vol_fn=ex_ante_vol,
             vol_lag: int = 1, signal_lag: int = 1, size: bool = True):
    """Strategy returns and the position in force while each r_t accrues.

    Correct: signal_lag=1 (the signal from close t-1) and vol_lag=1 (sigma from returns
    through t-1). signal_lag=0 is the missing shift; vol_lag=0 sizes r_t with a sigma that
    contains r_t. Returns (strategy_returns, applied_position).
    """
    if signal_lag < 0 or vol_lag < 0:
        raise ValueError("lags must be >= 0")
    sig = signal.shift(signal_lag)
    pos = sig * target / vol_fn(returns, lag=vol_lag) if size else sig
    return pos * returns, pos


def tsmom_backtest(returns, lookback: int = LOOKBACK, skip: int = 0, **kw):
    """Daily-rebalanced TSMOM. The paper rebalances monthly - see `monthly_tsmom`."""
    return backtest(tsmom_signal(returns, lookback, skip), returns, **kw)


def month_end_mask(index) -> np.ndarray:
    """True on the last row of each calendar month. No frequency alias, so no pandas
    'M' vs 'ME' deprecation to trip over."""
    per = index.to_period("M")
    return np.r_[np.asarray(per[1:] != per[:-1]), True]


def monthly_tsmom(returns, lookback_m: int = 12, skip_m: int = 0, target: float = VOL_TARGET,
                  com: float = COM_DAYS):
    """MOP eq. (5) at the paper's own frequency: 12-month lookback, 1-month holding period.

    The position set at the end of month t is sign(r_{t-lookback_m, t}) * target / sigma_t,
    with sigma_t the daily eq. (1) estimate read at that month end (which by eq. (1) uses
    daily returns through the previous day). It earns the month t -> t+1 return.
    `skip_m=1` drops the most recent month - footnote 10's cross-sectional convention, not
    the TSMOM definition. Returns (monthly strategy returns, month-end position).
    """
    if lookback_m < 1 or skip_m < 0:
        raise ValueError("lookback_m must be >= 1 and skip_m >= 0")
    me = month_end_mask(returns.index)
    level = (1.0 + returns).cumprod()
    m_level = level[me]
    m_ret = m_level.pct_change()                                  # month t-1 -> t
    look = m_level / m_level.shift(lookback_m) - 1.0              # r_{t-lookback_m, t}
    sig = np.sign(look.shift(skip_m))
    sigma = ex_ante_vol(returns, com=com, lag=1)[me]
    pos = sig * target / sigma
    return pos.shift(1) * m_ret, pos


def diversified(strategy_returns):
    """MOP's equal-weighted average across instruments (their 1/S_t)."""
    return strategy_returns.dropna(how="all").mean(axis=1)


def sharpe(r, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(r) -> float:
    w = (1.0 + pd.Series(r).dropna()).cumprod()
    return float((w / w.cummax() - 1.0).min())


def summary(r, periods_per_year: int = PERIODS_PER_YEAR) -> dict:
    r = pd.Series(r).dropna()
    return {"ann_return": float(r.mean() * periods_per_year),
            "ann_vol": float(r.std(ddof=1) * np.sqrt(periods_per_year)),
            "sharpe": sharpe(r, periods_per_year), "max_dd": max_drawdown(r)}


# ---------------------------------------------------------------------- causality harness
def close_from_returns(returns, s0: float = 100.0):
    """Rebuild a close path from simple returns, so a signal defined on prices can be
    recomputed from a perturbed RETURN frame inside `position_is_causal`.

    Accumulated in logs and clipped at +/-700 so that the extreme tail SHOCKS below cannot
    overflow the path to inf. Real return paths never come near the clip; the rows that do
    are all past the perturbation point and are never compared.
    """
    return s0 * np.exp(np.log1p(returns).cumsum().clip(-700.0, 700.0))


SHOCKS = (-0.90, 9.0)       # a -90% day and a +900% day: either sign of trend gets flipped


def position_is_causal(position_fn, frame, k: int, tol: float = 1e-9,
                       shock: float | Sequence[float] | None = None) -> bool:
    """True if the position in force while r_t accrues depends only on rows before t.

    Perturbs every numeric value from row k onward and checks that the applied positions
    for rows <= k did not move. This is signal-construction's assert_causal with a one-bar
    offset: a *signal* at t may use row t, the *position that earns r_t* may not. The
    equivalent call is assert_causal(lambda d: position_fn(d).shift(-1), frame, k).

    `shock=None` is assert_causal's own perturbation, x2.0. That perturbation cannot see a
    leak through np.sign(): doubling one daily return moves a 12-month sum by a fraction of
    a percent, the sign does not flip, and the test passes on a position that is reading
    the future. Pass `shock=SHOCKS` to OVERWRITE rows >= k with each extreme return in turn
    and report a leak if any of them moves an earlier position.

    Both directions are needed. A single -90% day only flips a signal that was positive; an
    instrument already trending down is unmoved by it, so a one-sided shock silently passes
    on exactly the names it should catch. SHOCKS is two-sided for that reason.

    `position_fn` must rebuild EVERYTHING it needs from `frame`. Closing over a signal
    computed outside makes this return True for any input, which is the other silent way to
    write a look-ahead test that never fires.
    """
    if not 0 < k < len(frame):
        raise ValueError("k must be inside the frame")
    shocks: tuple = ()
    if shock is not None:
        shocks = (float(shock),) if np.isscalar(shock) else tuple(float(s) for s in shock)
        if any(s <= -1.0 for s in shocks):
            raise ValueError("every shock must be > -1 so prices stay positive")
    a = position_fn(frame)
    num = frame.select_dtypes("number").columns
    for perturb in (shocks or (None,)):
        f2 = frame.copy()
        if perturb is None:
            f2.loc[f2.index[k:], num] = f2.loc[f2.index[k:], num] * 2.0
        else:
            f2.loc[f2.index[k:], num] = perturb
        b = position_fn(f2)
        d = (a.iloc[: k + 1].fillna(0.0) - b.iloc[: k + 1].fillna(0.0)).abs()
        if bool((d > tol).to_numpy().any()):
            return False
    return True


def signal_of(returns, kind: str):
    """The three signals as pure functions of a return frame (prices rebuilt inside)."""
    if kind == "tsmom":
        return tsmom_signal(returns)
    if kind == "ma":
        return ma_crossover_signal(close_from_returns(returns))
    if kind == "donchian":
        return donchian_signal(close_from_returns(returns))
    raise ValueError(f"unknown signal {kind!r}")


def position_of(returns, kind: str, signal_lag: int = 1, target: float = VOL_TARGET):
    """The applied position, rebuilt end to end from `returns` - the form the causality
    harness needs."""
    return signal_of(returns, kind).shift(signal_lag) * target / ex_ante_vol(returns, lag=1)


# ---------------------------------------------------------------------------------- demo
if __name__ == "__main__":
    panel = simulate_panel()
    ret, close = panel["returns"], panel["close"]
    n_days, n_assets = ret.shape
    print("=" * 78)
    print(f"TREND-FOLLOWING MODELS  --  {n_assets} synthetic futures, {n_days} days, seed {SEED}")
    print("=" * 78)

    # ---- 1. the MOP volatility estimator, parameter by parameter
    delta = COM_DAYS / (1.0 + COM_DAYS)
    w = (1.0 - delta) * delta ** np.arange(20_000)
    col = ret.columns[0]
    pandas_var = float(ex_ante_vol(ret, lag=1)[col].iloc[-1] ** 2)
    literal_var = eq1_variance(ret[col].to_numpy())
    print("\n1. MOP (2012) eq. (1), the ex-ante volatility")
    print(f"   d = com / (1 + com) = {delta:.6f}   centre of mass sum_i (1-d) d^i i "
          f"= {ewma_center_of_mass(delta):.4f} days")
    print(f"   weights sum to {w.sum():.6f}; variance scaled by {PERIODS_PER_YEAR}; eq. (1) "
          f"sums over r_(t-1-i), so sigma_t excludes r_t")
    print(f"   pandas ewm(com=60).var(bias=True) vs eq. (1) written out literally: "
          f"{pandas_var:.12e} vs {literal_var:.12e}")
    print(f"   max relative difference {abs(pandas_var - literal_var) / literal_var:.3e} "
          f"-- the pandas one-liner IS the paper's estimator")

    # ---- 2. TSMOM at the paper's own frequency: 12-month lookback, 1-month hold
    print("\n2. TSMOM, MOP eq. (5): sign(r_(t-12,t)) * 40% / sigma_t, MONTHLY rebalance")
    m_base, _ = monthly_tsmom(ret)
    m_skip, _ = monthly_tsmom(ret, skip_m=1)
    d_base = diversified(m_base)
    d_skip = diversified(m_skip)
    print(f"   {'variant':<44}{'ann ret':>9}{'ann vol':>9}{'Sharpe':>8}{'max DD':>9}")
    for name, s in (("12 months, nothing skipped (eq. 5)", summary(d_base, MONTHS_PER_YEAR)),
                    ("12-1: most recent month skipped (fn. 10)",
                     summary(d_skip, MONTHS_PER_YEAR))):
        print(f"   {name:<44}{s['ann_return']:>9.1%}{s['ann_vol']:>9.1%}{s['sharpe']:>8.2f}"
              f"{s['max_dd']:>9.1%}")
    print(f"   months: {int(d_base.notna().sum())}; the skip is footnote 10's CROSS-sectional "
          f"convention,")
    print(f"   not part of eq. (5) - MOP: 'our results do not depend on whether the most")
    print(f"   recent month is excluded or not'")

    # ---- 3. daily rebalancing, and what the volatility target does
    strat, pos = tsmom_backtest(ret)
    base = summary(diversified(strat))
    per_asset = [sharpe(strat[c]) for c in strat.columns]
    unsized_strat, _ = tsmom_backtest(ret, size=False)
    unsized = summary(diversified(unsized_strat))
    print("\n3. Same signal rebalanced DAILY, and what the 40% target buys")
    print(f"   per-asset Sharpe: {min(per_asset):.2f} .. {max(per_asset):.2f} "
          f"(assets run from 10% to 40% annual vol)")
    print(f"   {'variant':<44}{'ann ret':>9}{'ann vol':>9}{'Sharpe':>8}{'max DD':>9}")
    for name, s in (("daily rebalance, 40% vol target", base),
                    ("daily rebalance, sign only, no target", unsized)):
        print(f"   {name:<44}{s['ann_return']:>9.1%}{s['ann_vol']:>9.1%}{s['sharpe']:>8.2f}"
              f"{s['max_dd']:>9.1%}")
    hv = strat.columns[-1]
    share_sized = float(strat[hv].var() / strat.var().sum())
    share_unsized = float(unsized_strat[hv].var() / unsized_strat.var().sum())
    print(f"   variance share of the 40%-vol asset: sized {share_sized:.0%}, "
          f"sign-only {share_unsized:.0%} (1/{n_assets} = {1 / n_assets:.0%} is even)")
    print(f"   MOP on the constant: 'The choice of 40% is inconsequential' - it sets the "
          f"scale, not the Sharpe")

    # ---- 4. trap 1: the sigma that contains the return it sizes
    print("\n4. TRAP 1 - sizing r_t with sigma_t (contains r_t) instead of sigma_(t-1)")
    print(f"   {'volatility estimator':<26}{'Sharpe s_(t-1)':>16}{'Sharpe s_t':>12}"
          f"{'gain':>7}{'x2 finds leak?':>16}")
    k = n_days // 2
    rows = []
    for name, fn in (("EWMA com=60 (MOP)", lambda r, lag: ex_ante_vol(r, com=60.0, lag=lag)),
                     ("EWMA com=20", lambda r, lag: ex_ante_vol(r, com=20.0, lag=lag)),
                     ("EWMA com=5", lambda r, lag: ex_ante_vol(r, com=5.0, lag=lag)),
                     ("rolling 60-day", lambda r, lag: rolling_vol(r, 60, lag=lag)),
                     ("rolling 20-day", lambda r, lag: rolling_vol(r, 20, lag=lag)),
                     ("rolling 5-day", lambda r, lag: rolling_vol(r, 5, lag=lag))):
        s1 = sharpe(diversified(tsmom_backtest(ret, vol_fn=fn, vol_lag=1)[0]))
        s0 = sharpe(diversified(tsmom_backtest(ret, vol_fn=fn, vol_lag=0)[0]))
        found = not position_is_causal(
            lambda f, fn=fn: tsmom_backtest(f, vol_fn=fn, vol_lag=0)[1], ret, k)
        rows.append((name, s1, s0, s0 - s1, found))
        print(f"   {name:<26}{s1:>16.2f}{s0:>12.2f}{s0 - s1:>+7.2f}{str(found):>16}")
    print(f"   The look-ahead is worth {min(r[3] for r in rows):+.2f} to "
          f"{max(r[3] for r in rows):+.2f} Sharpe here, and it grows as the")
    print(f"   window shrinks: the shorter the estimator, the more of r_t sits in its own "
          f"denominator.")
    print(f"   A one-line .shift(1) is the whole fix, and nothing warns you it is missing.")

    # ---- 5. trap 2: alignment and fill
    print("\n5. TRAP 2 - which return the position is paid")
    print("   no shift   : position from close_t is paid close_(t-1) -> close_t")
    print("   same close : position from close_(t-1) filled AT close_(t-1), paid to close_t")
    print("   next open  : position from close_(t-1) filled at open_t, paid open_t -> open_(t+1)")
    print(f"   {'model':<18}{'no shift':>10}{'same close':>12}{'next open':>11}"
          f"{'x2 finds leak?':>16}{'shock finds it?':>17}")
    oo = panel["open_to_open"]
    for label, kind in (("TSMOM 12m", "tsmom"), ("MA 50/200", "ma"),
                        ("Donchian 20/10", "donchian")):
        p0 = position_of(ret, kind, signal_lag=0)
        p1 = position_of(ret, kind, signal_lag=1)
        f0 = (lambda f, kd=kind: position_of(f, kd, signal_lag=0))
        f1 = (lambda f, kd=kind: position_of(f, kd, signal_lag=1))
        scale_finds = not position_is_causal(f0, ret, k)
        shock_finds = not position_is_causal(f0, ret, k, shock=SHOCKS)
        assert position_is_causal(f1, ret, k, shock=SHOCKS), "the shifted position must be causal"
        print(f"   {label:<18}{sharpe(diversified(p0 * ret)):>10.2f}"
              f"{sharpe(diversified(p1 * ret)):>12.2f}"
              f"{sharpe(diversified(p1 * oo)):>11.2f}{str(scale_finds):>16}"
              f"{str(shock_finds):>17}")
    print("   The no-shift column is a return the strategy never had. The same-close column")
    print("   is a price it could not get: the close is not known until the bar is over.")
    print("   TRAP 3, in the test itself - assert_causal's x2 perturbation finds the sizing")
    print("   leak of section 4 but NOT this one, because np.sign() is flat under a small")
    print("   perturbation. Shock the tail instead (a -90% and a +900% day) and every model fires.")
    print("   The correctly shifted position stays causal under both (asserted above).")

    print("\nRule: the position that earns r_t must be known at t-1 - signal from close t-1,"
          " sigma from returns through t-1, fill at the next open.")
