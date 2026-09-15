#!/usr/bin/env python3
"""What is different about the crypto VENUE: 24/7, no tape, and a counterparty you cannot net out.

Five demonstrations, each printing the numbers quoted in ../SKILL.md:

  1. annualisation - the factor is ROWS PER YEAR; the same series at 365 and at 252, and the
                     mistake in the other direction. Delegates to fin_skills.api.conventions
                     when the package is importable, rather than re-deriving the convention
  2. windows       - a rolling window keyed to business days is a DIFFERENT window here; a
                     252-row and a 365-row lookback on the same series disagree about the sign
  3. no session    - there is no universal session close; price jumps and outages remain possible, so a business-hours desk is simply absent
                     for most of the week; how much of the stop-breach risk lands there
  4. no tape       - the same asset across seeded venue feeds, and the same daily bar sampled
                     at different UTC hours: two different "the price"s
  5. counterparty  - outages you cannot trade through, socialised losses that take a WINNING
                     position, and a stablecoin depeg that moves a quote with no asset move

Seeded synthetic series; the venue conventions are cited with their date. No network at run
time, numpy/pandas only, fixed seed, no file writes, ASCII output.

Run:  python market_structure.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20260910
W = 96

CRYPTO_DAYS = 365
EQUITY_DAYS = 252
HOURS_PER_YEAR = 365 * 24
FUNDING_8H_PER_YEAR = 365 * 3

# ----------------------------------------------------------------------------------------------
# Dated conventions (see ../SKILL.md for what was verified where)
# ----------------------------------------------------------------------------------------------
# OKX docs, read 2026-09-08 by ../../crypto-data-and-execution/scripts/perp_mechanics.py: the
# `1D` candle bar opens at 00:00 UTC+8, `1Dutc` at 00:00 UTC; Deribit's 1D chart bars open at
# 08:00 UTC, its settlement hour. Three venues, three different "daily closes" for one date.
DAILY_CLOSE_HOURS_UTC = {"00:00 UTC": 0,
                         "16:00 UTC (OKX 1D historical convention)": 16,
                         "08:00 UTC (Deribit historical convention)": 8}
# circle.com/pressroom/3-3-billion-of-usdc-reserve-risk-removed-dollar-de-peg-closes, read
# 2026-09-10: "$3.3B USDC reserve deposit held at Silicon Valley Bank, about 8% of the USDC
# total reserve". A stablecoin is a claim on a balance sheet, and the claim can be questioned.
USDC_SVB = dict(exposure_usd=3.3e9, share_of_reserve=0.08, event="2023-03 SVB failure")
# bybit.com/en/help-center/article/Auto-Deleveraging-ADL, rechecked 2026-09-14:
# Current rules include drawdown thresholds and do not require complete fund exhaustion.
# Historical Binance example, read 2026-09-10: when losses from bankrupt
# positions exceed the insurance fund, "the matching engine will automatically liquidate the
# Bankrupt Positions and some opposing non-bankrupt trader's' positions", i.e. Auto-Deleveraging.
ADL = dict(trigger="venue insurance-risk threshold", takes="selected opposing positions",
           name="auto-deleveraging")


def api_conventions():
    """fin_skills.api.conventions if the package is importable, else None (standalone run)."""
    try:
        from fin_skills.api import conventions
        return conventions
    except ModuleNotFoundError as exc:
        if exc.name == "fin_skills":
            return None
        raise


# ----------------------------------------------------------------------------------------------
# 1. Annualisation
# ----------------------------------------------------------------------------------------------
def sharpe(returns: pd.Series | np.ndarray, periods_per_year: int) -> float:
    """mean / sd(ddof=1) * sqrt(periods). Delegates to fin_skills.api.conventions when present."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 2 or not np.isfinite(r).all():
        raise ValueError("need at least two finite returns in one dimension")
    if not isinstance(periods_per_year, (int, np.integer)) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")
    if r.std(ddof=1) == 0:
        return float("nan")
    c = api_conventions()
    if c is not None:
        return float(c.annualize_sharpe(r, int(periods_per_year)))
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def annualization_factor(calendar: str) -> int:
    """Rows per year. Delegates to fin_skills.api.conventions when the package is importable."""
    c = api_conventions()
    if c is not None:
        return int(c.annualization_factor(calendar))
    table = {"equity": EQUITY_DAYS, "crypto": CRYPTO_DAYS, "hourly": HOURS_PER_YEAR,
             "funding_8h": FUNDING_8H_PER_YEAR}
    if calendar not in table:
        raise ValueError(f"calendar must be one of {sorted(table)}, got {calendar!r}")
    return table[calendar]


def calendar_series(years: int = 4, annual_vol: float = 0.65, drift: float = 0.25,
                    seed: int = SEED) -> pd.Series:
    """A 24/7 daily return series - one row per CALENDAR day, weekends included."""
    if years < 1:
        raise ValueError("years must be positive")
    n = years * CRYPTO_DAYS
    rng = np.random.default_rng(seed)
    dt = 1.0 / CRYPTO_DAYS
    r = rng.normal(drift * dt, annual_vol * np.sqrt(dt), n)
    return pd.Series(r, index=pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC"))


def annualisation_table(r: pd.Series) -> dict[str, float]:
    """The same series both ways, and the weekday-only mistake in the other direction."""
    weekday = r[r.index.dayofweek < 5]
    padded = r.where(r.index.dayofweek < 5, 0.0)             # the same P&L, weekend rows zeroed
    right, wrong = annualization_factor("crypto"), annualization_factor("equity")
    return {
        "rows": int(r.size), "weekday_rows": int(weekday.size),
        "sharpe_365": sharpe(r, right), "sharpe_252": sharpe(r, wrong),
        "vol_365": float(r.std(ddof=1) * np.sqrt(right)),
        "vol_252": float(r.std(ddof=1) * np.sqrt(wrong)),
        "ratio": float(np.sqrt(wrong / right)),
        "weekday_sharpe_365": sharpe(weekday, right),        # the mistake in the other direction
        "weekday_sharpe_252": sharpe(weekday, wrong),
        "padded_sharpe_365": sharpe(padded, right),
        "weekend_share_of_rows": float((r.index.dayofweek >= 5).mean()),
    }


# ----------------------------------------------------------------------------------------------
# 2. Rolling windows
# ----------------------------------------------------------------------------------------------
def window_disagreement(r: pd.Series, short_rows: int = EQUITY_DAYS,
                        long_rows: int = CRYPTO_DAYS) -> dict[str, float]:
    """"One year" of momentum computed with 252 rows and with 365 rows, on the same series."""
    if short_rows < 2 or long_rows <= short_rows:
        raise ValueError("need 2 <= short_rows < long_rows")
    px = (1.0 + r).cumprod()
    a = px / px.shift(short_rows) - 1.0
    b = px / px.shift(long_rows) - 1.0
    both = pd.concat([a, b], axis=1).dropna()
    sa, sb = both.iloc[:, 0], both.iloc[:, 1]
    return {"short_rows": short_rows, "long_rows": long_rows, "observations": int(len(both)),
            "sign_disagreement": float((np.sign(sa) != np.sign(sb)).mean()),
            "correlation": float(np.corrcoef(sa, sb)[0, 1]),
            "mean_abs_gap": float((sa - sb).abs().mean()),
            "calendar_days_short": short_rows, "calendar_months_short": short_rows / 30.44}


# ----------------------------------------------------------------------------------------------
# 3. No session
# ----------------------------------------------------------------------------------------------
def unwatched_breaches(stop_move: float = 0.05, days: int = 365, annual_vol: float = 0.65,
                       open_utc: int = 13, close_utc: int = 21, n_paths: int = 2000,
                       seed: int = SEED) -> dict[str, float]:
    """When a stop level is first breached, relative to one desk's business hours.

    The desk watches `open_utc`..`close_utc` on weekdays. The market does not stop. This counts
    where the FIRST breach of a -stop_move level lands, over seeded hourly paths.
    """
    if not 0.0 < stop_move < 1.0 or not 0 <= open_utc < close_utc <= 24:
        raise ValueError("stop_move must be a fraction and the session hours ordered")
    rng = np.random.default_rng(seed + 2)
    n = days * 24
    dt = 1.0 / HOURS_PER_YEAR
    px = np.exp(np.cumsum(rng.normal(-0.5 * annual_vol ** 2 * dt,
                                     annual_vol * np.sqrt(dt), (n_paths, n)), axis=1))
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    watched = np.asarray((idx.dayofweek < 5) & (idx.hour >= open_utc) & (idx.hour < close_utc))
    breach = px <= (1.0 - stop_move)
    any_breach = breach.any(axis=1)
    first = np.argmax(breach, axis=1)[any_breach]
    in_hours = watched[first]
    return {"paths": int(n_paths), "breached": int(any_breach.sum()),
            "watched_hours_share": float(watched.mean()),
            "breaches_in_hours": float(in_hours.mean()) if in_hours.size else float("nan"),
            "breaches_unwatched": float(1.0 - in_hours.mean()) if in_hours.size else float("nan"),
            "stop_move": stop_move, "session": f"{open_utc:02d}-{close_utc:02d} UTC Mon-Fri"}


# ----------------------------------------------------------------------------------------------
# 4. No consolidated tape
# ----------------------------------------------------------------------------------------------
def venue_feeds(n_days: int = 730, n_venues: int = 5, annual_vol: float = 0.65,
                premium_sd: float = 0.0009, phi: float = 0.96, noise_bps: float = 4.0,
                seed: int = SEED) -> pd.DataFrame:
    """One latent asset, `n_venues` books: a persistent per-venue premium plus quote noise."""
    if n_venues < 2 or n_days < 2:
        raise ValueError("need at least two venues and two days")
    rng = np.random.default_rng(seed + 4)
    dt = 1.0 / CRYPTO_DAYS
    latent = 30_000.0 * np.exp(np.cumsum(rng.normal(-0.5 * annual_vol ** 2 * dt,
                                                    annual_vol * np.sqrt(dt), n_days)))
    eps = rng.normal(0.0, premium_sd * np.sqrt(1 - phi ** 2), (n_venues, n_days))
    prem = np.empty_like(eps)
    prem[:, 0] = eps[:, 0]
    for t in range(1, n_days):
        prem[:, t] = phi * prem[:, t - 1] + eps[:, t]
    noise = rng.normal(0.0, noise_bps * 1e-4, (n_venues, n_days))
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    return pd.DataFrame((latent * (1.0 + prem + noise)).T, index=idx,
                        columns=[f"venue{i + 1}" for i in range(n_venues)])


def cross_venue_dispersion(feeds: pd.DataFrame) -> dict[str, float]:
    if feeds.empty or feeds.shape[1] < 2 or not np.isfinite(feeds.to_numpy()).all():
        raise ValueError("need finite prices from at least two venues")
    if (feeds <= 0).any().any():
        raise ValueError("prices must be positive")
    rng_bps = (feeds.max(axis=1) / feeds.min(axis=1) - 1.0) * 1e4
    best = feeds.idxmax(axis=1)
    return {"venues": int(feeds.shape[1]), "days": int(feeds.shape[0]),
            "median_range_bps": float(rng_bps.median()),
            "p99_range_bps": float(rng_bps.quantile(0.99)),
            "max_range_bps": float(rng_bps.max()),
            "highest_venue_changes": float((best.iloc[1:] != best.shift().iloc[1:]).mean()),
            "distinct_leaders": int(best.nunique())}


def same_rule_five_feeds(feeds: pd.DataFrame, lookback: int = 30) -> dict[str, float]:
    """One momentum rule, five venue feeds. The rule never changes; the answer does."""
    if lookback < 2 or lookback >= len(feeds):
        raise ValueError("lookback must be inside the sample")
    out = {}
    for col in feeds.columns:
        px = feeds[col]
        signal = (px / px.shift(lookback) - 1.0 > 0).astype(float).shift(1).fillna(0.0)
        out[col] = float((1.0 + signal * px.pct_change().fillna(0.0)).prod() - 1.0)
    vals = np.array(list(out.values()))
    return {"per_venue": out, "best": float(vals.max()), "worst": float(vals.min()),
            "spread": float(vals.max() - vals.min()), "mean": float(vals.mean()),
            "lookback": lookback}


def close_hour_effect(n_days: int = 730, annual_vol: float = 0.65,
                      hours: dict[str, int] | None = None,
                      seed: int = SEED) -> dict[str, float]:
    """One 24/7 hourly path sampled as a DAILY close at three different UTC hours."""
    hours = DAILY_CLOSE_HOURS_UTC if hours is None else hours
    rng = np.random.default_rng(seed + 6)
    n = n_days * 24
    dt = 1.0 / HOURS_PER_YEAR
    px = pd.Series(np.exp(np.cumsum(rng.normal(-0.5 * annual_vol ** 2 * dt,
                                               annual_vol * np.sqrt(dt), n))),
                   index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
    out = {}
    for label, h in hours.items():
        daily = px[px.index.hour == h]
        r = daily.pct_change().dropna()
        out[label] = {"rows": int(r.size), "vol": float(r.std(ddof=1) * np.sqrt(CRYPTO_DAYS)),
                      "sharpe": sharpe(r, CRYPTO_DAYS),
                      "total": float(daily.iloc[-1] / daily.iloc[0] - 1.0),
                      "maxdd": float((daily / daily.cummax() - 1.0).min())}
    sharpes = [v["sharpe"] for v in out.values()]
    return {"per_hour": out, "sharpe_spread": float(max(sharpes) - min(sharpes)),
            "sharpe_min": float(min(sharpes)), "sharpe_max": float(max(sharpes))}


# ----------------------------------------------------------------------------------------------
# 5. The venue is a counterparty
# ----------------------------------------------------------------------------------------------
def outage_cost(n_days: int = 365, annual_vol: float = 0.65, outages_per_year: int = 6,
                mean_hours: float = 4.0, n_paths: int = 3000,
                seed: int = SEED) -> dict[str, float]:
    """A stop you cannot act on because the venue is down. What the delay costs, per event."""
    if outages_per_year < 1 or mean_hours <= 0:
        raise ValueError("need at least one outage and a positive duration")
    rng = np.random.default_rng(seed + 8)
    dt = 1.0 / HOURS_PER_YEAR
    dur = np.maximum(1, rng.poisson(mean_hours, n_paths))
    slip = np.array([np.exp(rng.normal(-0.5 * annual_vol ** 2 * dt * d,
                                       annual_vol * np.sqrt(dt * d))) - 1.0 for d in dur])
    adverse = slip[slip < 0]
    return {"events": int(n_paths), "mean_hours": float(dur.mean()),
            "outages_per_year": outages_per_year,
            "median_move": float(np.median(slip)), "p05_move": float(np.quantile(slip, 0.05)),
            "worst_move": float(slip.min()), "mean_adverse": float(adverse.mean()),
            "expected_annual_drag": float(np.minimum(slip, 0).mean() * outages_per_year),
            "unconditional_annual_change": float(slip.mean() * outages_per_year)}


def adl_haircut(entry: float, mark: float, adl_price: float, side: str = "short") -> dict[str, float]:
    """A WINNING position closed at the bankrupt counterparty's price instead of the market's."""
    if entry <= 0 or mark <= 0 or adl_price <= 0:
        raise ValueError("prices must be positive")
    sign = -1.0 if side == "short" else 1.0
    if side not in ("short", "long"):
        raise ValueError("side must be 'short' or 'long'")
    full = sign * (mark - entry) / entry
    got = sign * (adl_price - entry) / entry
    return {"side": side, "pnl_at_mark": full, "pnl_after_adl": got,
            "given_up": full - got, "share_given_up": (full - got) / full if full else 0.0}


def depeg_shock(n_days: int = 180, depeg_start: int = 90, depeg_days: int = 6,
                depth: float = 0.087, annual_vol: float = 0.65,
                seed: int = SEED) -> dict[str, float]:
    """A stablecoin below par: the STABLE-quoted price of an asset rises with no asset move."""
    if (not 0 < depeg_start < n_days or depeg_days < 3
            or depeg_start + depeg_days >= n_days or not 0.0 < depth < 1.0):
        raise ValueError("the depeg must fit inside the sample and be a fraction")
    rng = np.random.default_rng(seed + 10)
    dt = 1.0 / CRYPTO_DAYS
    usd = 30_000.0 * np.exp(np.cumsum(rng.normal(-0.5 * annual_vol ** 2 * dt,
                                                 annual_vol * np.sqrt(dt), n_days)))
    peg = np.ones(n_days)
    shape = np.sin(np.linspace(0.0, np.pi, depeg_days))
    peg[depeg_start:depeg_start + depeg_days] -= depth * shape
    quoted = usd / peg                                       # price in units of the stablecoin
    r_usd = pd.Series(usd).pct_change().dropna().to_numpy()
    r_quo = pd.Series(quoted).pct_change().dropna().to_numpy()
    lo, hi = depeg_start - 1, depeg_start + depeg_days
    return {"worst_peg": float(peg.min()), "depeg_days": depeg_days,
            "phantom_return_in_window": float(quoted[hi] / quoted[lo] - 1.0
                                              - (usd[hi] / usd[lo] - 1.0)),
            "peak_phantom": float((quoted / usd).max() - 1.0),
            "tracking_error_bps": float((r_quo - r_usd).std(ddof=1) * np.sqrt(CRYPTO_DAYS) * 1e4),
            "correlation": (float(np.corrcoef(r_usd, r_quo)[0, 1])
                            if r_usd.std() > 0 and r_quo.std() > 0 else float("nan"))}


THE_RULE = "RULE: annualize for observed rows; identify venue, bar boundary, quote currency and event risks."


def main() -> None:                                          # noqa: C901 - a printed report
    print("=" * W)
    print("CRYPTO MARKET STRUCTURE - 24/7, no tape, and a venue that is also a counterparty")
    print("=" * W)
    c = api_conventions()
    print("   conventions: " + ("fin_skills.api.conventions (imported)" if c is not None
                                else "fin_skills not importable - same one-liner inline"))

    # ---- 1. annualisation
    print("\n1. THE FACTOR IS ROWS PER YEAR, AND THE ERROR RUNS BOTH WAYS")
    r = calendar_series()
    a = annualisation_table(r)
    print(f"   {a['rows']} calendar-day rows ({a['weekend_share_of_rows']:.1%} of them weekends, "
          f"which a business calendar drops)")
    print(f"   {'statistic':<22}{'sqrt(365) right':>18}{'sqrt(252) wrong':>18}{'error':>12}")
    print(f"   {'Sharpe':<22}{a['sharpe_365']:>18.4f}{a['sharpe_252']:>18.4f}"
          f"{a['sharpe_252'] / a['sharpe_365'] - 1:>12.1%}")
    print(f"   {'annual vol':<22}{a['vol_365']:>18.2%}{a['vol_252']:>18.2%}"
          f"{a['vol_252'] / a['vol_365'] - 1:>12.1%}")
    print(f"   sqrt(252/365) = {a['ratio']:.4f}, so both are understated by "
          f"{1 - a['ratio']:.1%} -- exactly, every time")
    print(f"   Sensitivity example: keep only the {a['weekday_rows']} WEEKDAY rows and annualise")
    print(f"   at 365 and the Sharpe reads {a['weekday_sharpe_365']:.4f} against "
          f"{a['weekday_sharpe_252']:.4f} at 252 -- a scaling difference of "
          f"{a['weekday_sharpe_365'] / a['weekday_sharpe_252'] - 1:.1%}.")
    print("   This subset drops weekends only. 252 is a comparison, not its measured rows/year.")
    print(f"   Pad the same P&L with weekend zeros and 365 gives {a['padded_sharpe_365']:.4f}.")
    print(f"   ! Rows per year: crypto {annualization_factor('crypto')}, equity "
          f"{annualization_factor('equity')}, hourly {annualization_factor('hourly')}, "
          f"8h funding {annualization_factor('funding_8h')}.")

    # ---- 2. windows
    print("\n2. A WINDOW KEYED TO BUSINESS DAYS IS A DIFFERENT WINDOW HERE")
    w = window_disagreement(r)
    print(f"   '12-month momentum' as {w['short_rows']} rows is "
          f"{w['calendar_months_short']:.1f} calendar months in a 24/7 series, not 12.")
    print(f"   Over {w['observations']} observations the {w['short_rows']}-row and "
          f"{w['long_rows']}-row versions correlate {w['correlation']:.4f},")
    print(f"   the mean absolute gap is {w['mean_abs_gap']:.2%} of price, and they disagree "
          f"about the SIGN")
    print(f"   {w['sign_disagreement']:.1%} of the time.")
    print("   ! The correlation and sign-disagreement rate above")
    print("     is a result for this synthetic path; differently sized windows can disagree and")
    print("     should be named by their actual duration.")

    # ---- 3. no session
    print("\n3. THERE IS NO CLOSE, SO A BUSINESS-HOURS DESK IS SIMPLY ABSENT")
    u = unwatched_breaches()
    print(f"   A {u['stop_move']:.0%} stop, {u['paths']} seeded hourly paths over a year, a desk "
          f"watching {u['session']}")
    print(f"   ({u['watched_hours_share']:.1%} of the week's hours):")
    print(f"   {u['breached']} of {u['paths']} paths breached the level; of those, "
          f"{u['breaches_in_hours']:.1%} first breached")
    print(f"   during the session and {u['breaches_unwatched']:.1%} did NOT.")
    print("   ! The off-desk share is measured only in the assumed homogeneous-volatility model:")
    print("     real market activity varies by hour. Continuous trading does not eliminate")
    print("     price jumps, local outages, or the need to monitor orders outside office hours.")

    # ---- 4. no tape
    print("\n4. THERE IS NO CONSOLIDATED TAPE - 'the price' is a venue and a UTC hour")
    feeds = venue_feeds()
    d = cross_venue_dispersion(feeds)
    print(f"   {d['venues']} seeded venue feeds of ONE asset over {d['days']} days:")
    print(f"   cross-venue range median {d['median_range_bps']:.1f} bp, 99th pct "
          f"{d['p99_range_bps']:.0f} bp, max {d['max_range_bps']:.0f} bp;")
    print(f"   the highest-priced venue changes on {d['highest_venue_changes']:.0%} of days and "
          f"{d['distinct_leaders']} different venues lead at some point")
    s = same_rule_five_feeds(feeds)
    print(f"   The SAME {s['lookback']}-day momentum rule on each of those feeds:")
    print("   " + "  ".join(f"{k}: {v:.1%}" for k, v in s["per_venue"].items()))
    print(f"   best {s['best']:.1%}, worst {s['worst']:.1%}, spread "
          f"{s['spread']:.1%} of return on one asset, one rule, one period")
    ch = close_hour_effect()
    print("   And one 24/7 path sampled as a DAILY close at three real venue hours:")
    print(f"   {'sampling hour':<44}{'rows':>6}{'vol':>9}{'Sharpe':>9}{'total':>9}{'maxDD':>9}")
    for label, v in ch["per_hour"].items():
        print(f"   {label:<44}{v['rows']:>6}{v['vol']:>9.2%}{v['sharpe']:>9.4f}"
              f"{v['total']:>9.1%}{v['maxdd']:>9.1%}")
    print(f"   Same path sampled at different endpoints: Sharpe {ch['sharpe_min']:.4f} to "
          f"{ch['sharpe_max']:.4f}, a spread of {ch['sharpe_spread']:.4f}")
    print("   ! The hour is a CHOICE and it is not documented in your dataframe. Store the venue")
    print("     and the bar's opening hour beside every series, or the backtest is unrepeatable.")

    # ---- 5. counterparty
    print("\n5. THE VENUE IS A COUNTERPARTY, NOT A UTILITY")
    o = outage_cost()
    print(f"   Outage: a stop you cannot act on for a mean of {o['mean_hours']:.1f} hours. "
          f"Over {o['events']} seeded events the")
    print(f"   median move is {o['median_move']:+.2%}, the 5th percentile "
          f"{o['p05_move']:.2%} and the worst {o['worst_move']:.2%};")
    print(f"   Conditional adverse mean {o['mean_adverse']:.2%}; "
          f"{o['outages_per_year']} assumed events/year, negatives alone:")
    print(f"   {o['expected_annual_drag']:.2%}; this excludes favorable moves, so it is NOT expected drag.")
    h = adl_haircut(30_000.0, 21_000.0, 27_000.0)
    print(f"   Forced-close example ({ADL['name']}): closes "
          f"{ADL['takes']}.")
    print(f"   A short from 30,000 marked at 21,000 is up {h['pnl_at_mark']:.1%}; closed at "
          f"27,000 it keeps {h['pnl_after_adl']:.1%} --")
    print(f"   {h['share_given_up']:.0%} of a WINNING position handed back, and no order of "
          f"yours caused it.")
    dp = depeg_shock()
    print(f"   Stablecoin depeg: the quote currency falls to {dp['worst_peg']:.3f} for "
          f"{dp['depeg_days']} days. The STABLE-quoted price")
    print(f"   has a {dp['peak_phantom']:.1%} quote-conversion premium at the peak;"
          f" over the window")
    print(f"   the phantom return is {dp['phantom_return_in_window']:+.2%} and the annualised "
          f"tracking error {dp['tracking_error_bps']:.0f} bp")
    print(f"   at a {dp['correlation']:.4f} daily correlation.")
    print(f"   ! Source-verified: ${USDC_SVB['exposure_usd'] / 1e9:.1f}bn of USDC reserve sat at "
          f"one bank, {USDC_SVB['share_of_reserve']:.0%} of the reserve")
    print(f"     ({USDC_SVB['event']}). Preserve the USD conversion of stablecoin quotes")
    print("     and the collateral backing positions margined in them.")

    print("\n" + "=" * W)
    print(THE_RULE)
    print("=" * W)


if __name__ == "__main__":
    main()
