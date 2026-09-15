#!/usr/bin/env python3
"""Crypto token events - the corporate actions nobody adjusts for, and what they do to a series.

Five demonstrations, each printing the numbers quoted in ../SKILL.md:

  1. rebase          - a rebase changes BALANCE; price also responds to demand. The price-only return is
                       missing the balance factor, and the multiplicative identity is exact
  2. redenomination  - a 1-old-for-10-new swap under the SAME ticker prints a -90% day that
                       never happened; what the adjustment factor fixes and what it does not
  3. survivorship    - a top-N-by-volume pair list reconstructed as of a past date, how many of
                       those pairs still trade, and how much a survivor-only cohort overstates
  4. wrapped         - a wrapped or bridged representation is a claim, not the asset: the basis
                       is near zero until it is not, and substituting one series for the other
                       has a measurable tracking error
  5. forks           - a chain split hands you a second asset at a block height, and the
                       arithmetic of "the price series" depends on which chain kept the ticker

Everything is seeded synthetic arithmetic plus constants read at the venues' own announcement
and documentation pages on 2026-09-10 (URL next to each constant). No network at run time,
numpy/pandas only, fixed seed, no file writes, ASCII output.

Run:  python token_events.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20260910
W = 96

# ----------------------------------------------------------------------------------------------
# Source-verified constants - read 2026-09-10 at the pages named, no key, no API
# ----------------------------------------------------------------------------------------------

# binance.com/en/support/announcement/binance-will-support-the-stratis-strax-token-swap-and-
#   redenomination-plan-2e6eb8e644664a6ca7b6659a226c41b0
# "All old STRAX tokens will be swapped to new STRAX tokens at a ratio of 1 old STRAX =
#  10 new STRAX"; pairs delisted and pending orders cancelled 2024-03-20 03:00 UTC.
STRAX_SWAP = dict(ticker_before="STRAX", ticker_after="STRAX", new_per_old=10.0,
                  halt_utc="2024-03-20T03:00Z", pairs=("STRAX/BTC", "STRAX/USDT", "STRAX/TRY"))

# binance.com/en/support/announcement/binance-will-support-the-galxe-gal-token-swap-
#   redenomination-and-rebranding-to-gravity-g-27449abd481b473493b9c058350b4e2e
# "All GAL tokens will be swapped to G at a ratio of 1 GAL = 60 G"; GAL pairs delisted and
# pending orders cancelled 2024-07-15 03:00 UTC, G/USDT and G/TRY opened 2024-07-19 08:00 UTC.
GAL_SWAP = dict(ticker_before="GAL", ticker_after="G", new_per_old=60.0,
                halt_utc="2024-07-15T03:00Z", relist_utc="2024-07-19T08:00Z")

# binance.com/en-IN/support/faq/binance-delisting-guidelines-frequently-asked-questions-
#   e5a9718ccb794acda1c48db5c71753e4  - pairs are removed from the exchange, open orders are
# cancelled, withdrawals generally cease two months after the delisting date, and delisted
# tokens may then be converted to stablecoins on the holder's behalf.
BINANCE_DELISTING = dict(orders="cancelled at the halt", withdrawal_grace_months=2,
                         after="balance may be converted to a stablecoin",
                         review="quarterly, first week of the quarter (Monitoring Tag)")

# docs.ampleforth.org/learn/about-the-ampleforth-protocol - supply adjustments ("rebases") are
# applied "by updating a global scalar coefficient of expansion every day at 2AM UTC", the
# adjustment "is applied universally to all addresses", and it is non-dilutive: a holder of Y%
# of the network still holds Y% afterwards.
REBASE_PROTOCOL = dict(cadence_hours=24, utc_hour=2, scope="every address",
                       dilutive=False, ownership_share="unchanged")

# blog.ethereum.org/2016/07/20/hard-fork-completed - the DAO hard fork executed at block
# 1920000 on 2016-07-20, "an irregular state change which transferred ~12 million ETH"; clients
# that rejected it ran with --oppose-dao-fork and were warned about transaction replay.
DAO_FORK = dict(block=1_920_000, date="2016-07-20", eth_moved_millions=12.0,
                opt_out_flag="--oppose-dao-fork", hazard="transaction replay")

# wbtc.network - "Every WBTC is backed 1:1 by Bitcoin in secure custody, fully verifiable
# through the on-chain proof of Reserves"; only identity-verified institutions approved through
# DAO governance can mint and burn.
WRAPPED = dict(ratio=1.0, backing="custodied BTC", mint="approved merchants only",
               verification="on-chain proof of reserves")


# ----------------------------------------------------------------------------------------------
# 1. Rebase: the supply moves, not the price
# ----------------------------------------------------------------------------------------------
def rebase_history(n_days: int = 730, target: float = 1.0, lag: float = 10.0,
                   mcap_drift: float = 0.60, mcap_vol: float = 0.45,
                   mcap0: float = 1.0e9, supply0: float | None = None,
                   seed: int = SEED) -> pd.DataFrame:
    """A rebasing token: network value is a GBM, the protocol moves SUPPLY toward a price target.

    The protocol rule is `supply *= 1 + (price/target - 1) / lag`, applied to every address, so
    a holder's share of the network never changes. `lag` and `target` are protocol parameters,
    not claims about any particular token. price = mcap / supply by construction, and the
    default supply starts the token exactly AT its target so the demo shows the ongoing effect
    rather than an initial convergence.
    """
    if n_days < 2 or lag < 1 or target <= 0 or mcap0 <= 0:
        raise ValueError("need n_days >= 2, lag >= 1 and positive target and market cap")
    supply0 = mcap0 / target if supply0 is None else supply0
    if supply0 <= 0 or mcap_vol < 0:
        raise ValueError("supply must be positive and volatility non-negative")
    rng = np.random.default_rng(seed)
    dt = 1.0 / 365.0
    shocks = rng.normal((mcap_drift - 0.5 * mcap_vol ** 2) * dt, mcap_vol * np.sqrt(dt), n_days)
    mcap = mcap0 * np.exp(np.cumsum(shocks))

    supply = np.empty(n_days)
    price = np.empty(n_days)
    s = supply0
    for t in range(n_days):
        supply[t] = s
        price[t] = mcap[t] / s
        s = s * (1.0 + (price[t] / target - 1.0) / lag)     # tomorrow's supply, set at 02:00 UTC
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    return pd.DataFrame({"mcap": mcap, "supply": supply, "price": price,
                         "supply_index": supply / supply0}, index=idx)


def rebase_returns(df: pd.DataFrame) -> dict[str, float]:
    """Price-only vs balance-adjusted returns, and the exact identity that links them."""
    p, s = df["price"].to_numpy(), df["supply_index"].to_numpy()
    value = p * s                                            # one unit held from day 0
    r_price = p[-1] / p[0] - 1.0
    r_supply = s[-1] / s[0] - 1.0
    r_value = value[-1] / value[0] - 1.0
    dp = pd.Series(p).pct_change().dropna().to_numpy()
    ds = pd.Series(s).pct_change().dropna().to_numpy()
    dv = pd.Series(value).pct_change().dropna().to_numpy()
    n = len(dv)
    return {
        "days": len(p),
        "price_return": r_price,
        "supply_return": r_supply,
        "value_return": r_value,
        # (1+r_value) = (1+r_price)(1+r_supply) EXACTLY, at every horizon
        "identity_residual": abs((1 + r_value) - (1 + r_price) * (1 + r_supply)),
        "daily_identity_max": float(np.abs((1 + dv) - (1 + dp) * (1 + ds)).max()),
        "price_vol": float(dp.std(ddof=1) * np.sqrt(365)),
        "value_vol": float(dv.std(ddof=1) * np.sqrt(365)),
        "price_sharpe": float(dp.mean() / dp.std(ddof=1) * np.sqrt(365)),
        "value_sharpe": float(dv.mean() / dv.std(ddof=1) * np.sqrt(365)),
        "corr": float(np.corrcoef(dp, dv)[0, 1]),
        "n_returns": n,
    }


# ----------------------------------------------------------------------------------------------
# 2. Redenomination: the same ticker, a different unit
# ----------------------------------------------------------------------------------------------
def redenominated_series(n_days: int = 500, event_day: int = 250, new_per_old: float = 10.0,
                         vol: float = 0.80, drift: float = 0.0, p0: float = 4.0,
                         seed: int = SEED) -> pd.DataFrame:
    """A price path with a swap at `event_day`: from then on the ticker quotes the NEW unit.

    `raw` is what an exchange's OHLCV endpoint returns for the ticker across the event;
    `adjusted` multiplies post-event prints by `new_per_old`, expressing the series in old
    units throughout. Back-adjusting the earlier prints instead gives identical returns.
    `balance` is what the holder actually owns.
    """
    if not 0 < event_day < n_days:
        raise ValueError("event_day must be inside the sample")
    if new_per_old <= 0 or p0 <= 0 or vol < 0:
        raise ValueError("ratio and price must be positive and volatility non-negative")
    rng = np.random.default_rng(seed + 1)
    dt = 1.0 / 365.0
    steps = rng.normal((drift - 0.5 * vol ** 2) * dt, vol * np.sqrt(dt), n_days)
    true_px = p0 * np.exp(np.cumsum(steps))                  # price of one OLD unit throughout
    raw = true_px.copy()
    raw[event_day:] = true_px[event_day:] / new_per_old      # the ticker now quotes new units
    balance = np.ones(n_days)
    balance[event_day:] = new_per_old
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    return pd.DataFrame({"true_old_unit": true_px, "raw": raw, "balance": balance,
                         "adjusted": raw * np.where(np.arange(n_days) < event_day, 1.0,
                                                    new_per_old)}, index=idx)


def redenomination_damage(df: pd.DataFrame) -> dict[str, float]:
    """What the unadjusted ticker series claims, against what the holder experienced."""
    raw, adj, bal = df["raw"].to_numpy(), df["adjusted"].to_numpy(), df["balance"].to_numpy()
    r_raw = pd.Series(raw).pct_change().dropna().to_numpy()
    r_adj = pd.Series(adj).pct_change().dropna().to_numpy()
    wealth = raw * bal                                       # what the holder is worth
    r_true = pd.Series(wealth).pct_change().dropna().to_numpy()
    dd = lambda x: float((x / np.maximum.accumulate(x) - 1.0).min())
    return {
        "raw_total": raw[-1] / raw[0] - 1.0,
        "true_total": wealth[-1] / wealth[0] - 1.0,
        "raw_worst_day": float(r_raw.min()),
        "true_worst_day": float(r_true.min()),
        "raw_vol": float(r_raw.std(ddof=1) * np.sqrt(365)),
        "true_vol": float(r_true.std(ddof=1) * np.sqrt(365)),
        "raw_maxdd": dd(raw),
        "true_maxdd": dd(wealth),
        "adjusted_matches_true": float(np.abs(r_adj - r_true).max()),
    }


# ----------------------------------------------------------------------------------------------
# 3. Survivorship: reconstructing a pair list as of a past date
# ----------------------------------------------------------------------------------------------
def venue_history(n_pairs: int = 500, n_days: int = 1095, annual_delist: float = 0.20,
                  annual_floor: float = 0.05, tilt: float = 1.2,
                  new_listings_per_day: float = 0.25,
                  seed: int = SEED) -> dict[str, np.ndarray]:
    """A venue's listings over `n_days`: volume paths, delisting dates, and later listings.

    Two hazards, because tokens die two ways. `annual_delist` is the VOLUME-tilted one the
    MEDIAN pair faces, `h = base * exp(-tilt * z) / median(...)` on standardised log volume, so
    a fading pair's hazard rises as it fades. `annual_floor` is flat across every pair however
    liquid - exploit, fraud, regulatory - and is what stops the top of the book looking immortal.
    The realised overall rate is measured, not assumed; `survivorship()` reports it.
    """
    if not 0.0 <= annual_delist < 1.0 or not 0.0 <= annual_floor < 1.0:
        raise ValueError("annual_delist and annual_floor must be probabilities")
    if n_pairs < 1 or n_days < 2 or new_listings_per_day < 0:
        raise ValueError("need positive pairs, at least two days and non-negative new listings")
    rng = np.random.default_rng(seed + 2)
    total = n_pairs + int(new_listings_per_day * n_days)
    listed_on = np.zeros(total, dtype=int)
    listed_on[n_pairs:] = np.sort(rng.integers(1, n_days, total - n_pairs))

    # log-volume: a pair-specific level plus a random walk with a downward drift for most names
    level = rng.normal(13.0, 1.6, total)
    drift = rng.normal(-0.0012, 0.0025, total)               # most names fade, a few grow
    shocks = rng.normal(0.0, 0.09, (total, n_days))
    logv = level[:, None] + np.cumsum(shocks, axis=1) + drift[:, None] * np.arange(n_days)

    price = np.exp(np.cumsum(rng.normal(-0.0009, 0.055, (total, n_days)), axis=1))
    base = -np.log1p(-annual_delist) / 365.0
    floor = -np.log1p(-annual_floor) / 365.0
    z_all = (logv - logv.mean()) / logv.std(ddof=1)
    mult = np.exp(-tilt * z_all)
    mult /= np.median(mult)                                  # the MEDIAN pair faces `base`
    dead_on = np.full(total, n_days, dtype=int)
    for i in range(total):
        h = floor + base * mult[i]
        u = rng.random(n_days)
        hit = np.nonzero((u < h) & (np.arange(n_days) > listed_on[i] + 30))[0]
        if hit.size:
            dead_on[i] = int(hit[0])
    return {"logv": logv, "price": price, "listed_on": listed_on, "dead_on": dead_on,
            "n_days": n_days}


def _trading(u: dict, day: int) -> np.ndarray:
    return (u["listed_on"] <= day) & (u["dead_on"] > day)


def top_n_as_of(u: dict, day: int, n: int = 50, window: int = 30) -> np.ndarray:
    """Up to n trading pairs, ranked on mean volume since listing within the trailing window."""
    if not 0 <= day < u["n_days"] or n < 1 or window < 1:
        raise ValueError("day must be in the sample, n and window must be positive")
    live = _trading(u, day)
    lo = max(0, day - window + 1)
    eligible = np.flatnonzero(live)
    days = np.arange(lo, day + 1)
    observed = days[None, :] >= u["listed_on"][eligible, None]
    volume = np.exp(u["logv"][eligible, lo:day + 1])
    vol = np.where(observed, volume, 0.0).sum(axis=1) / observed.sum(axis=1)
    return eligible[np.argsort(-vol, kind="stable")[:n]]


def survivorship(u: dict, n: int = 50) -> dict[str, float]:
    """Both directions of the bias, plus what a survivor-only cohort return overstates by."""
    last = u["n_days"] - 1
    then, now = top_n_as_of(u, 0, n), top_n_as_of(u, last, n)
    if len(then) < n or len(now) < n:
        raise ValueError("need at least n trading pairs at each comparison date")
    alive_now = _trading(u, last)

    survived = int(alive_now[then].sum())
    existed_then = int(_trading(u, 0)[now].sum())

    # honest cohort return: a delisted name is marked at its LAST traded price, no haircut.
    end_idx = np.where(u["dead_on"][then] < u["n_days"], u["dead_on"][then] - 1, last)
    p0 = u["price"][then, 0]
    ret_all = u["price"][then, end_idx] / p0 - 1.0
    ret_surv = ret_all[alive_now[then]]
    n_live = int(_trading(u, last).sum())
    n_start = int(_trading(u, 0).sum())
    realised = 1.0 - (1.0 - (u["dead_on"] < u["n_days"]).mean()) ** (365.0 / u["n_days"])
    return {
        "n": n,
        "years": u["n_days"] / 365.0,
        "pairs_at_start": n_start,
        "pairs_at_end": n_live,
        "realised_annual_delist": float(realised),
        "survived": survived,
        "survival_rate": survived / n,
        "existed_then": existed_then,
        "backward_rate": existed_then / n,
        "cohort_mean_return": float(ret_all.mean()),
        "survivor_mean_return": float(ret_surv.mean()) if ret_surv.size else float("nan"),
        "overstatement": float(ret_surv.mean() - ret_all.mean()) if ret_surv.size else float("nan"),
        "cohort_median_return": float(np.median(ret_all)),
    }


# ----------------------------------------------------------------------------------------------
# 4. Wrapped and bridged: a claim, not the asset
# ----------------------------------------------------------------------------------------------
def wrapped_basis(n_days: int = 730, n_stress: int = 3, calm_bps: float = 4.0,
                  stress_bps: float = 900.0, seed: int = SEED) -> pd.DataFrame:
    """A 1:1 wrapper's basis to its underlying: a few bps, until a stress episode says otherwise."""
    rng = np.random.default_rng(seed + 3)
    basis = rng.normal(0.0, calm_bps * 1e-4, n_days)
    for start in rng.choice(np.arange(60, n_days - 30), size=n_stress, replace=False):
        length = int(rng.integers(4, 12))
        depth = -abs(rng.normal(stress_bps, 250.0)) * 1e-4
        shape = np.sin(np.linspace(0.0, np.pi, length))      # widen then heal
        basis[start:start + length] += depth * shape
    under = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.03, n_days)))
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    return pd.DataFrame({"underlying": under, "wrapped": under * (1.0 + basis),
                         "basis": basis}, index=idx)


def wrapped_metrics(df: pd.DataFrame, threshold_bps: float = 50.0) -> dict[str, float]:
    b = df["basis"].to_numpy()
    ru = df["underlying"].pct_change().dropna().to_numpy()
    rw = df["wrapped"].pct_change().dropna().to_numpy()
    return {
        "median_abs_bps": float(np.median(np.abs(b)) * 1e4),
        "p99_abs_bps": float(np.quantile(np.abs(b), 0.99) * 1e4),
        "worst_bps": float(b.min() * 1e4),
        "days_beyond": int((np.abs(b) * 1e4 > threshold_bps).sum()),
        "threshold_bps": threshold_bps,
        "tracking_error_bps": float((rw - ru).std(ddof=1) * np.sqrt(365) * 1e4),
        "return_gap_bps": float((df["wrapped"].iloc[-1] / df["wrapped"].iloc[0]
                                 - df["underlying"].iloc[-1] / df["underlying"].iloc[0]) * 1e4),
        "correlation": float(np.corrcoef(ru, rw)[0, 1]),
    }


# ----------------------------------------------------------------------------------------------
# 5. Forks: one history, two assets
# ----------------------------------------------------------------------------------------------
def fork_split(value_before: float, share_minority: float, units: float = 1.0) -> dict[str, float]:
    """Illustrative value-preserving allocation, assuming the holder is credited both assets.

    Conservation of market value is an input to this example, not a property of real forks.
    Actual chain prices, claim eligibility and venue crediting must be observed separately.
    """
    if not 0.0 <= share_minority <= 1.0:
        raise ValueError("share_minority must be in [0, 1]")
    if value_before <= 0 or units <= 0:
        raise ValueError("value and units must be positive")
    major = value_before * (1.0 - share_minority) * units
    minor = value_before * share_minority * units
    return {"majority": major, "minority": minor, "total": major + minor,
            "ticker_only_return": (major / (value_before * units)) - 1.0,
            "holder_return": 0.0, "understatement": -share_minority}


THE_RULE = "RULE: carry balance and unit adjustments with price, and select the universe as of its date."


def main() -> None:                                          # noqa: C901 - a printed report
    print("=" * W)
    print("CRYPTO TOKEN EVENTS - synthetic accounting examples; historical sources retain their dates")
    print("=" * W)

    # ---- 1. rebase
    print("\n1. REBASE - price alone omits the balance factor")
    df = rebase_history()
    m = rebase_returns(df)
    print(f"   {m['days']} days, protocol rule supply *= 1 + (price/target - 1)/10 applied to "
          f"every address")
    print(f"   {'price only':<26}{m['price_return']:>12.2%}   what an OHLCV series shows")
    print(f"   {'supply index':<26}{m['supply_return']:>12.2%}   what the rebase did to the "
          f"balance")
    print(f"   {'holder value':<26}{m['value_return']:>12.2%}   price x balance, the real answer")
    print(f"   gap: the price-only return is wrong by {m['value_return'] - m['price_return']:.2%} "
          f"of notional over the sample")
    print(f"   identity (1+r_value) = (1+r_price)(1+r_supply): residual "
          f"{m['identity_residual']:.2e} at the")
    print(f"   full horizon and {m['daily_identity_max']:.2e} worst over "
          f"{m['n_returns']} daily steps -- it is arithmetic, not an approximation")
    print(f"   annual vol   price-only {m['price_vol']:>7.2%}   holder {m['value_vol']:>7.2%}")
    print(f"   Sharpe       price-only {m['price_sharpe']:>7.4f}   holder {m['value_sharpe']:>7.4f}"
          f"   daily-return correlation {m['corr']:.4f}")
    print("   ! Rebasing is one reason why price returns and holder returns differ; store the")
    print("     balance index separately. Other distributions and cash flows can also matter.")

    # ---- 2. redenomination
    print("\n2. REDENOMINATION - the same ticker, a different unit")
    rd = redenominated_series(new_per_old=STRAX_SWAP["new_per_old"])
    d = redenomination_damage(rd)
    print(f"   Source-verified: 1 old {STRAX_SWAP['ticker_before']} = "
          f"{STRAX_SWAP['new_per_old']:.0f} new {STRAX_SWAP['ticker_after']} -- SAME ticker -- "
          f"halt {STRAX_SWAP['halt_utc']},")
    print(f"   pairs {', '.join(STRAX_SWAP['pairs'])} delisted and pending orders cancelled.")
    print(f"   Source-verified: 1 {GAL_SWAP['ticker_before']} = "
          f"{GAL_SWAP['new_per_old']:.0f} {GAL_SWAP['ticker_after']} -- ticker CHANGES -- halt "
          f"{GAL_SWAP['halt_utc']},")
    print(f"   relisted {GAL_SWAP['relist_utc']} under the new symbol.")
    print(f"   {'':<26}{'unadjusted ticker':>20}{'holder':>14}")
    print(f"   {'total return':<26}{d['raw_total']:>20.2%}{d['true_total']:>14.2%}")
    print(f"   {'worst single day':<26}{d['raw_worst_day']:>20.2%}{d['true_worst_day']:>14.2%}")
    print(f"   {'annual vol':<26}{d['raw_vol']:>20.2%}{d['true_vol']:>14.2%}")
    print(f"   {'max drawdown':<26}{d['raw_maxdd']:>20.2%}{d['true_maxdd']:>14.2%}")
    print(f"   dividing the pre-event prints by {STRAX_SWAP['new_per_old']:.0f} reproduces the "
          f"holder's return series to {d['adjusted_matches_true']:.2e}")
    print("   ! The -90% day is the FACTOR, not a loss, and nothing in the OHLCV response says")
    print("     so. A ticker CHANGE at least breaks the join loudly; a ticker that survives the")
    print("     swap fails silently.")

    # ---- 3. survivorship
    print("\n3. SURVIVORSHIP - the pair list you can build is not the pair list that existed")
    print("   Same seeded volume and price paths in every row; only the median hazard changes.")
    print(f"   {'median pair':>12}{'realised':>10}{'pairs t0':>10}{'pairs t1':>10}"
          f"{'top-50 alive':>14}{'today in t0':>13}{'cohort ret':>12}{'survivors':>11}")
    rows = []
    for h in (0.10, 0.20, 0.35):
        s = survivorship(venue_history(annual_delist=h))
        rows.append((h, s))
        print(f"   {h:>12.0%}{s['realised_annual_delist']:>10.1%}{s['pairs_at_start']:>10}"
              f"{s['pairs_at_end']:>10}{s['survival_rate']:>14.0%}{s['backward_rate']:>13.0%}"
              f"{s['cohort_mean_return']:>12.1%}{s['survivor_mean_return']:>11.1%}")
    base = rows[1][1]
    print(f"   At a {rows[1][0]:.0%} hazard over {base['years']:.1f} years: "
          f"{base['survived']} of the day-0 top {base['n']} still trade, and only "
          f"{base['existed_then']} of")
    print(f"   today's top {base['n']} existed on day 0: {base['backward_rate']:.0%} "
          f"backward membership, with a different denominator.")
    print(f"   of TODAY'S selected list. Cohort mean return {base['cohort_mean_return']:.1%} against "
          f"{base['survivor_mean_return']:.1%} for survivors only:")
    print(f"   +{base['overstatement']:.1%} sample overstatement. Delisted positions are "
          f"assumed converted to cash at their")
    print("   last traded price. This is a valuation scenario, not a bound on real recovery.")
    print("   ! Historical Binance guidelines (2026-09-10 review), not a current policy check:")
    print(f"     orders {BINANCE_DELISTING['orders']}, and")
    print(f"     withdrawals stop ~{BINANCE_DELISTING['withdrawal_grace_months']} months later, "
          f"after which the {BINANCE_DELISTING['after']}.")
    print(f"     Review cadence: {BINANCE_DELISTING['review']}.")

    # ---- 4. wrapped and bridged
    print("\n4. WRAPPED AND BRIDGED - a 1:1 claim is 1:1 until the claim is questioned")
    wb = wrapped_basis()
    wm = wrapped_metrics(wb)
    print(f"   median |basis| {wm['median_abs_bps']:.1f} bp, 99th pct "
          f"{wm['p99_abs_bps']:.0f} bp, worst {wm['worst_bps']:.0f} bp, "
          f"{wm['days_beyond']} days beyond {wm['threshold_bps']:.0f} bp")
    print(f"   substituting the wrapper for the underlying: tracking error "
          f"{wm['tracking_error_bps']:.0f} bp/yr, total-return gap "
          f"{wm['return_gap_bps']:+.0f} bp,")
    print(f"   daily-return correlation {wm['correlation']:.4f} -- high correlation is exactly")
    print("   what makes the substitution look safe right up to the episode that matters")
    print(f"   ! {WRAPPED['backing']}, {WRAPPED['mint']}, {WRAPPED['verification']}.")
    print("     The basis can reflect funding, liquidity and redemption friction as well as")
    print("     custodian or bridge risk. Neither a zero basis nor mean reversion is guaranteed.")

    # ---- 5. forks
    print("\n5. FORKS - one history, two assets, and only one keeps the ticker")
    print(f"   Source-verified: the DAO hard fork executed at block {DAO_FORK['block']:,} on "
          f"{DAO_FORK['date']},")
    print(f"   moving ~{DAO_FORK['eth_moved_millions']:.0f}m ETH by irregular state change; "
          f"clients rejecting it ran {DAO_FORK['opt_out_flag']}")
    print(f"   and were warned about {DAO_FORK['hazard']}. Both chains continued.")
    print(f"   {'minority share':>16}{'majority':>12}{'minority':>12}{'total':>10}"
          f"{'ticker-only return':>21}")
    for share in (0.02, 0.10, 0.25):
        f = fork_split(100.0, share)
        print(f"   {share:>16.0%}{f['majority']:>12.2f}{f['minority']:>12.2f}{f['total']:>10.2f}"
              f"{f['ticker_only_return']:>21.1%}")
    print("   Value conservation is ASSUMED in this illustration; actual fork prices can differ.")
    print("   ! Record chain and block height, both asset identifiers, and claim eligibility.")
    print("     Whether you were even credited the second asset can depend on your")
    print("     custodian, so the correct entry is venue-specific and must be stored, not "
          "derived.")

    print("\n" + "=" * W)
    print(THE_RULE)
    print("=" * W)


if __name__ == "__main__":
    main()
