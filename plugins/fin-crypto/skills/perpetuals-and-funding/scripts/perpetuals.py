#!/usr/bin/env python3
"""Perpetual swaps: funding is a cash flow, and the price that liquidates you is not the one you
mark against.

Six demonstrations, each printing the numbers quoted in ../SKILL.md:

  1. funding      - accumulated over a seeded year of settlements; what a price-only perp
                    backtest is wrong by, in notional and in margin
  2. interval     - the venue's own formula already divides by (8/N), so a per-interval rate is
                    NOT comparable across symbols; the annualisation error of assuming 8h
  3. mark vs last - mark = median(price1, price2, last); how often a seeded path liquidates on
                    MARK while the last price never touched the level, and the reverse
  4. liquidation  - the level at 3x/5x/10x, what is left at it, and funding walking it toward you
  5. basis        - the perp's funding stream against real dated marks: two prices for one carry
  6. open interest- the same volume opens, closes or transfers a position; OI is the only one
                    of the three that says which

Seeded synthetic paths, arithmetic on formulas and marks read at the venues' own pages (URL and
date beside each constant). No network at run time, numpy/pandas only, fixed seed, no file
writes, ASCII output.

Run:  python perpetuals.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20260910
W = 96

# ----------------------------------------------------------------------------------------------
# Source-verified constants - read 2026-09-10 at the pages named
# ----------------------------------------------------------------------------------------------

# binance.com/en/support/faq/detail/360033525031  (Introduction to Binance Futures Funding Rates)
#   "Funding Rate (F) = [Average Premium Index (P) + clamp(interest rate - Premium Index (P),
#    0.05%, -0.05%)] / (8 / N)"
#   default settlement "every 8 hours at 00:00 (UTC), 08:00 (UTC), and 16:00 (UTC)"
#   "Binance does not charge fees on Funding Payments. Funding Payments are transferred directly
#    between traders holding opposing positions."
#   "Funding Amount = Nominal Value of Positions * Funding Rate",
#   "Nominal Value of Positions = Mark Price * Size of a Contract"
BINANCE_FUNDING = dict(clamp=0.0005, interest_per_interval=0.0001, default_hours=8,
                       settlements_utc=(0, 8, 16), venue_fee=0.0,
                       notional_basis="mark price x contract size")

# binance.com/en/support/faq/detail/360033525071  (Mark Price in USDT-Margined Futures)
#   "Price 1 = Price Index * (1 + Last Funding Rate * (Time Until Next Funding / Funding Period))"
#   "Price 2 = Price Index + Moving Average (30 seconds basis)"
#   "Mark Price = Median (Price 1, Price 2, Contract Price)"
BINANCE_MARK = dict(price1="index x funding adjustment", price2="index + MA(basis)",
                    price3="contract (last) price", rule="median", ma_seconds=30)

# binance.com/en/support/faq/detail/360033525271  (Binance Futures Liquidation Protocols)
#   "Liquidation occurs when the Mark Price hits the liquidation price of a position."
#   a Liquidation Clearance Fee is deducted; when the insurance fund cannot absorb a bankrupt
#   position the engine runs Auto-Deleveraging against opposing non-bankrupt traders.
BINANCE_LIQUIDATION = dict(trigger="mark price", fee="liquidation clearance fee",
                           backstop=("insurance fund", "auto-deleveraging"))

# bybit-exchange.github.io/docs/v5/market/instrument - instruments-info carries a per-instrument
# `fundingInterval` field, documented in MINUTES. The interval is a property of the symbol.
BYBIT_INTERVAL_FIELD = dict(endpoint="/v5/market/instruments-info", field="fundingInterval",
                            units="minutes", scope="per instrument")

# Fetched 2026-09-08 by ../../crypto-data-and-execution/scripts/perp_mechanics.py from the
# venues' own public endpoints; the constants are restated here so this file's arithmetic is
# self-contained. OKX GET /api/v5/public/funding-rate?instId=ANY (nextFundingTime - fundingTime
# per swap, 644 live swaps), GET /api/v5/public/position-tiers (tier 1), and
# Deribit GET /api/v2/public/get_book_summary_by_currency?currency=BTC&kind=future.
OKX_INTERVALS = {"8h": 357, "4h": 286, "1h": 1}
OKX_FUNDING_CAP = 0.00375                                    # maxFundingRate per settlement
OKX_TIER1_MMR = 0.004
DERIBIT_AS_OF = pd.Timestamp("2026-09-08T13:25:00Z")
DERIBIT_INDEX = 78449.67                                     # estimated_delivery_price
DERIBIT_FUTURES = {"BTC-25SEP26": (78561.31, "2026-09-25"),
                   "BTC-30OCT26": (78888.05, "2026-10-30"),
                   "BTC-27NOV26": (79192.98, "2026-11-27"),
                   "BTC-25DEC26": (79513.65, "2026-12-25"),
                   "BTC-26MAR27": (80458.36, "2027-03-26"),
                   "BTC-25JUN27": (81460.24, "2027-06-25")}
DERIBIT_EXPIRY_HOUR = 8                                      # every expiry settles at 08:00 UTC


# ----------------------------------------------------------------------------------------------
# 1-2. Funding
# ----------------------------------------------------------------------------------------------
def funding_rates(n_settlements: int = 1095, interval_hours: int = 8,
                  premium_sd: float = 0.0004, phi: float = 0.85,
                  interest: float = BINANCE_FUNDING["interest_per_interval"],
                  clamp: float = BINANCE_FUNDING["clamp"], cap: float = OKX_FUNDING_CAP,
                  seed: int = SEED) -> np.ndarray:
    """A synthetic funding history using a clamp-shaped illustrative formula.

    `F = [P + clamp(interest - P, +/-clamp)] / (8 / N)`, then clipped to the venue's per
    settlement cap. The default clamp and cap originate from DIFFERENT venues' historical
    examples; this is not a reproduction of any current exchange's complete funding rule.
    `P` is an AR(1) premium index. Scaling an identical premium path by `N/8` makes a 4-hour
    interval's rate half the 8-hour rate before the cap; actual venues may set different rules.
    """
    if n_settlements < 1 or interval_hours <= 0:
        raise ValueError("need at least one settlement and a positive interval")
    if not -1 < phi < 1 or min(premium_sd, clamp, cap) < 0:
        raise ValueError("need abs(phi) < 1 and non-negative premium scale, clamp and cap")
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, premium_sd * np.sqrt(1.0 - phi ** 2), n_settlements)
    prem = np.empty(n_settlements)
    p = 0.0
    for t in range(n_settlements):
        p = phi * p + eps[t]
        prem[t] = p
    raw = prem + np.clip(interest - prem, -clamp, clamp)
    return np.clip(raw / (8.0 / interval_hours), -cap, cap)


def funding_cost(rates: np.ndarray, notional: float = 100_000.0, leverage: float = 10.0,
                 side: str = "long") -> dict[str, float]:
    """What the position actually pays. Positive rate = longs pay shorts (both venues' docs)."""
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 1 or rates.size == 0 or not np.isfinite(rates).all():
        raise ValueError("rates must be a non-empty finite vector")
    if notional <= 0 or leverage <= 0:
        raise ValueError("notional and leverage must be positive")
    sign = 1.0 if side == "long" else -1.0
    paid = sign * notional * rates
    margin = notional / leverage
    return {"settlements": int(rates.size), "total_paid": float(paid.sum()),
            "pct_of_notional": float(paid.sum() / notional),
            "pct_of_margin": float(paid.sum() / margin),
            "mean_rate": float(rates.mean()), "median_rate": float(np.median(rates)),
            "positive_share": float((rates > 0).mean()),
            "worst_settlement": float(paid.max()), "best_settlement": float(paid.min()),
            "capped": int((np.abs(rates) >= OKX_FUNDING_CAP - 1e-12).sum())}


def perp_backtest(rates: np.ndarray, annual_vol: float = 0.60, drift: float = 0.10,
                  interval_hours: int = 8, seed: int = SEED) -> dict[str, float]:
    """Hold one linear contract, pay each funding rate on its SETTLEMENT mark notional.

    Initial equity and entry price are one. Cash funding is accumulated without reinvestment;
    the simulated price is also the funding mark. No leverage, fees or liquidation are modelled.
    """
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 1 or rates.size == 0 or not np.isfinite(rates).all():
        raise ValueError("rates must be a non-empty finite vector")
    if interval_hours <= 0 or annual_vol < 0:
        raise ValueError("interval must be positive and volatility non-negative")
    n = rates.size
    rng = np.random.default_rng(seed + 7)
    dt = interval_hours / (24.0 * 365.0)
    steps = rng.normal((drift - 0.5 * annual_vol ** 2) * dt, annual_vol * np.sqrt(dt), n)
    px = np.concatenate([[1.0], np.exp(np.cumsum(steps))])
    funding_cash = float(np.sum(px[1:] * rates))
    gross = float(px[-1] - 1.0)
    net = gross - funding_cash
    return {"gross_return": gross, "net_return": net, "funding_drag": float(gross - net),
            "funding_cash": funding_cash, "gross_final": 1.0 + gross, "net_final": 1.0 + net,
            "years": n * interval_hours / (24.0 * 365.0)}


def annualise_funding(rate_per_interval: float, interval_hours: float) -> float:
    """rate x intervals-per-day x 365. The interval is a property of the SYMBOL, not a constant."""
    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")
    return rate_per_interval * (24.0 / interval_hours) * 365.0


# ----------------------------------------------------------------------------------------------
# 3. Mark vs index vs last
# ----------------------------------------------------------------------------------------------
def _rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over `window` columns, expanding until the window fills."""
    if window < 1:
        raise ValueError("window must be at least 1")
    c = np.cumsum(a, axis=1)
    out = np.empty_like(a)
    k = min(window, a.shape[1])
    out[:, :k] = c[:, :k] / (np.arange(k) + 1.0)
    if a.shape[1] > window:
        out[:, window:] = (c[:, window:] - c[:, :-window]) / window
    return out


def venue_paths(n_paths: int = 1000, n_steps: int = 2160, bar_seconds: float = 10.0,
                annual_vol: float = 0.80, jump_rate: float = 8e-5, jump_sd: float = 0.010,
                book_lag: float = 0.40, premium_sd: float = 0.0010, phi: float = 0.99,
                wick_rate: float = 0.0020, wick_sd: float = 0.008,
                ma_seconds: float = BINANCE_MARK["ma_seconds"], last_funding: float = 0.0001,
                seed: int = SEED) -> dict[str, np.ndarray]:
    """Index, venue last price and mark price, built with Binance's own median formula.

    The composite index is a GBM plus rare jumps - a spot dislocation the perp's own book has
    not seen yet. The venue's last price follows the index with an exponential BOOK LAG, times
    an AR(1) premium, plus WICKS (a thin book printing away from the index for a bar or two).
    Mark = median(price1, price2, last) with price2 = index + MA(basis) over `ma_seconds`.

    That single median produces both errors at once: a wick is not in the MA yet, so mark stays
    at the index and does NOT follow it; a fast index move is not in the lagging book yet, so
    mark follows the INDEX and leaves the last price behind.
    """
    if n_paths < 1 or n_steps < 4 or bar_seconds <= 0:
        raise ValueError("need paths, at least 4 steps and a positive bar length")
    if not 0.0 < book_lag <= 1.0:
        raise ValueError("book_lag must be in (0, 1]")
    rng = np.random.default_rng(seed + 11)
    shape = (n_paths, n_steps)
    dt = bar_seconds / (365.0 * 24.0 * 3600.0)
    diffusive = rng.normal(-0.5 * annual_vol ** 2 * dt, annual_vol * np.sqrt(dt), shape)
    jumps = np.where(rng.random(shape) < jump_rate, rng.normal(0.0, jump_sd, shape), 0.0)
    idx = np.exp(np.cumsum(diffusive + jumps, axis=1))

    eps = rng.normal(0.0, premium_sd * np.sqrt(1 - phi ** 2), shape)
    prem = np.empty(shape)
    prem[:, 0] = eps[:, 0]
    for t in range(1, n_steps):
        prem[:, t] = phi * prem[:, t - 1] + eps[:, t]
    wick = np.where(rng.random(shape) < wick_rate, rng.normal(0.0, wick_sd, shape), 0.0)

    followed = np.empty(shape)                               # the book catches up geometrically
    followed[:, 0] = idx[:, 0]
    for t in range(1, n_steps):
        followed[:, t] = book_lag * idx[:, t] + (1.0 - book_lag) * followed[:, t - 1]
    last = followed * (1.0 + prem + wick)

    window = max(1, int(round(ma_seconds / bar_seconds)))
    price2 = idx + _rolling_mean(last - idx, window)
    price1 = idx * (1.0 + last_funding * 0.5)                # mid-interval, per the venue formula
    mark = np.median(np.stack([price1, price2, last]), axis=0)
    return {"index": idx, "last": last, "mark": mark, "price1": price1, "price2": price2,
            "ma_window_bars": np.array([window])}


def liq_ratio(leverage: float, mmr: float = OKX_TIER1_MMR, side: str = "long") -> float:
    """Isolated linear liquidation level as a fraction of entry, fees ignored."""
    if leverage <= 1.0 or not 0.0 <= mmr < 1.0:
        raise ValueError("leverage must exceed 1 and mmr must be a fraction")
    if side == "long":
        return (1.0 - 1.0 / leverage) / (1.0 - mmr)
    if side == "short":
        return (1.0 + 1.0 / leverage) / (1.0 + mmr)
    raise ValueError("side must be 'long' or 'short'")


def liquidation_gap(paths: dict[str, np.ndarray], leverage: float = 20.0,
                    mmr: float = OKX_TIER1_MMR, trigger: str = "mark") -> dict[str, float]:
    """How often `trigger` and the LAST price disagree about whether the position closed.

    `trigger` is the price the venue liquidates on - "mark" for the median formula, "index" for
    a venue that marks straight off the composite index. The backtest's assumption is always
    the last traded price, because that is what an OHLCV series contains.
    """
    if trigger not in ("mark", "index"):
        raise ValueError("trigger must be 'mark' or 'index'")
    lvl = liq_ratio(leverage, mmr, "long")
    trig_hit = paths[trigger].min(axis=1) <= lvl
    last_hit = paths["last"].min(axis=1) <= lvl
    n = int(trig_hit.size)
    return {"leverage": leverage, "trigger": trigger, "liq_level": lvl, "paths": n,
            "trigger_hits": int(trig_hit.sum()), "last_hits": int(last_hit.sum()),
            "trigger_only": int((trig_hit & ~last_hit).sum()),
            "last_only": int((last_hit & ~trig_hit).sum()),
            "trigger_only_share": float((trig_hit & ~last_hit).sum() / max(int(trig_hit.sum()), 1)),
            "last_only_share": float((last_hit & ~trig_hit).sum() / max(int(last_hit.sum()), 1)),
            "disagree_share": float(int((trig_hit ^ last_hit).sum()) / n),
            "max_gap_bps": float((np.abs(paths[trigger] - paths["last"])
                                  / paths["index"]).max() * 1e4)}


def liquidation_gap_sweep(leverages: tuple[float, ...] = (20.0, 50.0, 100.0),
                          triggers: tuple[str, ...] = ("mark", "index"),
                          n_paths: int = 6000, chunk: int = 1000,
                          seed: int = SEED, **kwargs) -> list[dict[str, float]]:
    """`liquidation_gap` over many paths and both trigger conventions, run in chunks."""
    if chunk < 1 or n_paths < 1:
        raise ValueError("n_paths and chunk must be positive")
    keys = ("paths", "trigger_hits", "last_hits", "trigger_only", "last_only")
    totals = [{k: 0 for k in keys} | {"leverage": lev, "trigger": tr, "max_gap_bps": 0.0}
              for tr in triggers for lev in leverages]
    done = 0
    while done < n_paths:
        m = min(chunk, n_paths - done)
        p = venue_paths(n_paths=m, seed=seed + done, **kwargs)
        for slot in totals:
            g = liquidation_gap(p, leverage=slot["leverage"], trigger=slot["trigger"])
            for k in keys:
                slot[k] += g[k]
            slot["liq_level"] = g["liq_level"]
            slot["max_gap_bps"] = max(slot["max_gap_bps"], g["max_gap_bps"])
        done += m
    for slot in totals:
        slot["trigger_only_share"] = slot["trigger_only"] / max(slot["trigger_hits"], 1)
        slot["last_only_share"] = slot["last_only"] / max(slot["last_hits"], 1)
        slot["disagree_share"] = (slot["trigger_only"] + slot["last_only"]) / slot["paths"]
    return totals


def funding_walks_the_line(leverage: float = 10.0, rate: float = 0.0001, days: int = 30,
                           interval_hours: int = 8,
                           mmr: float = OKX_TIER1_MMR) -> dict[str, float]:
    """Funding is paid out of margin, so a long's liquidation level rises with the price flat."""
    n = int(days * 24 / interval_hours)
    paid = rate * n                                          # of NOTIONAL
    start = liq_ratio(leverage, mmr, "long")
    eff = 1.0 / leverage - paid                              # margin left, as a share of notional
    end = (1.0 - eff) / (1.0 - mmr)
    return {"days": days, "settlements": n, "paid_pct_notional": paid,
            "paid_pct_margin": paid * leverage, "liq_start": start, "liq_end": end,
            "moved_pct": end - start}


# ----------------------------------------------------------------------------------------------
# 5. Basis
# ----------------------------------------------------------------------------------------------
def annualised_basis(mark: float, index: float, days: float, compound: bool = False) -> float:
    """(F/S - 1) * 365/days, or the compounded form. The dated leg of the same carry."""
    if days <= 0 or index <= 0:
        raise ValueError("days and index must be positive")
    b = mark / index - 1.0
    return (1.0 + b) ** (365.0 / days) - 1.0 if compound else b * 365.0 / days


def deribit_curve() -> pd.DataFrame:
    """The 2026-09-08 Deribit BTC futures marks, annualised. Fetched marks, this file's maths."""
    rows = []
    for name, (mark, expiry) in DERIBIT_FUTURES.items():
        exp = pd.Timestamp(expiry, tz="UTC") + pd.Timedelta(hours=DERIBIT_EXPIRY_HOUR)
        days = (exp - DERIBIT_AS_OF).total_seconds() / 86400.0
        rows.append({"contract": name, "mark": mark, "days": days,
                     "basis": mark / DERIBIT_INDEX - 1.0,
                     "simple_yr": annualised_basis(mark, DERIBIT_INDEX, days),
                     "compound_yr": annualised_basis(mark, DERIBIT_INDEX, days, compound=True)})
    return pd.DataFrame(rows).set_index("contract")


# ----------------------------------------------------------------------------------------------
# 6. Open interest
# ----------------------------------------------------------------------------------------------
def oi_tape(n_trades: int = 200_000, p_open: float = 0.30, p_close: float = 0.25,
            per_day: int = 500, seed: int = SEED,
            initial_oi: float = 1_000_000.0) -> dict[str, float]:
    """Synthetic trade classifications: aggregate OI change cannot identify trader motives.

    open/open -> OI +size, close/close -> OI -size, open/close -> OI unchanged (a transfer).
    Volume is the size in all three cases.
    """
    if not 0.0 <= p_open + p_close <= 1.0 or p_open < 0 or p_close < 0:
        raise ValueError("p_open and p_close must be probabilities that sum to at most 1")
    if per_day < 2 or n_trades < per_day:
        raise ValueError("need at least two trades a day and a day of them")
    rng = np.random.default_rng(seed + 13)
    size = np.exp(rng.normal(0.0, 1.1, n_trades))            # heavy-tailed trade sizes
    u = rng.random(n_trades)
    d_oi = np.where(u < p_open, size, np.where(u < p_open + p_close, -size, 0.0))
    levels = initial_oi + np.cumsum(d_oi)
    if initial_oi < 0 or np.any(levels < 0):
        raise ValueError("initial_oi must cover all simulated closing trades")
    days = n_trades // per_day
    v_day = size[: days * per_day].reshape(days, per_day).sum(axis=1)
    o_day = d_oi[: days * per_day].reshape(days, per_day).sum(axis=1)
    return {"trades": int(n_trades), "volume": float(size.sum()),
            "oi_start": initial_oi, "oi_end": float(levels[-1]), "oi_change": float(d_oi.sum()),
            "oi_per_volume": float(d_oi.sum() / size.sum()),
            "share_transfers": float((d_oi == 0.0).mean()),
            "corr_daily_volume_oi": float(np.corrcoef(v_day, o_day)[0, 1]),
            "corr_daily_volume_abs_oi": float(np.corrcoef(v_day, np.abs(o_day))[0, 1]),
            "days": int(days)}


THE_RULE = "RULE: debit funding on settlement notional; keep mark-based liquidation separate from last price."


def main() -> None:                                          # noqa: C901 - a printed report
    print("=" * W)
    print("PERPETUAL SWAPS - the carry a price series has not got, and the price that closes you")
    print("=" * W)

    # ---- 1. funding accumulates
    print("\n1. FUNDING IS A CASH FLOW, NOT A PRICE ADJUSTMENT")
    rates = funding_rates()
    c = funding_cost(rates)
    bt = perp_backtest(rates)
    print(f"   {c['settlements']} eight-hour settlements ({bt['years']:.1f} years), rate from "
          f"a synthetic clamp-shaped model:")
    print(f"   F = P + clamp({BINANCE_FUNDING['interest_per_interval']:.2%} - P, +/-"
          f"{BINANCE_FUNDING['clamp']:.2%}), capped at +/-{OKX_FUNDING_CAP:.3%} per settlement")
    print(f"   mean rate {c['mean_rate']:.5%}/8h   median {c['median_rate']:.5%}   positive "
          f"{c['positive_share']:.0%} of settlements")
    print(f"   a $100,000 long paid {c['total_paid']:,.0f} = {c['pct_of_notional']:.2%} of "
          f"notional and {c['pct_of_margin']:.1%} OF MARGIN at 10x")
    print(f"   {'fixed quantity, moving mark':<34}{'return':>12}")
    print(f"   {'price-only backtest':<34}{bt['gross_return']:>12.2%}")
    print(f"   {'funding-inclusive':<34}{bt['net_return']:>12.2%}")
    print(f"   {'drag':<34}{bt['funding_drag']:>12.2%}")
    print("   ! The fixed-notional bill and fixed-quantity P&L use different exposure policies;")
    print("     cash funding uses the mark at settlement, not a price-series adjustment.")

    # ---- 2. the interval belongs to the symbol
    print("\n2. THE INTERVAL BELONGS TO THE SYMBOL, AND THE VENUE ALREADY SCALED THE RATE")
    print(f"   Source-verified: Bybit publishes {BYBIT_INTERVAL_FIELD['field']} in "
          f"{BYBIT_INTERVAL_FIELD['units']}, {BYBIT_INTERVAL_FIELD['scope']}")
    print(f"   ({BYBIT_INTERVAL_FIELD['endpoint']}). This repo's 2026-09-08 OKX pull found "
          f"{OKX_INTERVALS['8h']} of {sum(OKX_INTERVALS.values())}")
    print(f"   live swaps at 8h, {OKX_INTERVALS['4h']} at 4h and {OKX_INTERVALS['1h']} at 1h. "
          f"The illustrative model scales by (8/N), using")
    print("   the same seeded distribution at each sampling interval:")
    print(f"   {'interval':>10}{'mean rate/interval':>21}{'per day':>11}{'annualised':>13}"
          f"{'x3x365, the 8h habit':>23}")
    for hours in (8, 4, 1):
        r = funding_rates(n_settlements=int(1095 * 8 / hours), interval_hours=hours)
        per = r.mean()
        print(f"   {f'{hours}h':>10}{per:>21.5%}{per * 24 / hours:>11.4%}"
              f"{annualise_funding(per, hours):>13.2%}{per * 3.0 * 365.0:>23.2%}")
    print("   ! The per-interval rate is NOT comparable across symbols. Annualise with the")
    print("     symbol's own interval; the 'three settlements a day' habit understates a 4h")
    print("     symbol by 2x and a 1h symbol by 8x.")

    # ---- 3. mark vs last
    print("\n3. MARK vs INDEX vs LAST - the price that closes you is not the one on the chart")
    print(f"   Historical median-mark example: mark = {BINANCE_MARK['rule']} of ({BINANCE_MARK['price1']}),")
    print(f"   ({BINANCE_MARK['price2']}) and ({BINANCE_MARK['price3']}); liquidation triggers "
          f"on the {BINANCE_LIQUIDATION['trigger']},")
    print(f"   with a {BINANCE_LIQUIDATION['fee']} and, behind it, an "
          f"{BINANCE_LIQUIDATION['backstop'][0]} and {BINANCE_LIQUIDATION['backstop'][1]}.")
    sweep = liquidation_gap_sweep()
    print(f"   {sweep[0]['paths']} seeded 6-hour paths at 10-second bars: index jumps the local "
          f"book has not seen,")
    print(f"   a book that catches up geometrically, wicks, and a "
          f"{BINANCE_MARK['ma_seconds']}-second MA of the basis.")
    print("   Two trigger conventions against the same paths; the backtest always uses LAST.")
    print(f"   {'trigger':>9}{'leverage':>10}{'liq level':>11}{'trig hits':>11}{'last hits':>11}"
          f"{'TRIG only':>11}{'LAST only':>11}{'disagree':>10}")
    for g in sweep:
        print(f"   {g['trigger']:>9}{g['leverage']:>9.0f}x{g['liq_level'] - 1.0:>11.2%}"
              f"{g['trigger_hits']:>11}{g['last_hits']:>11}{g['trigger_only']:>11}"
              f"{g['last_only']:>11}{g['disagree_share']:>10.1%}")
    med = [g for g in sweep if g["trigger"] == "mark" and g["leverage"] == 50.0][0]
    ix = [g for g in sweep if g["trigger"] == "index" and g["leverage"] == 50.0][0]
    print(f"   At 50x under the MEDIAN mark: {med['trigger_only']} liquidations on a level the "
          f"last price never touched,")
    print(f"   and {med['last_only']} paths a last-price backtest calls liquidated were not "
          f"({med['last_only_share']:.1%} of its hits).")
    print(f"   Under an INDEX-only mark: {ix['trigger_only']} and {ix['last_only']} - "
          f"{ix['trigger_only'] / max(med['trigger_only'], 1):.0f}x more of the first kind.")
    print(f"   Widest gap to last on any bar: median mark {med['max_gap_bps']:.0f} bp, index "
          f"{ix['max_gap_bps']:.0f} bp.")
    print("   ! Both errors are live at once and they are one mechanism. The median ignores a")
    print("     wick the MA has not absorbed (so LAST-only hits dominate) and follows the index")
    print("     away from a lagging book (TRIG-only). How far it leans is a VENUE DESIGN")
    print("     CHOICE: read the venue's mark formula before you believe your own stop.")

    # ---- 4. liquidation is not a stop
    print("\n4. LIQUIDATION IS NOT A STOP-LOSS")
    print(f"   Isolated linear, maintenance margin {OKX_TIER1_MMR:.1%}, fees ignored:")
    print(f"   {'leverage':>9}{'long liq':>11}{'short liq':>11}{'bankruptcy':>12}"
          f"{'margin left':>13}")
    for lev in (3.0, 5.0, 10.0):
        lo, sh = liq_ratio(lev, side="long"), liq_ratio(lev, side="short")
        print(f"   {lev:>8.0f}x{lo - 1.0:>11.2%}{sh - 1.0:>11.2%}{-1.0 / lev:>12.2%}"
              f"{(1.0 / lev - (1.0 - lo)) * lev:>13.1%}")
    fw = funding_walks_the_line()
    print(f"   The level also moves on its own: a 10x long paying "
          f"{BINANCE_FUNDING['interest_per_interval']:.2%} per 8h for {fw['days']} days")
    print(f"   spends {fw['paid_pct_notional']:.2%} of notional = {fw['paid_pct_margin']:.1%} of "
          f"margin, walking its liquidation from")
    print(f"   {fw['liq_start'] - 1:.2%} to {fw['liq_end'] - 1:.2%} with the price unchanged "
          f"({fw['moved_pct']:+.2%} closer).")
    print("   ! At 10x it fires with a few percent of margin left, and a clearance fee takes")
    print("     part of that. It is a forced market order at the worst moment, not a stop.")

    # ---- 5. basis
    print("\n5. BASIS - the dated contract's price for the same carry")
    curve = deribit_curve()
    print(f"   Source-verified Deribit marks, {DERIBIT_AS_OF:%Y-%m-%d %H:%M} UTC, index "
          f"{DERIBIT_INDEX:,.2f}:")
    print(f"   {'contract':<14}{'mark':>12}{'days':>8}{'basis':>10}{'simple /yr':>13}"
          f"{'compounded /yr':>16}")
    for name, r in curve.iterrows():
        print(f"   {name:<14}{r['mark']:>12,.2f}{r['days']:>8.1f}{r['basis']:>10.3%}"
              f"{r['simple_yr']:>13.2%}{r['compound_yr']:>16.2%}")
    print(f"   Synthetic annual funding is "
          f"{annualise_funding(rates.mean(), 8):.2%}/yr; the historical dated curve is "
          f"{curve['simple_yr'].min():.2%}-{curve['simple_yr'].max():.2%}/yr.")
    print("   ! These are different samples, not a market comparison. A dated hedge can lock")
    print("     basis conditional on fees, financing and margin survival; future perp funding")
    print("     is unknown and can change sign. See futures-continuous-contracts for rolls.")

    # ---- 6. open interest
    print("\n6. OPEN INTEREST - volume cannot tell you whether a position was opened")
    oi = oi_tape()
    print(f"   {oi['trades']:,} trades, each with a buyer and a seller who is opening or closing:")
    print("   open/open -> OI up   close/close -> OI down   open/close -> OI unchanged")
    print(f"   volume {oi['volume']:,.0f}, OI change {oi['oi_change']:,.0f} = "
          f"{oi['oi_per_volume']:.3f} of volume; {oi['share_transfers']:.0%} of trades were "
          f"pure transfers")
    print(f"   correlation of daily volume with daily OI change over {oi['days']} days: "
          f"{oi['corr_daily_volume_oi']:+.4f}")
    print(f"   with the ABSOLUTE OI change: {oi['corr_daily_volume_abs_oi']:+.4f}")
    print("   ! A volume spike is compatible with OI rising, falling or flat. Price and OI")
    print("     alone do not identify bullish buying versus shorts covering: every trade has")
    print("     both a buyer and a seller. These correlations belong only to the simulated tape.")

    print("\n" + "=" * W)
    print(THE_RULE)
    print("=" * W)


if __name__ == "__main__":
    main()
