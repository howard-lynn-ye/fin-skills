"""FX quote conventions, pip sizing and carry -- the three things that silently invert or
mis-scale an FX backtest.

WHY this exists. FX has no exchange, so it has no enforced conventions -- only customs that
everyone assumes you know. Three of those customs, unstated, destroy results:

1. QUOTE DIRECTION. 'EURUSD' is dollars per euro; 'USDJPY' is yen per dollar. The market
   orders currencies EUR > GBP > AUD > NZD > USD > CAD > CHF > JPY and quotes the higher one
   as base, so USD is the BASE in USDJPY/USDCHF/USDCAD and the QUOTE in EURUSD/GBPUSD/
   AUDUSD/NZDUSD. A "long dollars" book is therefore LONG USDJPY and SHORT EURUSD. Stack
   those into one signal without inverting and half your positions are backwards -- and the
   backtest will not crash, it will just report the wrong sign with a straight face.

2. PIP SIZE. JPY-quoted pairs carry TWO decimal places, so a pip is 0.01. Almost everything
   else carries four, so a pip is 0.0001. Size a USDJPY position with the 0.0001 default
   and you are off by a factor of 100 -- not 1%, not 10%, ONE HUNDRED TIMES. A $150k
   position becomes a $15mm position, and a 100-pip day turns a $1,000 budgeted loss into
   $100,000. This exact error is why brokers put JPY pairs behind a separate lot calculator.

3. CARRY. An FX position is two money-market legs: you are long the base's rate and short
   the quote's. That interest differential is credited or debited every rollover, and it is
   the ENTIRE return of the carry trade. Backtest a carry pair on the spot series alone and
   you will see a loser -- because the spot really does drift down, roughly by the forward
   discount. The strategy only exists in total return. It is not a rounding adjustment; it
   is the position's whole reason for being.

   Sign trap: the forward points (F - S, the market's quote) and the carry return have
   OPPOSITE signs by construction. A high-yielding base trades at a forward DISCOUNT while
   its holder is CREDITED interest. Both numbers below are labelled so you cannot mix them.

Usage:
    from fx_conventions import parse_pair, pip_value, carry_return, total_return
    pip_value("USDJPY", 100_000, 150.25).value_usd     # 6.66, not 0.067
    tr = total_return(audusd_spot, r_base=0.0475, r_quote=0.0025, pair="AUDUSD")
    tr[["spot_ret", "carry_ret", "total_ret"]].add(1).prod() - 1
"""
from __future__ import annotations

from typing import Mapping, NamedTuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Market quoting precedence: the currency appearing EARLIER is quoted as the base.
# This is the rule; USD-base vs USD-quote falls out of it rather than being memorised.
CCY_PRECEDENCE: tuple[str, ...] = (
    "EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "NOK", "SEK", "DKK", "JPY",
)
_RANK = {c: i for i, c in enumerate(CCY_PRECEDENCE)}

# Pip = one unit of the last quoted decimal. JPY quotes to 2dp, everything else to 4dp.
PIP_DEFAULT = 1e-4
PIP_JPY = 1e-2
_TWO_DP_QUOTES = frozenset({"JPY"})

# Money-market day count of the DEPOSIT leg. Sterling-bloc currencies accrue ACT/365,
# the rest ACT/360. Using 360 for AUD overstates a year of carry by ~1.4% of the rate.
DAY_COUNT: Mapping[str, int] = {"GBP": 365, "AUD": 365, "NZD": 365, "HKD": 365,
                                "SGD": 365, "ZAR": 365}
DAY_COUNT_DEFAULT = 360


class Pip(NamedTuple):
    """Result of `pip_value`. `value_usd` is what your risk system needs."""
    pip_size: float
    value_quote: float      # one pip on this notional, in the QUOTE currency
    value_usd: float        # the same thing in USD (nan for non-USD crosses)
    quote_ccy: str


class Carry(NamedTuple):
    """Result of `carry_return`.

    `ret`    -- fractional P&L on notional from the rate differential. THIS is the number
                a backtest adds to the spot return.
    `points` -- the market's swap quote, F - S, in price terms. OPPOSITE SIGN to `ret`.
    `pips`   -- `points` expressed in pips for the pair (JPY-aware).
    """
    ret: float
    forward: float
    points: float
    pips: float


def parse_pair(pair: str) -> tuple[str, str]:
    """'EURUSD' | 'EUR/USD' | 'eurusd' -> ('EUR', 'USD'). Base first, quote second.

    Failure prevented: a config file that mixes 'EURUSD' and 'EUR/USD' and a slicing
    routine that assumes one of them. Six characters versus seven is a silent off-by-one
    that turns 'GBP/USD' into ('GBP', '/US').
    """
    s = str(pair).strip().upper().replace("/", "").replace("-", "").replace("_", "")
    if len(s) != 6 or not s.isalpha():
        raise ValueError(f"cannot parse FX pair {pair!r}: expected 6 letters like "
                         f"'EURUSD' or 'EUR/USD', got {s!r}")
    return s[:3], s[3:]


def check_convention(pair: str) -> None:
    """Raise if the pair is written backwards from the market convention.

    Failure prevented: a data vendor (or a colleague) handing you 'JPYUSD'. It parses
    cleanly, the numbers look like prices, and every return you compute is inverted.
    """
    base, quote = parse_pair(pair)
    rb, rq = _RANK.get(base), _RANK.get(quote)
    if rb is None or rq is None:
        return                                     # unknown currency: nothing to assert
    if rb > rq:
        raise ValueError(
            f"{base}{quote} is quoted backwards: market convention is {quote}{base} "
            f"(precedence {CCY_PRECEDENCE}). If your data really is {base}{quote}, invert "
            f"it with 1/price BEFORE anything else touches it -- otherwise every return, "
            f"pip value and carry sign in the book is flipped.")


def is_inverted(pair: str) -> bool:
    """True when USD is the BASE (USDJPY, USDCHF, USDCAD): the price is FOREIGN PER USD.

    False for EURUSD, GBPUSD, AUDUSD, NZDUSD, where the price is USD per foreign unit.

    Why it matters: for an inverted pair a RISING price means a STRONGER dollar, and a USD
    P&L needs a 1/price conversion. Combine a "long USD" signal across both families
    without checking this and half the book is short.
    """
    base, quote = parse_pair(pair)
    if "USD" not in (base, quote):
        raise ValueError(f"{pair} is a cross with no USD leg; 'inverted' is undefined. "
                         f"Decide explicitly which leg your P&L is measured in.")
    return base == "USD"


def pip_size(pair: str) -> float:
    """0.01 for JPY-quoted pairs, 0.0001 otherwise. The 100x error lives here."""
    _, quote = parse_pair(pair)
    return PIP_JPY if quote in _TWO_DP_QUOTES else PIP_DEFAULT


def pip_value(pair: str, notional: float, price: float) -> Pip:
    """Value of one pip on `notional` units of the BASE currency, at `price`.

    EURUSD, 100,000 EUR  -> 100,000 * 0.0001            = $10.00
    USDJPY, 100,000 USD  -> 100,000 * 0.01   / 150.25   = $6.66   (NOT $0.0666)

    The USD leg of an inverted pair is divided by the price because the pip is denominated
    in the QUOTE currency (yen) and has to be converted back. Skipping that division is the
    second-most-common FX sizing bug, after the pip size itself.
    """
    base, quote = parse_pair(pair)
    if not np.isfinite(price) or price <= 0:
        raise ValueError(f"price must be positive and finite, got {price!r}")
    ps = pip_size(pair)
    value_quote = float(notional) * ps
    if quote == "USD":
        usd = value_quote                          # pip is already in dollars
    elif base == "USD":
        usd = value_quote / float(price)           # pip is in the foreign ccy: convert back
    else:
        usd = float("nan")                         # cross: needs a third rate, not guessed
    return Pip(pip_size=ps, value_quote=value_quote, value_usd=usd, quote_ccy=quote)


def notional_for_pip_risk(pair: str, target_usd_per_pip: float, price: float) -> float:
    """Base-currency notional that makes one pip worth `target_usd_per_pip`.

    This is the function a sizing bug actually flows through. Feed it the wrong pip size
    and it hands back a position 100x too large without a murmur.
    """
    unit = pip_value(pair, 1.0, price).value_usd
    if not np.isfinite(unit) or unit == 0:
        raise ValueError(f"cannot size {pair} in USD without a third leg (it is a cross)")
    return float(target_usd_per_pip) / unit


def _basis(ccy: str) -> int:
    return DAY_COUNT.get(ccy, DAY_COUNT_DEFAULT)


def carry_return(spot: float, r_base: float, r_quote: float, days: float,
                 pair: str | None = None) -> Carry:
    """Interest differential on a LONG-base / SHORT-quote position held `days`.

    A spot FX position is financed: you hold the base currency on deposit at `r_base` and
    borrow the quote at `r_quote`. Over `days` that is
        ret = (1 + r_base * days/basis_base) / (1 + r_quote * days/basis_quote) - 1
    which is exactly the swap/rollover your prime broker credits or debits. Long AUDUSD at
    4.75% vs 0.25% earns ~4.5% a year for doing nothing but staying long.

    `pair` (optional) selects the ACT/360 vs ACT/365 basis per leg. Omit it and both legs
    use ACT/360, which overstates a sterling-bloc carry by about 1.4% of the rate.

    Covered interest parity then puts the forward at F = S * (1+r_q*T_q)/(1+r_b*T_b), so a
    high-yielding base trades at a forward DISCOUNT while its holder is paid interest:
    `points` and `ret` have opposite signs, always. That is not a bug, it is the trade.
    """
    if days < 0:
        raise ValueError(f"days must be non-negative, got {days}")
    if pair is not None:
        base, quote = parse_pair(pair)
        bb, bq = _basis(base), _basis(quote)
    else:
        bb = bq = DAY_COUNT_DEFAULT
    tb, tq = float(days) / bb, float(days) / bq
    growth_base = 1.0 + float(r_base) * tb
    growth_quote = 1.0 + float(r_quote) * tq
    ret = growth_base / growth_quote - 1.0
    fwd = float(spot) * growth_quote / growth_base
    points = fwd - float(spot)
    pips = points / (pip_size(pair) if pair is not None else PIP_DEFAULT)
    return Carry(ret=ret, forward=fwd, points=points, pips=pips)


def total_return(spot_series: pd.Series, r_base: float, r_quote: float,
                 pair: str | None = None) -> pd.DataFrame:
    """Spot return, carry return and TOTAL return of a long-base position, per period.

    Carry accrues on CALENDAR days, not trading days -- a Friday-to-Monday roll earns three
    days of interest, which is the real "triple swap Wednesday" a spot book sees. Inferring
    the gaps from the index gets that right for free; multiplying by 1/252 does not, and
    understates a year of carry by roughly 30%.

    Returns a DataFrame with `spot_ret`, `carry_ret`, `total_ret`. The total compounds the
    two, because both are returns on the same notional over the same period.

    NOTE: measured in the QUOTE currency. For AUDUSD that is USD; for USDJPY it is JPY.
    Check `is_inverted(pair)` before booking the number into a dollar P&L.
    """
    s = pd.Series(spot_series).astype(float).dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("spot_series needs a DatetimeIndex so calendar-day carry accrual "
                        "can be inferred; trading-day counts understate carry ~30%")
    if not s.index.is_monotonic_increasing:
        raise ValueError("spot_series must be sorted ascending")
    if (s <= 0).any():
        raise ValueError("spot prices must be positive")

    days = pd.Series(s.index.to_series().diff().dt.days.to_numpy(), index=s.index)
    spot_ret = s.pct_change(fill_method=None)
    carry = pd.Series(
        [carry_return(1.0, r_base, r_quote, d, pair).ret if np.isfinite(d) else np.nan
         for d in days], index=s.index)
    out = pd.DataFrame({"spot": s, "calendar_days": days, "spot_ret": spot_ret,
                        "carry_ret": carry})
    out["total_ret"] = (1.0 + out["spot_ret"]) * (1.0 + out["carry_ret"]) - 1.0
    return out.iloc[1:]


def _stats(r: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict[str, float]:
    r = pd.Series(r).dropna()
    total = float((1.0 + r).prod() - 1.0)
    ann = float((1.0 + total) ** (periods_per_year / len(r)) - 1.0)
    sd = float(r.std(ddof=1))
    return {"total": total, "ann": ann, "vol": sd * np.sqrt(periods_per_year),
            "sharpe": float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan}


def render_carry(tr: pd.DataFrame, pair: str, r_base: float, r_quote: float,
                 title: str = "SPOT-ONLY vs TOTAL RETURN") -> str:
    """Fixed-width report. If your FX backtest cannot produce this table, it is on spot."""
    w = 84
    base, quote = parse_pair(pair)
    spot_s, tot_s = _stats(tr["spot_ret"]), _stats(tr["total_ret"])
    car_s = _stats(tr["carry_ret"])
    lines = [f"{title}: long {pair}", "=" * w,
             f"{base} deposit {r_base:.2%}   {quote} funding {r_quote:.2%}   "
             f"differential {r_base - r_quote:+.2%}   ({len(tr):,} days)",
             "-" * w,
             f"{'series':<26} {'cumulative':>12} {'annualised':>12} {'vol':>9} "
             f"{'Sharpe':>9}   verdict", "-" * w]
    for name, st in (("spot only (what you'd see)", spot_s),
                     ("carry alone", car_s),
                     ("TOTAL (spot + carry)", tot_s)):
        verdict = "LOSER" if st["ann"] < 0 else "winner"
        # Carry is deterministic while rates are held fixed, so its standalone Sharpe is
        # an artefact of the assumption, not a result. Refuse to print it.
        sharpe = "n/m" if st["vol"] < 0.01 else f"{st['sharpe']:.2f}"
        lines.append(f"{name:<26} {st['total']:>12.2%} {st['ann']:>12.2%} "
                     f"{st['vol']:>9.2%} {sharpe:>9}   {verdict}")
    lines.append("-" * w)
    lines.append(f"Backtesting on spot alone throws away {car_s['total']:.1%} of "
                 f"cumulative return and flips the")
    lines.append(f"conclusion from {spot_s['ann']:+.2%}/yr (Sharpe {spot_s['sharpe']:+.2f}) "
                 f"to {tot_s['ann']:+.2%}/yr (Sharpe {tot_s['sharpe']:+.2f}).")
    lines.append("=" * w)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
def _synthetic_carry_pair(years: int = 6, drift: float = -0.030, vol: float = 0.095,
                          s0: float = 0.9800, seed: int = 13) -> pd.Series:
    """A high-yielder that grinds DOWN in spot. Offline, no network.

    Shaped like AUDUSD through the post-GFC carry years: the Australian cash rate sat near
    4.75% against a US rate near zero, and spot slowly gave back roughly (but not all of)
    the differential -- exactly what uncovered interest parity predicts and exactly what
    the carry trade is a bet against.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=TRADING_DAYS * years)
    dt = 1.0 / TRADING_DAYS
    steps = rng.normal((drift - 0.5 * vol ** 2) * dt, vol * np.sqrt(dt), len(idx))
    return pd.Series(s0 * np.exp(np.cumsum(steps)), index=idx, name="AUDUSD")


if __name__ == "__main__":
    pd.set_option("display.width", 140)

    # ------------------------------------------------------- 1. quote-direction table
    print("=" * 84)
    print("1. QUOTE CONVENTIONS -- which side is the dollar on?")
    print("=" * 84)
    print(f"{'pair':<9} {'base':>5} {'quote':>6} {'USD is':>9} {'is_inverted':>12} "
          f"{'pip size':>10}   a rising price means")
    print("-" * 84)
    for p in ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"):
        b, q = parse_pair(p)
        inv = is_inverted(p)
        print(f"{p:<9} {b:>5} {q:>6} {'BASE' if inv else 'QUOTE':>9} {str(inv):>12} "
              f"{pip_size(p):>10.4f}   "
              f"{'STRONGER dollar' if inv else 'WEAKER dollar'}")
    print("-" * 84)
    print("A single 'long USD' signal must BUY the USD-base rows and SELL the others.")
    try:
        check_convention("JPYUSD")
    except ValueError as e:
        print(f"\ncheck_convention('JPYUSD') CAUGHT: {str(e)[:150]}...")

    # ---------------------------------------------------------- 2. the JPY 100x error
    print("\n" + "=" * 84)
    print("2. THE JPY PIP ERROR, AS A CONCRETE MIS-SIZED POSITION")
    print("=" * 84)
    price, budget = 150.25, 10.0                   # USDJPY, target $10 risk per pip
    right = pip_value("USDJPY", 100_000, price)
    # USD value of one pip per 1 unit of notional -- correct, and with the 4dp default.
    unit_ok = pip_value("USDJPY", 1.0, price).value_usd
    unit_bug = PIP_DEFAULT / price
    print(f"USDJPY @ {price}, notional $100,000")
    print(f"  correct  pip = {right.pip_size:<7.4f} -> one pip is "
          f"{right.value_quote:>9,.2f} JPY = ${100_000 * unit_ok:>8,.2f}")
    print(f"  4dp bug  pip = {PIP_DEFAULT:<7.4f} -> one pip is "
          f"{100_000 * PIP_DEFAULT:>9,.2f} JPY = ${100_000 * unit_bug:>8,.2f}")
    print(f"  the bug UNDERSTATES risk per pip by {unit_ok / unit_bug:,.0f}x ...")
    print()
    # ... which is why it OVERSTATES the position: notional = budget / value-per-pip.
    n_right = notional_for_pip_risk("USDJPY", budget, price)
    n_wrong = budget / unit_bug
    print(f"... so sizing a position for ${budget:.0f} of risk per pip gives:")
    print(f"  correct  -> ${n_right:>14,.0f} notional")
    print(f"  4dp bug  -> ${n_wrong:>14,.0f} notional   <-- "
          f"{n_wrong / n_right:,.0f}x OVERSIZED")
    move = 100                                     # a routine 1-yen day
    print(f"\nUSDJPY then moves {move} pips (1.00 yen -- an ordinary session):")
    print(f"  budgeted loss           ${budget * move:>14,.0f}")
    print(f"  loss on the bug's size  ${n_wrong * unit_ok * move:>14,.0f}"
          f"   <-- the account is gone")
    print("\nEURUSD sanity check (why nobody notices until they trade yen):")
    e = pip_value("EURUSD", 100_000, 1.0850)
    print(f"  EURUSD 100,000 EUR -> pip {e.pip_size:.4f}, "
          f"${e.value_usd:,.2f}/pip -- the 4dp default is CORRECT here.")

    # ------------------------------------------------------------ 3. the carry trade
    print("\n" + "=" * 84)
    print("3. THE CARRY TRADE: NEGATIVE ON SPOT, POSITIVE IN TOTAL")
    print("=" * 84)
    spot = _synthetic_carry_pair()
    R_AUD, R_USD = 0.0475, 0.0025
    tr = total_return(spot, r_base=R_AUD, r_quote=R_USD, pair="AUDUSD")
    print(f"AUDUSD {spot.iloc[0]:.4f} -> {spot.iloc[-1]:.4f} over "
          f"{spot.index[0]:%Y-%m} to {spot.index[-1]:%Y-%m}\n")
    print(render_carry(tr, "AUDUSD", R_AUD, R_USD))

    c1y = carry_return(float(spot.iloc[-1]), R_AUD, R_USD, 365, pair="AUDUSD")
    print(f"\n1-year swap on 1,000,000 AUD at spot {spot.iloc[-1]:.4f}:")
    print(f"  carry return  {c1y.ret:+.3%}  ->  {c1y.ret * 1_000_000 * spot.iloc[-1]:+,.0f} "
          f"USD credited")
    print(f"  forward       {c1y.forward:.4f}  (points {c1y.points:+.4f} = "
          f"{c1y.pips:+,.0f} pips)")
    print("  note the OPPOSITE SIGNS: the high-yielder is at a forward DISCOUNT while its")
    print("  holder is PAID interest. Mixing those two up double-counts the whole trade.")

    print("\n" + "=" * 84)
    print("THE RULES")
    print("  * parse the pair, do not slice it; check_convention() before you trust a feed")
    print("  * is_inverted() before any cross-pair signal -- USD is BASE in USDJPY/CHF/CAD")
    print("  * pip = 0.01 on JPY quotes, 0.0001 elsewhere; get it wrong and size is 100x")
    print("  * NEVER backtest FX on spot alone -- carry IS the strategy, and here it is")
    print(f"    the difference between {_stats(tr['spot_ret'])['ann']:+.2%}/yr and "
          f"{_stats(tr['total_ret'])['ann']:+.2%}/yr on identical prices")
    print("=" * 84)
