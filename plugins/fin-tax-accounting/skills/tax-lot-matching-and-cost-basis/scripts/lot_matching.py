"""One blotter, four lot-matching rules, four different reported P&Ls.

WHY this exists: a backtest reports the P&L of a POSITION. A tax return reports the P&L of
a set of LOTS, and which lot you sold is a choice. The same trades, matched four ways,
produce four different realised gains and four different short-term/long-term splits, so
"the strategy made $X" is not a statement about the tax return until you say how the lots
were matched.

What the choice can and cannot do -- this is the whole file in two lines:

    realised(rule) + unrealised(rule)  is IDENTICAL for every rule.
    realised(rule)                     is not.

Lot selection moves P&L between this year and a later year, and between the short-term and
the long-term bucket. It does not create or destroy a dollar of it. Anyone who tells you a
lot method "made money" is quoting the first term and hiding the second.

THE RULES, and what they are under US law (Publication 550 (2025), Chapter 4) --

  fifo   the DEFAULT. "If you buy and sell securities at various times in varying
         quantities and you cannot adequately identify the shares you sell, the basis of
         the securities you sell is the basis of the securities you acquired first."
  lifo   NOT a statutory method for securities. It is a broker preset that produces a
         specific identification, and it is only defensible if the identification was
         actually made at or before the sale and confirmed in writing.
  hifo   same status as lifo. Highest cost first. Minimises current-year gain.
  lofo   same status. Lowest cost first. Included because it is the mirror image and it
         brackets the range a P&L report can legally land in.

  average  Publication 550 permits it only for shares in a mutual fund (or other regulated
           investment company) and for DRIP shares. `average_basis()` REFUSES on ordinary
           stock rather than returning a number that is wrong on a tax return.

Usage:
    from lot_matching import make_blotter, compare_rules, match_lots
    print(compare_rules(make_blotter(), mark=..., rules=("fifo", "lifo", "hifo", "lofo")))

This is a MODELLING tool for backtests. It is not tax advice, and it is not a tax
return. Rules change; check current law with a professional before relying on any of it.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Publication 550 (2025) Chapter 4: FIFO is what happens when nothing is identified.
DEFAULT_RULE = "fifo"
RULES = ("fifo", "lifo", "hifo", "lofo")

# Publication 550 (2025), "Average Basis": shares in a mutual fund (or other regulated
# investment company), or DRIP shares. Ordinary stock is not on the list.
AVERAGE_BASIS_ELIGIBLE = ("mutual_fund", "ric", "drip")

BLOTTER_COLUMNS = ("date", "side", "qty", "price")


# --------------------------------------------------------------------------- input
def normalise_blotter(blotter: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort a single-symbol blotter. Columns: date, side ('B'/'S'), qty, price."""
    missing = [c for c in BLOTTER_COLUMNS if c not in blotter.columns]
    if missing:
        raise ValueError(f"blotter is missing column(s) {missing}; needs {list(BLOTTER_COLUMNS)}")
    out = blotter.loc[:, list(BLOTTER_COLUMNS)].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["side"] = out["side"].astype(str).str.upper().str[0]
    if not out["side"].isin(("B", "S")).all():
        raise ValueError("side must be 'B' (buy) or 'S' (sell) on every row")
    out["qty"] = out["qty"].astype(float)
    out["price"] = out["price"].astype(float)
    if (out["qty"] <= 0).any():
        raise ValueError("qty must be positive on every row; the side column carries the sign")
    if (out["price"] < 0).any():
        raise ValueError("price must be non-negative")
    if out[["date", "qty", "price"]].isna().any().any():
        raise ValueError("blotter contains NaN")
    # Stable sort: same-day buys keep their input order, which is what 'first bought' means.
    return out.sort_values("date", kind="stable").reset_index(drop=True)


def is_long_term(open_date: pd.Timestamp, close_date: pd.Timestamp) -> bool:
    """Long-term = held MORE than one year (IRC 1222). One calendar year, not 365 days.

    A lot bought 2024-02-29 and sold 2025-02-28 is short-term; sold 2025-03-01 it is
    long-term. Counting 365 days instead gets leap years and month-ends wrong, which is
    exactly where a year-end harvesting rule lives.
    """
    return pd.Timestamp(close_date) > pd.Timestamp(open_date) + pd.DateOffset(years=1)


# --------------------------------------------------------------------------- matching
def _pick(lots: list[dict], rule: str) -> int:
    """Index of the lot this rule sells next. `lots` is in acquisition order."""
    if not lots:
        raise ValueError("sell exceeds inventory: this blotter goes short, which needs "
                         "short-sale rules (IRC 1233), not lot matching")
    if rule == "fifo":
        return 0
    if rule == "lifo":
        return len(lots) - 1
    if rule == "hifo":
        return max(range(len(lots)), key=lambda i: (lots[i]["price"], -i))
    if rule == "lofo":
        return min(range(len(lots)), key=lambda i: (lots[i]["price"], i))
    raise ValueError(f"unknown rule {rule!r}; expected one of {list(RULES)}")


def match_lots(blotter: pd.DataFrame, rule: str = DEFAULT_RULE) -> dict:
    """Match every sale against open lots under `rule`.

    Returns {'disposals': DataFrame, 'open_lots': DataFrame, 'rule': str}. Each disposal
    row is one (sale, lot) pair: sell_date, open_date, qty, proceeds, basis, gain, term.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {list(RULES)}")
    df = normalise_blotter(blotter)
    lots: list[dict] = []
    rows: list[dict] = []
    for t in df.itertuples(index=False):
        if t.side == "B":
            lots.append({"date": t.date, "qty": float(t.qty), "price": float(t.price)})
            continue
        left = float(t.qty)
        while left > 1e-12:
            i = _pick(lots, rule)
            lot = lots[i]
            take = min(left, lot["qty"])
            rows.append({
                "sell_date": t.date, "open_date": lot["date"], "qty": take,
                "proceeds": take * t.price, "basis": take * lot["price"],
                "gain": take * (t.price - lot["price"]),
                "term": "long" if is_long_term(lot["date"], t.date) else "short",
            })
            lot["qty"] -= take
            left -= take
            if lot["qty"] <= 1e-12:
                lots.pop(i)
    disposals = pd.DataFrame(rows, columns=["sell_date", "open_date", "qty", "proceeds",
                                            "basis", "gain", "term"])
    open_lots = pd.DataFrame(lots, columns=["date", "qty", "price"])
    return {"disposals": disposals, "open_lots": open_lots, "rule": rule}


def realised_gain(result: dict) -> float:
    return float(result["disposals"]["gain"].sum())


def term_split(result: dict) -> dict[str, float]:
    """Realised gain split into the short-term and long-term buckets."""
    d = result["disposals"]
    if d.empty:
        return {"short": 0.0, "long": 0.0}
    g = d.groupby("term")["gain"].sum()
    return {"short": float(g.get("short", 0.0)), "long": float(g.get("long", 0.0))}


def unrealised(result: dict, mark: float) -> float:
    """Mark-to-market gain still open in the remaining lots."""
    lots = result["open_lots"]
    if lots.empty:
        return 0.0
    return float((lots["qty"] * (float(mark) - lots["price"])).sum())


def compare_rules(blotter: pd.DataFrame, mark: float,
                  rules: Sequence[str] = RULES) -> pd.DataFrame:
    """One row per rule: realised, short/long split, unrealised, and their invariant sum."""
    rows = []
    for rule in rules:
        res = match_lots(blotter, rule)
        split = term_split(res)
        rlz = realised_gain(res)
        unr = unrealised(res, mark)
        rows.append({"rule": rule, "realised": rlz, "short_term": split["short"],
                     "long_term": split["long"], "unrealised": unr, "total": rlz + unr,
                     "n_disposals": int(len(res["disposals"]))})
    return pd.DataFrame(rows).set_index("rule")


def tax_due(split: dict[str, float], short_rate: float, long_rate: float) -> float:
    """Tax on a realised short/long split at rates YOU state. There is no default rate.

    Rates are inputs, not knowledge: they depend on jurisdiction, entity, bracket and year,
    and a hard-coded one is the fastest way to make an after-tax number incomparable. See
    ../../../after-tax-backtesting/SKILL.md for the guard that enforces this at the report.
    """
    for name, r in (("short_rate", short_rate), ("long_rate", long_rate)):
        if not 0.0 <= float(r) <= 1.0:
            raise ValueError(f"{name}={r} is not a rate in [0, 1]")
    return float(split["short"]) * float(short_rate) + float(split["long"]) * float(long_rate)


def spread_fraction(table: pd.DataFrame) -> dict[str, float]:
    """How far apart the rules land, as a fraction of the strategy's own total P&L."""
    total = float(table["total"].iloc[0])
    spread = float(table["realised"].max() - table["realised"].min())
    return {"spread": spread, "total_pnl": total,
            "fraction_of_pnl": spread / total if total else float("nan"),
            "long_term_spread": float(table["long_term"].max() - table["long_term"].min())}


# --------------------------------------------------------------------------- average basis
def average_basis(blotter: pd.DataFrame, share_class: str = "stock") -> dict:
    """Average-basis disposals, but ONLY for share classes Publication 550 allows.

    Pub 550 (2025), "Average Basis": the account must hold identical shares acquired at
    different times and prices, and they must be shares in a mutual fund (or other
    regulated investment company) or DRIP shares. Passing share_class='stock' raises.

    Holding period still runs FIFO -- Pub 550: "To determine your holding period, the
    shares disposed of are considered to be those acquired first." So average basis
    changes the amount of the gain and not its short/long character.
    """
    if share_class not in AVERAGE_BASIS_ELIGIBLE:
        raise ValueError(
            f"average basis is not available for share_class={share_class!r}. IRS "
            f"Publication 550 (2025), 'Average Basis', limits it to shares in a mutual fund "
            f"(or other regulated investment company) and to DRIP shares. For ordinary "
            f"stock the default is FIFO and anything else is a specific identification. "
            f"Eligible values: {list(AVERAGE_BASIS_ELIGIBLE)}")
    df = normalise_blotter(blotter)
    fifo_terms = match_lots(df, "fifo")["disposals"]
    qty = 0.0
    cost = 0.0
    rows = []
    for t in df.itertuples(index=False):
        if t.side == "B":
            qty += float(t.qty)
            cost += float(t.qty) * float(t.price)
            continue
        if float(t.qty) - qty > 1e-9:
            raise ValueError("sell exceeds inventory")
        avg = cost / qty if qty else 0.0
        rows.append({"sell_date": t.date, "qty": float(t.qty),
                     "avg_basis_per_share": avg,
                     "proceeds": float(t.qty) * float(t.price),
                     "basis": float(t.qty) * avg,
                     "gain": float(t.qty) * (float(t.price) - avg)})
        qty -= float(t.qty)
        cost -= float(t.qty) * avg
    disposals = pd.DataFrame(rows, columns=["sell_date", "qty", "avg_basis_per_share",
                                            "proceeds", "basis", "gain"])
    return {"disposals": disposals, "remaining_qty": qty, "remaining_cost": cost,
            "holding_period_source": "FIFO (Pub 550)", "fifo_terms": fifo_terms}


# --------------------------------------------------------------------------- demo blotter
def make_blotter(seed: int = 20260909, n_days: int = 1008, start: str = "2022-01-03",
                 s0: float = 100.0, mu: float = 0.08, sigma: float = 0.28) -> pd.DataFrame:
    """A seeded four-year accumulate-and-trim blotter in one name.

    Deliberately ordinary: buy the dips, trim the rips, never go short. That is enough to
    build a deep, price-dispersed lot stack, which is the only condition the four rules
    need in order to disagree.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    step = rng.normal(mu / 252.0 - 0.5 * sigma**2 / 252.0, sigma / np.sqrt(252.0), n_days)
    px = pd.Series(s0 * np.exp(np.cumsum(step)), index=dates)
    ret5 = px.pct_change(5)
    rows = [{"date": dates[0], "side": "B", "qty": 200.0, "price": float(px.iloc[0])}]
    inventory = 200.0
    for i in range(10, n_days, 7):
        d, p, r = dates[i], float(px.iloc[i]), float(ret5.iloc[i])
        if r < -0.02:
            q = 100.0 if r < -0.05 else 50.0
            rows.append({"date": d, "side": "B", "qty": q, "price": p})
            inventory += q
        elif r > 0.03 and inventory >= 150.0:
            q = 100.0 if r > 0.06 else 50.0
            rows.append({"date": d, "side": "S", "qty": q, "price": p})
            inventory -= q
    out = pd.DataFrame(rows)
    out.attrs["final_price"] = float(px.iloc[-1])
    out.attrs["prices"] = px
    return out


def _fmt(table: pd.DataFrame) -> str:
    lines = ["  rule    realised   short-term    long-term   unrealised        total"]
    for rule, row in table.iterrows():
        lines.append(f"  {rule:<6}{row['realised']:11,.2f}{row['short_term']:13,.2f}"
                     f"{row['long_term']:13,.2f}{row['unrealised']:13,.2f}"
                     f"{row['total']:13,.2f}")
    return "\n".join(lines)


def main() -> None:
    blotter = make_blotter()
    mark = blotter.attrs["final_price"]
    n_buy = int((blotter["side"] == "B").sum())
    n_sell = int((blotter["side"] == "S").sum())
    print("Tax lot matching -- one blotter, four rules")
    print(f"  seeded blotter: {len(blotter)} trades ({n_buy} buys, {n_sell} sells), "
          f"{blotter['date'].iloc[0].date()} to {blotter['date'].iloc[-1].date()}, "
          f"mark {mark:.4f}")

    table = compare_rules(blotter, mark)
    print()
    print(_fmt(table))

    s = spread_fraction(table)
    print()
    print(f"  same trades, realised gain spread {s['spread']:,.2f} = "
          f"{100 * s['fraction_of_pnl']:.1f}% of the strategy's total P&L {s['total_pnl']:,.2f}")
    print(f"  long-term bucket spread {s['long_term_spread']:,.2f}")
    inv = table["total"].round(6).nunique()
    print(f"  invariant check: distinct values of realised+unrealised across rules = {inv}")

    short_rate, long_rate = 0.37, 0.20
    print()
    print(f"  ASSUMED rates (not a default, an input): short {short_rate:.0%}, "
          f"long {long_rate:.0%}. Tax on the realised column:")
    bill = {r: tax_due({"short": row["short_term"], "long": row["long_term"]},
                       short_rate, long_rate) for r, row in table.iterrows()}
    for r, amt in bill.items():
        print(f"    {r:<6}{amt:11,.2f}")
    print(f"  same trades, tax bill spread {max(bill.values()) - min(bill.values()):,.2f}")

    print()
    print("  Publication 550 (2025) status of each rule:")
    print("    fifo  DEFAULT when the shares are not adequately identified")
    print("    lifo  broker preset for a specific identification, not a statutory method")
    print("    hifo  broker preset for a specific identification, not a statutory method")
    print("    lofo  broker preset for a specific identification, not a statutory method")
    try:
        average_basis(blotter, share_class="stock")
    except ValueError as exc:
        print(f"    average -> refused on ordinary stock: {str(exc).split('.')[0]}.")
    fund = average_basis(blotter, share_class="mutual_fund")
    print(f"    average allowed for a mutual fund: realised "
          f"{fund['disposals']['gain'].sum():,.2f}")

    print()
    print("RULE: lot choice moves P&L between years and between the short and long "
          "buckets; it never changes the total, so report the method with the number.")


if __name__ == "__main__":
    main()
