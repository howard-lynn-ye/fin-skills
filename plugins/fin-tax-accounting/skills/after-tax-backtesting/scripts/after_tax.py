"""Attach lot matching, wash sales and section 1256 to a backtest, and report BOTH Sharpes.

WHY this exists: no backtester in this library computes an after-tax number, and the ones
that do elsewhere almost never say at what rate, in what jurisdiction, or under what lot
method -- which makes the number incomparable to anything, including itself next quarter.
So this module has a guard: `report()` REFUSES to print an after-tax result that does not
carry all three. Refusing is the feature.

THE MEASUREMENT it exists to make: the after-tax turnover penalty. Take one exposure path
and churn it at several frequencies with zero transaction cost. The gross return series is
IDENTICAL by construction -- same shares, same prices, same days -- so gross Sharpe is the
same at every turnover level and every difference in after-tax Sharpe is tax, nothing else.

And the churn is not symmetric, which is the part a mental model gets wrong:

    a round trip through a GAIN  ->  realises it, early, mostly at the short-term rate
    a round trip through a LOSS  ->  is a wash sale, so the loss is DEFERRED, not booked

Turnover therefore accelerates the bad half and defers the good half. That asymmetry is
not a rate assumption; it falls out of IRC 1091 and it is why the penalty is larger than a
back-of-envelope "you pay tax a bit sooner" estimate.

Usage:
    from after_tax import TaxAssumptions, after_tax_backtest, report, turnover_study
    a = TaxAssumptions(jurisdiction="US federal, individual", short_rate=0.37,
                       long_rate=0.20, lot_method="fifo")
    print(report(after_tax_backtest(prices, hold_days=21, assumptions=a)))

This is a MODELLING tool for backtests. It is not tax advice. Rules change; check current
law with a professional before relying on any of it.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Dual-mode sibling import: these three modules are separate SKILLS on disk and one package
# in fin_skills.tax_accounting, so both forms have to work.
try:  # inside the generated package
    from .lot_matching import is_long_term
    from .section_1256 import blended_rate, recognise_1256
    from .wash_sales import apply_wash_sales, economic_pnl
except ImportError:  # run as a script from this skill's scripts/ directory
    _SKILLS = Path(__file__).resolve().parents[2]
    for _sib in ("tax-lot-matching-and-cost-basis", "wash-sale-rules",
                 "section-1256-and-derivatives-tax"):
        _p = _SKILLS / _sib / "scripts"
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    from lot_matching import is_long_term
    from section_1256 import blended_rate, recognise_1256
    from wash_sales import apply_wash_sales, economic_pnl

TRADING_DAYS = 252
REQUIRED_ASSUMPTIONS = ("jurisdiction", "short_rate", "long_rate", "lot_method")


class MissingAssumptions(ValueError):
    """Raised when an after-tax number is asked to leave the building undressed."""


@dataclass(frozen=True)
class TaxAssumptions:
    """The three fields without which an after-tax result means nothing, plus the honest rest.

    There are no default rates here and there never will be. A rate depends on the
    jurisdiction, the entity, the bracket and the year; a default is how an after-tax figure
    becomes an unlabelled opinion.
    """
    jurisdiction: str
    short_rate: float
    long_rate: float
    lot_method: str
    wash_sales: bool = True
    loss_carryforward: bool = True
    ordinary_offset_per_year: float = 0.0     # US: 3000. Zero unless YOU set it.
    tax_paid: str = "at each year end, from outside the book"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not str(self.jurisdiction).strip():
            raise MissingAssumptions("jurisdiction must name the tax system, e.g. "
                                     "'US federal, individual, 2025 rules'")
        for name in ("short_rate", "long_rate"):
            r = getattr(self, name)
            if r is None or not 0.0 <= float(r) <= 1.0:
                raise MissingAssumptions(f"{name} must be a rate in [0, 1]; there is no default")
        if not str(self.lot_method).strip():
            raise MissingAssumptions("lot_method must be stated ('fifo', 'lifo', ...); see "
                                     "../../tax-lot-matching-and-cost-basis/SKILL.md")

    def line(self) -> str:
        return (f"jurisdiction={self.jurisdiction!r} short={self.short_rate:.2%} "
                f"long={self.long_rate:.2%} lots={self.lot_method!r} "
                f"wash_sales={self.wash_sales} carryforward={self.loss_carryforward}")


def require_assumptions(assumptions) -> TaxAssumptions:
    """The guard. Anything that is not a complete TaxAssumptions is refused, loudly."""
    if assumptions is None:
        raise MissingAssumptions(
            "after-tax results are not comparable without assumptions. Pass a "
            f"TaxAssumptions with {list(REQUIRED_ASSUMPTIONS)}.")
    if isinstance(assumptions, dict):
        missing = [k for k in REQUIRED_ASSUMPTIONS if assumptions.get(k) in (None, "")]
        if missing:
            raise MissingAssumptions(
                f"after-tax result is missing {missing}. An after-tax Sharpe without a "
                f"rate, a jurisdiction and a lot method is not comparable to anything.")
        return TaxAssumptions(**assumptions)
    if not isinstance(assumptions, TaxAssumptions):
        raise MissingAssumptions(f"expected TaxAssumptions, got {type(assumptions).__name__}")
    return assumptions


# --------------------------------------------------------------------------- netting
def net_and_tax(realised: pd.DataFrame, a: TaxAssumptions) -> pd.DataFrame:
    """Per-year capital-gains tax with character netting and loss carryforward.

    `realised` has columns short, long indexed by year. The netting order implemented is
    the US one: net within character, then cross-net, then carry the remainder forward
    WITH ITS CHARACTER. A net loss produces no refund; it waits for a gain.
    """
    st_cf = lt_cf = 0.0
    rows = []
    for year, r in realised.iterrows():
        st = float(r["short"]) + st_cf
        lt = float(r["long"]) + lt_cf
        # Cross-netting: a loss in one bucket meets a gain in the other.
        if st < 0 < lt:
            take = min(-st, lt)
            st, lt = st + take, lt - take
        elif lt < 0 < st:
            take = min(-lt, st)
            lt, st = lt + take, st - take
        taxable_st, taxable_lt = max(st, 0.0), max(lt, 0.0)
        carry_st, carry_lt = min(st, 0.0), min(lt, 0.0)
        offset = 0.0
        if a.ordinary_offset_per_year and (carry_st + carry_lt) < 0:
            offset = min(a.ordinary_offset_per_year, -(carry_st + carry_lt))
            use_st = min(offset, -carry_st)
            carry_st += use_st
            carry_lt += offset - use_st
        tax = taxable_st * a.short_rate + taxable_lt * a.long_rate
        rows.append({"year": year, "short_taxable": taxable_st, "long_taxable": taxable_lt,
                     "ordinary_offset": offset, "tax": tax,
                     "carry_short": carry_st, "carry_long": carry_lt})
        if a.loss_carryforward:
            st_cf, lt_cf = carry_st, carry_lt
        else:
            st_cf = lt_cf = 0.0
    return pd.DataFrame(rows, columns=["year", "short_taxable", "long_taxable",
                                       "ordinary_offset", "tax", "carry_short",
                                       "carry_long"]).set_index("year")


def realised_by_year(disposals: pd.DataFrame) -> pd.DataFrame:
    """Split allowed (post-wash-sale) gains into the short and long buckets, by year."""
    if disposals.empty:
        return pd.DataFrame(columns=["short", "long"], dtype=float)
    d = disposals.copy()
    d["year"] = pd.to_datetime(d["sell_date"]).dt.year
    d["term"] = [("long" if is_long_term(o, s) else "short")
                 for o, s in zip(d["open_date"], d["sell_date"])]
    out = d.pivot_table(index="year", columns="term", values="allowed", aggfunc="sum",
                        fill_value=0.0)
    for col in ("short", "long"):
        if col not in out.columns:
            out[col] = 0.0
    out = out.loc[:, ["short", "long"]]
    out.columns.name = None
    return out


# --------------------------------------------------------------------------- the engine
def sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe at a zero risk-free rate. Stated, not assumed silently."""
    sd = float(returns.std(ddof=1))
    return float(returns.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def churn_blotter(prices: pd.Series, hold_days: int, shares: float = 1000.0) -> pd.DataFrame:
    """Buy `shares`, then every `hold_days` sell the lot and buy it straight back.

    Exposure is constant at `shares` throughout, so the GROSS return path does not depend
    on hold_days at all. Only the realisation pattern does. hold_days <= 0 means never
    churn: buy and hold.
    """
    d = prices.index
    rows = [{"date": d[0], "side": "B", "qty": shares, "price": float(prices.iloc[0])}]
    if hold_days and hold_days > 0:
        for i in range(hold_days, len(prices), hold_days):
            p = float(prices.iloc[i])
            rows.append({"date": d[i], "side": "S", "qty": shares, "price": p})
            rows.append({"date": d[i], "side": "B", "qty": shares, "price": p})
    return pd.DataFrame(rows)


def after_tax_backtest(prices: pd.Series, hold_days: int, assumptions,
                       shares: float = 1000.0) -> dict:
    """Gross and after-tax equity for one churn frequency. Returns a result dict."""
    a = require_assumptions(assumptions)
    px = pd.Series(prices).astype(float)
    blotter = churn_blotter(px, hold_days, shares)
    res = apply_wash_sales(blotter, rule=a.lot_method if a.lot_method in ("fifo", "lifo")
                           else "fifo")
    disposals = res["disposals"].copy()
    if not a.wash_sales:                      # the comparison a naive engine actually makes
        disposals["allowed"] = disposals["gain"]
    realised = realised_by_year(disposals)
    tax = net_and_tax(realised, a)

    gross_equity = px * shares
    paid = pd.Series(0.0, index=px.index)
    for year, row in tax.iterrows():
        days = px.index[px.index.year == year]
        if len(days):
            paid.loc[days[-1]] += float(row["tax"])
    cum_tax = paid.cumsum()
    net_equity = gross_equity - cum_tax
    # A tax bill larger than the book is a real outcome (see the wash-sale ratchet), but a
    # Sharpe computed across a zero crossing is meaningless. Say so instead of printing it.
    solvent = bool((net_equity > 0).all())

    gross_ret = gross_equity.pct_change().dropna()
    net_ret = net_equity.pct_change().dropna() if solvent else pd.Series(dtype=float)
    return {"assumptions": a, "hold_days": hold_days, "blotter": blotter,
            "wash": res, "realised": realised, "tax": tax,
            "gross_equity": gross_equity, "net_equity": net_equity,
            "gross_returns": gross_ret, "net_returns": net_ret,
            "gross_sharpe": sharpe(gross_ret),
            "after_tax_sharpe": sharpe(net_ret) if solvent else float("nan"),
            "solvent": solvent, "total_tax": float(tax["tax"].sum()),
            "disallowed_loss": float(res["disposals"]["disallowed"].sum()),
            "pretax_pnl": economic_pnl(blotter, float(px.iloc[-1])),
            "round_trips": int((blotter["side"] == "S").sum())}


def report(result: dict) -> str:
    """Render an after-tax result. Refuses if the assumptions are not attached."""
    a = require_assumptions(result.get("assumptions"))
    gross = result["gross_sharpe"]
    net = result["after_tax_sharpe"]
    lines = [
        "AFTER-TAX RESULT -- hypothetical, and only comparable to another result computed",
        "under the SAME line below.",
        f"  assumptions: {a.line()}",
        f"  tax paid:    {a.tax_paid}",
        f"  round trips: {result['round_trips']}   pre-tax P&L {result['pretax_pnl']:,.2f}",
        f"  tax paid     {result['total_tax']:,.2f}   wash-sale loss deferred "
        f"{result['disallowed_loss']:,.2f}",
        (f"  SHARPE       gross {gross:.3f}   after tax {net:.3f}   "
         f"retained {net / gross:.1%}") if gross and result.get("solvent", True)
        else "  SHARPE       gross "
             f"{gross:.3f}   after tax n/a: the tax bill exceeded the book",
    ]
    for n in a.notes:
        lines.append(f"  note: {n}")
    return "\n".join(lines)


def turnover_study(prices: pd.Series, holds, assumptions, shares: float = 1000.0
                   ) -> pd.DataFrame:
    """The after-tax turnover penalty: identical gross path, several churn frequencies."""
    a = require_assumptions(assumptions)
    rows = []
    for h in holds:
        r = after_tax_backtest(prices, h, a, shares)
        rows.append({
            "hold_days": h if h and h > 0 else np.inf,
            "round_trips_per_year": (TRADING_DAYS / h) if h and h > 0 else 0.0,
            "gross_sharpe": r["gross_sharpe"], "after_tax_sharpe": r["after_tax_sharpe"],
            "retained": r["after_tax_sharpe"] / r["gross_sharpe"] if r["gross_sharpe"] else
            float("nan"),
            "tax_paid": r["total_tax"], "loss_deferred": r["disallowed_loss"],
            "pretax_pnl": r["pretax_pnl"]})
    return pd.DataFrame(rows).set_index("hold_days")


# --------------------------------------------------------------------------- demo
def demo_prices(seed: int = 20260909, years: int = 4, start: str = "2022-01-03",
                s0: float = 100.0, mu: float = 0.12, sigma: float = 0.20) -> pd.Series:
    rng = np.random.default_rng(seed)
    n = years * TRADING_DAYS
    dates = pd.bdate_range(start, periods=n)
    step = rng.normal(mu / 252.0 - 0.5 * sigma**2 / 252.0, sigma / np.sqrt(252.0), n)
    return pd.Series(s0 * np.exp(np.cumsum(step)), index=dates)


def main() -> None:
    px = demo_prices()
    a = TaxAssumptions(jurisdiction="US federal, individual, 2025 rules",
                       short_rate=0.37, long_rate=0.20, lot_method="fifo",
                       notes=("no state tax", "no NIIT", "zero transaction cost, so every "
                              "difference below is tax"))
    print("After-tax backtesting -- the turnover penalty that is not a commission")
    print(f"  one seeded 4-year path, 1,000 shares held CONSTANTLY; the only difference "
          f"between rows is how often the position is sold and bought straight back")
    print(f"  {a.line()}")

    try:
        after_tax_backtest(px, 21, assumptions=None)
    except MissingAssumptions as exc:
        print(f"  guard: {exc}")
    try:
        after_tax_backtest(px, 21, assumptions={"jurisdiction": "US", "short_rate": 0.37})
    except MissingAssumptions as exc:
        print(f"  guard: {exc}")

    holds = (1, 5, 21, 63, 252, 0)
    table = turnover_study(px, holds, a)
    print()
    print("  hold(d)  trips/yr  gross Sh  after-tax Sh  retained    tax paid  loss deferred")
    for h, r in table.iterrows():
        label = "hold" if np.isinf(h) else f"{int(h)}"
        print(f"  {label:>7}{r['round_trips_per_year']:>10.1f}{r['gross_sharpe']:>10.3f}"
              f"{r['after_tax_sharpe']:>14.3f}{r['retained']:>10.1%}"
              f"{r['tax_paid']:>12,.0f}{r['loss_deferred']:>15,.0f}")

    g = table["gross_sharpe"].round(12).nunique()
    print()
    print(f"  gross Sharpe is IDENTICAL across every row (distinct values: {g}), and the "
          f"pre-tax P&L is {table['pretax_pnl'].iloc[0]:,.2f} on all of them")
    worst, best = table["after_tax_sharpe"].min(), table["after_tax_sharpe"].max()
    print(f"  after-tax Sharpe {worst:.3f} to {best:.3f}: churning daily costs "
          f"{100 * (best - worst) / best:.1f}% of the after-tax Sharpe of buying and holding")
    fast = table.loc[[1.0, 5.0, 21.0], "tax_paid"]
    print(f"  and it SATURATES: monthly, weekly and daily churn pay "
          f"{fast.min():,.0f}-{fast.max():,.0f} tax, because once the basis is reset at every "
          f"new high there is nothing left to accelerate")

    r_naive = after_tax_backtest(px, 1, TaxAssumptions(
        jurisdiction=a.jurisdiction, short_rate=a.short_rate, long_rate=a.long_rate,
        lot_method="fifo", wash_sales=False))
    r_true = after_tax_backtest(px, 1, a)
    print(f"  ignoring wash sales at daily churn understates tax by "
          f"{r_true['total_tax'] - r_naive['total_tax']:,.2f} "
          f"({r_naive['after_tax_sharpe']:.3f} vs {r_true['after_tax_sharpe']:.3f} Sharpe)")

    print()
    print(report(r_true))

    print()
    print("  the SAME buy-and-hold exposure held as a section 1256 contract instead:")
    positions = pd.DataFrame([{"open_date": px.index[0], "close_date": pd.NaT, "qty": 1000.0,
                               "open_price": float(px.iloc[0]), "close_price": np.nan}])
    marks = pd.DataFrame([{"position": 0, "year": y,
                           "price": float(px[px.index.year == y].iloc[-1])}
                          for y in sorted(set(px.index.year))[:-1]])
    s1256 = recognise_1256(positions, marks, multiplier=1.0)
    t1256 = net_and_tax(s1256.rename(columns={"short_term": "short", "long_term": "long"})
                        .loc[:, ["short", "long"]], a)
    for y, r in s1256.iterrows():
        print(f"    {y}   recognised {r['recognised']:>12,.2f}   tax "
              f"{t1256.loc[y, 'tax']:>10,.2f}   vs 0.00 recognised and 0.00 tax as stock")
    print(f"    taxable income on a position nobody sold swings "
          f"{s1256['recognised'].min():,.2f} to {s1256['recognised'].max():,.2f} and costs "
          f"{t1256['tax'].sum():,.2f} at the "
          f"{blended_rate(a.short_rate, a.long_rate):.1%} blended rate;")
    print(f"    the identical exposure held as stock recognises 0.00 every year and pays "
          f"{table.loc[np.inf, 'tax_paid']:,.2f}")

    print()
    print("RULE: an after-tax result without a rate, a jurisdiction and a lot method is not "
          "comparable to anything, so state all three or report pre-tax only.")


if __name__ == "__main__":
    main()
