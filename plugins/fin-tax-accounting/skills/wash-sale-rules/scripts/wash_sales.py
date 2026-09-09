"""Wash sales: the loss a systematic strategy books and the tax code defers.

WHY this exists: a monthly-rebalanced strategy trades the same name every 21 business days.
The wash sale window is 30 days BEFORE and 30 days AFTER a sale -- 61 days in total -- so
every single one of those sales has a purchase inside its window on BOTH sides. A backtest
that books each realised loss when it happens is not slightly optimistic; it is claiming
deductions the code moves into the basis of shares still held.

THE MECHANISM, and the two halves people get backwards:

  1. The loss is DISALLOWED, and
  2. it is ADDED TO THE BASIS of the replacement shares, whose holding period also picks
     up the holding period of the shares sold.

(2) is why a wash sale is a DEFERRAL and not a penalty: the deduction comes back when the
replacement is finally sold outside a window. There is exactly one common case where (2)
does not happen and the loss is gone for good -- the replacement bought inside an IRA or
Roth IRA -- and this module models that separately because the arithmetic differs.

Usage:
    from wash_sales import apply_wash_sales, make_rebalance_blotter, summarise
    res = apply_wash_sales(make_rebalance_blotter())
    print(summarise(res))

This is a MODELLING tool for backtests. It is not tax advice. Rules change; check current
law with a professional before relying on any of it.

SOURCES (read 2026-09-09):
  IRC 1091(a): loss disallowed if, "within a period beginning 30 days before the date of
    such sale or disposition and ending 30 days after such date", substantially identical
    stock or securities are acquired.
  IRC 1091(d): the replacement's basis is the basis of the stock sold, adjusted by the
    difference between the replacement's price and the sale price.
  IRC 1223(3): the replacement's holding period includes the holding period of the shares
    whose loss was disallowed.
  Pub 550 (2025) p. 86-87: the four acquisition routes, including "(4) Acquire substantially
    identical stock for your individual retirement arrangement (IRA) or Roth IRA"; "add the
    disallowed loss to the cost of the new stock or securities (EXCEPT IN (4) ABOVE)"; and
    the matching rule -- "Match the shares bought in the same order that you bought them,
    beginning with the first shares bought."
  Rev. Rul. 2008-5, I.R.B. 2008-3 (2008-01-22): "The loss on the Sale of stock is disallowed
    under section 1091. A's basis in the individual retirement account or Roth IRA is not
    increased by virtue of section 1091(d)."
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_DAYS = 30            # IRC 1091(a): 30 before AND 30 after
WINDOW_TOTAL_DAYS = 61      # the sale day plus 30 on each side
RETIREMENT_ACCOUNTS = ("ira", "roth_ira", "roth")
BLOTTER_COLUMNS = ("date", "side", "qty", "price")


def _norm(blotter: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in BLOTTER_COLUMNS if c not in blotter.columns]
    if missing:
        raise ValueError(f"blotter is missing column(s) {missing}")
    out = blotter.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["side"] = out["side"].astype(str).str.upper().str[0]
    if not out["side"].isin(("B", "S")).all():
        raise ValueError("side must be 'B' or 'S' on every row")
    out["qty"] = out["qty"].astype(float)
    out["price"] = out["price"].astype(float)
    if (out["qty"] <= 0).any():
        raise ValueError("qty must be positive; the side column carries the sign")
    if "account" not in out.columns:
        out["account"] = "taxable"
    out["account"] = out["account"].astype(str).str.lower()
    return out.sort_values("date", kind="stable").reset_index(drop=True)


def in_window(sale_date, buy_date, days: int = WINDOW_DAYS) -> bool:
    """IRC 1091(a): 30 days before OR after, inclusive. 61 calendar days in total."""
    return abs((pd.Timestamp(buy_date) - pd.Timestamp(sale_date)).days) <= days


def apply_wash_sales(blotter: pd.DataFrame, rule: str = "fifo") -> dict:
    """Match lots, disallow wash-sale losses, and move them into the replacement's basis.

    Returns a dict with:
      disposals   one row per (sale, lot) pair: gain, disallowed, allowed, term,
                  replacements, permanent
      open_lots   remaining lots with ADJUSTED basis and carried-over holding-period start
      orphaned    disallowed loss that had nowhere to go because the replacement shares were
                  themselves already sold before the loss sale (rare; flagged, not hidden)
    """
    if rule not in ("fifo", "lifo"):
        raise ValueError("rule must be 'fifo' or 'lifo'; see ../../tax-lot-matching-and-cost-"
                         "basis/scripts/lot_matching.py for the full set and their status")
    df = _norm(blotter)

    # EVERY buy is a replacement candidate up front, including buys AFTER the loss sale.
    # Building this list incrementally is the bug that turns a +/-30-day rule into a
    # -30-day rule and silently reports the wrong disallowed loss: the classic wash sale --
    # sell today, buy back tomorrow -- lives entirely in the forward half of the window.
    purchases: list[dict] = [
        {"buy_id": i, "date": t.date, "capacity": float(t.qty), "account": t.account}
        for i, t in enumerate(df.itertuples(index=False)) if t.side == "B"]

    lots: list[dict] = []       # open lots, in acquisition order
    pending: dict[int, list[tuple]] = {}   # basis bumps for lots that do not exist yet
    rows: list[dict] = []
    orphaned = 0.0

    for buy_id, t in enumerate(df.itertuples(index=False)):
        if t.side == "B":
            lots.append({"buy_id": buy_id, "date": t.date, "qty": float(t.qty),
                         "basis_ps": float(t.price), "account": t.account,
                         "hp_start": t.date, "n_adj": 0})
            for shares, loss_ps, hp, depth in pending.pop(buy_id, []):
                orphaned += loss_ps * (
                    shares - _rebase(buy_id, shares, loss_ps, hp, lots, depth))
            continue

        left = float(t.qty)
        while left > 1e-9:
            if not lots:
                raise ValueError("sell exceeds inventory: short sales are IRC 1233, not this")
            i = 0 if rule == "fifo" else len(lots) - 1
            lot = lots[i]
            take = min(left, lot["qty"])
            gain = take * (float(t.price) - lot["basis_ps"])
            row = {"sell_date": t.date, "open_date": lot["hp_start"], "qty": take,
                   "proceeds": take * float(t.price), "basis": take * lot["basis_ps"],
                   "gain": gain, "disallowed": 0.0, "replacements": "", "permanent": False,
                   "carried_in": int(lot["n_adj"])}
            lot["qty"] -= take
            left -= take
            if lot["qty"] <= 1e-9:
                lots.pop(i)

            if gain < -1e-9:
                dis, tags, perm, orph = _disallow(gain, take, t.date, lot["buy_id"],
                                                  lot["hp_start"], purchases, lots,
                                                  buy_id, pending,
                                                  int(lot["n_adj"]) + 1)
                row["disallowed"] = dis
                row["replacements"] = ",".join(tags)
                row["permanent"] = perm
                orphaned += orph
            rows.append(row)

    disposals = pd.DataFrame(rows, columns=["sell_date", "open_date", "qty", "proceeds",
                                            "basis", "gain", "disallowed", "replacements",
                                            "permanent", "carried_in"])
    if not disposals.empty:
        disposals["allowed"] = disposals["gain"] - disposals["disallowed"]
    else:
        disposals["allowed"] = pd.Series(dtype=float)
    open_lots = pd.DataFrame(lots, columns=["buy_id", "date", "qty", "basis_ps", "account",
                                            "hp_start", "n_adj"])
    return {"disposals": disposals, "open_lots": open_lots, "orphaned_disallowed": orphaned,
            "rule": rule}


def _disallow(gain: float, qty: float, sell_date, sold_buy_id: int, sold_hp_start,
              purchases: list[dict], lots: list[dict], now_id: int,
              pending: dict[int, list[tuple]], depth: int):
    """Find replacements in the +/-30 day window, disallow pro rata, re-base the survivors.

    Pub 550 (2025) p. 87: "Match the shares bought in the same order that you bought them,
    beginning with the first shares bought." Purchases are already in acquisition order.
    A replacement bought AFTER this sale has no lot yet, so its bump is queued in `pending`
    and applied the moment the lot is created.
    """
    need = qty
    disallowed = 0.0
    orphan = 0.0
    tags: list[str] = []
    permanent = False
    loss_ps = -gain / qty                      # positive dollars of loss per share
    for p in purchases:
        if need <= 1e-9:
            break
        if p["buy_id"] == sold_buy_id or p["capacity"] <= 1e-9:
            continue
        if not in_window(sell_date, p["date"]):
            continue
        m = min(need, p["capacity"])
        p["capacity"] -= m
        need -= m
        disallowed += -loss_ps * m             # negative: it is a loss being taken away
        tags.append(f"{pd.Timestamp(p['date']).date()}x{m:g}")
        if p["account"] in RETIREMENT_ACCOUNTS:
            # Rev. Rul. 2008-5 / Pub 550 "except in (4) above": no basis step-up. Gone.
            permanent = True
            continue
        if p["buy_id"] > now_id:                # the replacement has not been bought yet
            pending.setdefault(p["buy_id"], []).append(
                (m, loss_ps, sold_hp_start, depth))
            continue
        moved = _rebase(p["buy_id"], m, loss_ps, sold_hp_start, lots, depth)
        orphan += loss_ps * (m - moved)
    return disallowed, tags, permanent, orphan


def _rebase(buy_id: int, shares: float, loss_ps: float, sold_hp_start, lots: list[dict],
            depth: int = 1) -> float:
    """Add loss_ps per share to `shares` of the still-open lot `buy_id`; carry the holding
    period. Splits the lot when only part of it is a replacement. Returns shares re-based."""
    moved = 0.0
    i = 0
    while i < len(lots) and shares - moved > 1e-9:
        lot = lots[i]
        if lot["buy_id"] != buy_id:
            i += 1
            continue
        take = min(shares - moved, lot["qty"])
        if take < lot["qty"] - 1e-9:                       # split: only `take` is replacement
            rest = dict(lot)
            rest["qty"] = lot["qty"] - take
            lot["qty"] = take
            lots.insert(i + 1, rest)
        lot["basis_ps"] += loss_ps
        lot["n_adj"] = max(int(lot["n_adj"]), int(depth))
        # IRC 1223(3): the replacement inherits the older holding period.
        lot["hp_start"] = min(pd.Timestamp(lot["hp_start"]), pd.Timestamp(sold_hp_start))
        moved += take
        i += 1
    return moved


# --------------------------------------------------------------------------- reporting
def summarise(res: dict, mark: float | None = None) -> dict:
    """Headline numbers: what was lost, what was booked, what was deferred, what is gone."""
    d = res["disposals"]
    losses = d.loc[d["gain"] < 0]
    perm = float(losses.loc[losses["permanent"], "disallowed"].sum())
    out = {
        "n_disposals": int(len(d)),
        "n_loss_sales": int(len(losses)),
        "n_wash_sales": int((d["disallowed"] < -1e-9).sum()),
        "gross_realised": float(d["gain"].sum()),
        "gross_loss": float(losses["gain"].sum()),
        "disallowed_loss": float(d["disallowed"].sum()),
        "permanent_loss": perm,
        "reported_realised": float(d["allowed"].sum()),
        "orphaned_disallowed": float(res["orphaned_disallowed"]),
        # A chain: a lot whose basis a previous wash sale already raised, sold again at a
        # loss that is itself disallowed. This is where "just book the loss" fails silently.
        "n_chained": int(((d["carried_in"] > 0) & (d["disallowed"] < -1e-9)).sum()),
        "max_chain_depth": int(d["carried_in"].max()) if len(d) else 0,
    }
    gl = out["gross_loss"]
    out["fraction_of_loss_deferred"] = out["disallowed_loss"] / gl if gl else 0.0
    if mark is not None:
        lots = res["open_lots"]
        out["unrealised"] = 0.0 if lots.empty else float(
            (lots["qty"] * (float(mark) - lots["basis_ps"])).sum())
        out["reported_total"] = out["reported_realised"] + out["unrealised"]
    return out


def economic_pnl(blotter: pd.DataFrame, mark: float) -> float:
    """Cash out minus cash in, plus the open position at `mark`. No tax rules in it at all."""
    df = _norm(blotter)
    signed = np.where(df["side"] == "S", 1.0, -1.0) * df["qty"] * df["price"]
    held = float(df.loc[df["side"] == "B", "qty"].sum() - df.loc[df["side"] == "S",
                                                                 "qty"].sum())
    return float(signed.sum() + held * float(mark))


def naive_error(res: dict, short_rate: float, long_rate: float | None = None) -> dict:
    """What a backtest that books every loss when it happens gets wrong, in this year.

    Rates are arguments. There is no default rate in this library; see
    ../../after-tax-backtesting/SKILL.md for why an after-tax number without one is not
    comparable to anything.
    """
    long_rate = short_rate if long_rate is None else long_rate
    s = summarise(res)
    overstated_deduction = -s["disallowed_loss"]          # positive dollars
    return {"overstated_deduction": overstated_deduction,
            "overstated_tax_saving": overstated_deduction * float(short_rate),
            "naive_realised": s["gross_realised"],
            "correct_realised": s["reported_realised"]}


# --------------------------------------------------------------------------- demo blotter
def price_path(seed: int = 20260909, months: int = 36, start: str = "2022-01-03",
               s0: float = 100.0, sigma: float = 0.32, mu: float = 0.02) -> pd.Series:
    """One seeded daily series. Both demo blotters trade THIS path, so they are comparable."""
    rng = np.random.default_rng(seed)
    n = months * 21 + 1
    dates = pd.bdate_range(start, periods=n)
    step = rng.normal(mu / 252.0 - 0.5 * sigma**2 / 252.0, sigma / np.sqrt(252.0), n)
    return pd.Series(s0 * np.exp(np.cumsum(step)), index=dates)


def make_rebalance_blotter(seed: int = 20260909, months: int = 36,
                           start: str = "2022-01-03", s0: float = 100.0,
                           sigma: float = 0.32, mu: float = 0.02,
                           target_notional: float = 100_000.0,
                           replacement_account: str = "taxable") -> pd.DataFrame:
    """A constant-dollar single-name book rebalanced every 21 business days.

    Nothing clever, and that is the point: 21 business days is 29-31 calendar days, so most
    rebalance trades sit inside the previous one's window and inside the next one's.
    """
    px = price_path(seed, months, start, s0, sigma, mu)
    dates, n = px.index, len(px)

    rows = []
    held = 0.0
    for i in range(0, n, 21):
        d, p = dates[i], float(px.iloc[i])
        target = target_notional / p
        delta = target - held
        if abs(delta) < 1.0:
            continue
        side = "B" if delta > 0 else "S"
        acct = replacement_account if side == "B" else "taxable"
        rows.append({"date": d, "side": side, "qty": abs(delta), "price": p, "account": acct})
        held = target
    out = pd.DataFrame(rows)
    out.attrs["final_price"] = float(px.iloc[-1])
    return out


def make_harvest_blotter(seed: int = 20260909, months: int = 36,
                         start: str = "2022-01-03", s0: float = 100.0,
                         sigma: float = 0.32, mu: float = 0.02,
                         shares: float = 1000.0) -> pd.DataFrame:
    """Monthly tax-loss harvesting on the SAME path: sell at a loss, buy back next day.

    This is the chain generator. Each repurchase carries the previous disallowed loss in its
    basis, so next month it is under water again, sells at a loss again, and the loss is
    disallowed again -- against a basis that already contains the one before it.
    """
    px = price_path(seed, months, start, s0, sigma, mu)
    dates = px.index
    rows = [{"date": dates[0], "side": "B", "qty": shares, "price": float(px.iloc[0]),
             "account": "taxable"}]
    basis = float(px.iloc[0])
    for i in range(21, len(px) - 1, 21):
        p = float(px.iloc[i])
        if p >= basis:                                   # nothing to harvest
            continue
        rows.append({"date": dates[i], "side": "S", "qty": shares, "price": p,
                     "account": "taxable"})
        rows.append({"date": dates[i + 1], "side": "B", "qty": shares,
                     "price": float(px.iloc[i + 1]), "account": "taxable"})
        basis = float(px.iloc[i + 1])
    out = pd.DataFrame(rows)
    out.attrs["final_price"] = float(px.iloc[-1])
    return out


def main() -> None:
    print("Wash sales -- what a monthly rebalance actually books")
    print(f"  IRC 1091(a): 30 days before AND 30 days after = {WINDOW_TOTAL_DAYS} days in total")

    b = make_rebalance_blotter()
    mark = b.attrs["final_price"]
    gaps = b["date"].diff().dt.days.dropna().astype(int)
    inside = int((gaps <= WINDOW_DAYS).sum())
    print(f"  seeded blotter: {len(b)} trades, constant $100k notional rebalanced every 21 "
          f"business days, mark {mark:.4f}")
    print(f"  consecutive trades are {gaps.min()}-{gaps.max()} calendar days apart; "
          f"{inside} of {len(gaps)} gaps are inside the 30-day window")

    res = apply_wash_sales(b)
    s = summarise(res, mark=mark)
    print()
    print(f"  loss sales                 {s['n_loss_sales']:>6d}")
    print(f"  of which wash sales        {s['n_wash_sales']:>6d}")
    print(f"  chained (basis already raised by an earlier wash sale) {s['n_chained']:>3d}, "
          f"max chain depth {s['max_chain_depth']}")
    print(f"  gross realised loss        {s['gross_loss']:>14,.2f}")
    print(f"  disallowed (deferred)      {s['disallowed_loss']:>14,.2f}")
    print(f"  fraction of the loss deferred {100 * s['fraction_of_loss_deferred']:>10.1f}%")
    print(f"  realised as reported       {s['reported_realised']:>14,.2f}")
    print(f"  naive backtest would book  {s['gross_realised']:>14,.2f}")

    err = naive_error(res, short_rate=0.37)
    print(f"  ASSUMED 37% rate: overstated tax saving {err['overstated_tax_saving']:,.2f}")

    # Deferral, not destruction: liquidate outside every window and the loss comes back.
    liq = pd.concat([b, pd.DataFrame([{
        "date": b["date"].iloc[-1] + pd.Timedelta(days=400), "side": "S",
        "qty": float(res["open_lots"]["qty"].sum()), "price": mark, "account": "taxable"}])],
        ignore_index=True)
    after = summarise(apply_wash_sales(liq))
    econ = economic_pnl(b, mark)
    print()
    print(f"  liquidate 400 days later, outside every window:")
    print(f"    reported realised then   {after['reported_realised']:>14,.2f}")
    print(f"    economic P&L of the book {econ:>14,.2f}")
    print(f"    difference               {after['reported_realised'] - econ:>14,.2f}"
          f"   <- deferral, not destruction")

    # The chain: harvest monthly on the same path and rebuy the next day.
    h = make_harvest_blotter()
    hres = apply_wash_sales(h)
    hs = summarise(hres, mark=h.attrs["final_price"])
    print()
    print(f"  same series, monthly loss harvest with a next-day repurchase "
          f"({len(h)} trades):")
    print(f"    loss sales {hs['n_loss_sales']}, ALL of them wash sales: "
          f"{hs['n_wash_sales']}")
    print(f"    gross loss {hs['gross_loss']:,.2f}, disallowed {hs['disallowed_loss']:,.2f}"
          f" = {100 * hs['fraction_of_loss_deferred']:.0f}%")
    print(f"    deepest chain: a lot re-based {hs['max_chain_depth']} time(s) before it was "
          f"sold at a loss again")
    h_econ = economic_pnl(h, h.attrs["final_price"])
    print(f"    economic P&L of the same 1,000 shares {h_econ:,.2f}; the reported gross loss "
          f"is {hs['gross_loss'] / h_econ:.1f}x that,")
    print( "      because each disallowed loss went back into the basis and was realised "
          "again next month")
    print(f"    deduction actually available this year: {hs['reported_realised']:,.2f}")

    # Lot method decides whether there are any wash sales at all.
    lifo = summarise(apply_wash_sales(b, "lifo"), mark=mark)
    print()
    print("  the SAME rebalance blotter matched LIFO instead of FIFO:")
    print(f"    loss sales {lifo['n_loss_sales']}, wash sales {lifo['n_wash_sales']}, "
          f"realised {lifo['reported_realised']:,.2f} (FIFO: {s['reported_realised']:,.2f})")

    # The one case where it IS destruction.
    ira = make_rebalance_blotter(replacement_account="ira")
    ira_res = apply_wash_sales(ira)
    ira_s = summarise(ira_res, mark=ira.attrs["final_price"])
    print()
    print("  same trades, every repurchase made inside an IRA (Rev. Rul. 2008-5):")
    print(f"    disallowed               {ira_s['disallowed_loss']:>14,.2f}")
    print(f"    of which PERMANENT       {ira_s['permanent_loss']:>14,.2f}"
          f"   <- no basis step-up, ever")

    print()
    print("RULE: a wash sale defers a loss into the replacement's basis, so book it there; "
          "the only common case where it is destroyed is a repurchase inside an IRA.")


if __name__ == "__main__":
    main()
