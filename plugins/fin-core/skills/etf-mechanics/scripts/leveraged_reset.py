#!/usr/bin/env python3
"""ETF mechanics that a price series hides.

Every computed number quoted in ../SKILL.md is printed by this file.

  A  daily reset: a 3x product is 3x the index for ONE day
  B  a flat-but-volatile year: the index ends unchanged, the 3x product does not
  C  the analytic decay approximation vs the exact path result
  D  a trending year: 3x can BEAT three times the index return
  E  holding-period sensitivity (Monte Carlo): median vs mean, by horizon
  F  financing and expense drag on a leveraged product
  G  NAV vs price: premium/discount from a holdings file, and the stale-NAV trap
  H  distributions: the phantom ex-date drop in an unadjusted price series
  I  expense-ratio drag; tracking difference vs tracking error
  J  today's holdings file used as a historical universe: the look-ahead size

Run:  python leveraged_reset.py
numpy + pandas only, fixed seed, ASCII-only output (a stock Windows console is cp1252).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 0
DAYS = 252


def pct(x: float, nd: int = 2) -> str:
    return f"{100.0 * x:+.{nd}f}%"


def leveraged_wealth(r: np.ndarray, lev: float, financing: float = 0.0,
                     expense: float = 0.0) -> np.ndarray:
    """Wealth path of a daily-reset `lev`-times product on index daily simple returns `r`.

    financing: annual rate paid on the borrowed (lev - 1) x NAV; applied only when lev > 1.
    expense:   annual expense ratio, accrued per trading day.
    Nothing else: no swap spread, no rebalancing cost, no tracking noise. Real products
    pay all three, so every figure here is a lower bound on the cost.
    """
    borrowed = max(lev - 1.0, 0.0)
    daily = 1.0 + lev * r - borrowed * financing / DAYS - expense / DAYS
    return np.cumprod(daily)


def shaped_path(z: np.ndarray, ann_vol: float, total_return: float) -> np.ndarray:
    """Daily simple returns with a random shape but pinned endpoints: realized log-vol is
    exactly `ann_vol` and the compounded return is exactly `total_return`."""
    n = len(z)
    z = (z - z.mean()) / z.std()
    logret = z * ann_vol / np.sqrt(DAYS) + np.log1p(total_return) / n
    return np.expm1(logret)


def analytic_lev_return(index_return: float, sum_r2: float, lev: float) -> float:
    """Second-order approximation of a daily-reset product's return over a path:
    W_L ~= (1 + R)^L * exp(-(L^2 - L) / 2 * sum(r^2)).  The exponent is the 'volatility
    drag'; sum(r^2) is the REALIZED variance of the path actually taken, not an ex-ante vol.
    """
    return (1.0 + index_return) ** lev * np.exp(-(lev * lev - lev) / 2.0 * sum_r2) - 1.0


# --------------------------------------------------------------------------------------
def section_a() -> None:
    print("A. Daily reset arithmetic, no costs")
    print("   one day: index +1.00% -> 3x +3.00%, -1x -1.00%.  That is the only horizon on"
          " which 'k times the index' holds.")
    cases = [("index +10% then -10%", [0.10, -0.10]),
             ("index +10% then back to flat", [0.10, 1.0 / 1.1 - 1.0])]
    print(f"   {'two-day path':<30} {'index':>8}   {'2x':>8} {'3x':>8} {'-1x':>8} {'-3x':>8}")
    for label, seq in cases:
        r = np.array(seq)
        idx = np.prod(1 + r) - 1
        outs = [np.prod(1 + lev * r) - 1 for lev in (2, 3, -1, -3)]
        print(f"   {label:<30} {pct(idx):>8}   " + " ".join(f"{pct(o):>8}" for o in outs))


def section_bc(rng: np.random.Generator) -> np.ndarray:
    print("\nB. Flat-but-volatile year: the index ends EXACTLY unchanged (252 days)")
    print(f"   {'ann vol':>7} {'index':>8} {'2x':>8} {'3x':>8} {'-1x':>8} {'-3x':>8} |"
          f" {'3x analytic':>11} {'error':>9}")
    z = rng.standard_normal(DAYS)
    keep = None
    for vol in (0.16, 0.25, 0.40):
        r = shaped_path(z, vol, 0.0)
        if vol == 0.16:
            keep = r
        sum_r2 = float(np.sum(r ** 2))
        idx = np.prod(1 + r) - 1
        outs = {lev: np.prod(1 + lev * r) - 1 for lev in (2, 3, -1, -3)}
        approx = analytic_lev_return(idx, sum_r2, 3)
        print(f"   {vol:>7.0%} {pct(idx):>8} {pct(outs[2]):>8} {pct(outs[3]):>8}"
              f" {pct(outs[-1]):>8} {pct(outs[-3]):>8} | {pct(approx):>11}"
              f" {100 * (outs[3] - approx):+6.2f} pp")
    print("   'analytic' = (1+R)^3 * exp(-3 * realized variance) - 1; the error is the"
          " third-order remainder.")

    print("\nC. How good is the approximation?  2,000 random flat 25%-vol years, 3x product")
    errs = []
    for _ in range(2000):
        r = shaped_path(rng.standard_normal(DAYS), 0.25, 0.0)
        exact = np.prod(1 + 3 * r) - 1
        errs.append(exact - analytic_lev_return(0.0, float(np.sum(r ** 2)), 3))
    errs = np.array(errs)
    print(f"   exact minus analytic: mean {100 * errs.mean():+.3f} pp, max abs"
          f" {100 * np.abs(errs).max():.3f} pp  (drag itself is about"
          f" {pct(analytic_lev_return(0.0, 0.25 ** 2, 3))})")
    print("   The approximation needs the REALIZED variance of the path.  It says nothing"
          " about the next year unless you also forecast the variance.")
    return keep


def section_d(rng: np.random.Generator) -> None:
    print("\nD. Trending year at 16% vol: 3x product vs three times the index return")
    print("   (same random shape for every row; only the pinned total return changes)")
    z = rng.standard_normal(DAYS)
    print(f"   {'index R':>8} {'3x product':>11} {'3 x R':>8} {'3x minus 3R':>12}")
    for total in (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.50):
        r = shaped_path(z, 0.16, total)
        w3 = np.prod(1 + 3 * r) - 1
        print(f"   {pct(total):>8} {pct(w3):>11} {pct(3 * total):>8}"
              f" {100 * (w3 - 3 * total):+9.2f} pp")
    print("   The gap is negative near zero (drag dominates) and positive in a strong trend"
          " (compounding dominates).  'Decay' is not the whole story.")


def section_e(rng: np.random.Generator, n_paths: int = 20_000) -> None:
    print(f"\nE. Holding-period sensitivity: {n_paths:,} Monte Carlo paths, zero-drift index,"
          " 3x product")
    print("   gap = (3x product return) - 3 x (index return) over the same days")
    horizons = [1, 5, 21, 63, 126, 252]
    for vol in (0.16, 0.25, 0.40):
        r = rng.normal(0.0, vol / np.sqrt(DAYS), size=(n_paths, DAYS))
        w_idx = np.cumprod(1 + r, axis=1)
        w_3x = np.cumprod(1 + 3 * r, axis=1)
        print(f"   ann vol {vol:.0%}:")
        print(f"   {'days':>6} {'median gap':>11} {'mean gap':>10} {'P(gap<0)':>9}"
              f" {'median 3x':>10} {'mean 3x':>9}")
        for h in horizons:
            gap = (w_3x[:, h - 1] - 1) - 3 * (w_idx[:, h - 1] - 1)
            print(f"   {h:>6} {100 * np.median(gap):+8.2f} pp {100 * gap.mean():+7.2f} pp"
                  f" {np.mean(gap < -1e-12):>9.1%} {pct(np.median(w_3x[:, h - 1]) - 1):>10}"
                  f" {pct(w_3x[:, h - 1].mean() - 1):>9}")
    print("   The MEAN gap stays near zero: E[prod(1 + 3r)] = 1 when E[r] = 0.  Volatility"
          " drag lowers the median outcome, not the expected one.  A one-day hold has no gap"
          " at all.")


def section_f(r_flat16: np.ndarray) -> None:
    print("\nF. Costs the expense ratio does not show (the flat 16%-vol year from B, 3x)")
    base = np.prod(1 + 3 * r_flat16) - 1
    fin = leveraged_wealth(r_flat16, 3, financing=0.04)[-1] - 1
    fin_er = leveraged_wealth(r_flat16, 3, financing=0.04, expense=0.0082)[-1] - 1
    print(f"   volatility drag only                       {pct(base):>8}")
    print(f"   + financing 2 x NAV borrowed at 4.00%/yr   {pct(fin):>8}   "
          f"({100 * (fin - base):+.2f} pp; the swap rate is NOT in the expense ratio)")
    print(f"   + 0.82%/yr expense ratio                   {pct(fin_er):>8}   "
          f"({100 * (fin_er - fin):+.2f} pp)")
    print("   4.00% is an input, not a measurement: the fund pays whatever its swap"
          " counterparties charge, which is not published as one number.")


def section_g(rng: np.random.Generator) -> None:
    print("\nG. NAV vs price")
    h = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                      "shares": [120_000, 80_000, 200_000, 50_000, 300_000],
                      "close": [187.34, 412.10, 55.72, 901.05, 23.18]})
    cash, liabilities, shares_out = 1_250_000.0, 310_000.0, 2_000_000
    h["value"] = h["shares"] * h["close"]
    nav = (h["value"].sum() + cash - liabilities) / shares_out
    price = 59.79
    prem = price / nav - 1
    print(f"   holdings file: {len(h)} lines, securities {h['value'].sum():,.0f} + cash"
          f" {cash:,.0f} - liabilities {liabilities:,.0f}, {shares_out:,} shares out")
    print(f"   NAV {nav:.4f}  close {price:.2f}  premium {pct(prem, 3)} ="
          f" {1e4 * prem:+.1f} bp")
    fee_bp = 1e4 * 500.0 / 50_000 / nav      # $500 creation fee on a 50,000-share unit
    basket_half_spread_bp, hedge_bp = 2.0, 1.0
    band = fee_bp + basket_half_spread_bp + hedge_bp
    print(f"   AP round-trip cost: creation fee {fee_bp:.1f} bp + basket half-spread"
          f" {basket_half_spread_bp:.1f} bp + hedge {hedge_bp:.1f} bp = {band:.1f} bp")
    print(f"   |premium| {1e4 * abs(prem):.1f} bp < {band:.1f} bp: no AP acts, and the premium"
          " is noise, not a signal.  (inputs chosen for the example; measure yours)")

    print("\n   The stale-NAV trap (home market closed during US hours), 2,520 days:")
    n = 2520
    home = rng.normal(0.0, 0.008, n)      # move while the home market is open
    us = rng.normal(0.0, 0.006, n)        # fair-value move during US hours, after home close
    fair = np.cumsum(home + us)           # log ETF price at the US close
    nav_log = np.cumsum(home) + np.concatenate([[0.0], np.cumsum(us)[:-1]])  # stale by us[t]
    prem = fair - nav_log
    nav_ret_next = np.diff(nav_log)
    px_ret_next = np.diff(fair)
    c_nav = np.corrcoef(prem[:-1], nav_ret_next)[0, 1]
    c_px = np.corrcoef(prem[:-1], px_ret_next)[0, 1]
    print(f"   premium std {100 * prem.std():.2f}% (it IS the US-hours move;"
          f" nothing is mispriced)")
    print(f"   corr(premium_t, next-day NAV return)   {c_nav:+.2f}   <- 'the premium"
          " mean-reverts!'  No: the NAV catches up.")
    print(f"   corr(premium_t, next-day PRICE return) {c_px:+.2f}   <- what you can"
          " actually trade")


def section_h(rng: np.random.Generator) -> None:
    print("\nH. Distributions: the phantom drop in an unadjusted price series (10 years)")
    n = DAYS * 10
    dates = pd.bdate_range("2016-01-04", periods=n)
    r = rng.normal(0.08 / DAYS, 0.15 / np.sqrt(DAYS), n)
    dist = np.zeros(n)
    quarter_ends = [i for i in range(1, n) if dates[i].quarter != dates[i - 1].quarter]
    dist[quarter_ends] = 0.0045                       # 1.80%/yr paid as four cash dividends
    cg_day = quarter_ends[11]                          # one 3% capital-gains payout, year 3
    dist[cg_day] += 0.03
    tr = np.cumprod(1 + r)                             # total return (distributions reinvested)
    px = np.cumprod(1 + r - dist)                      # what an unadjusted close series shows
    cagr = lambda w: w[-1] ** (DAYS / n) - 1
    print(f"   total-return CAGR {pct(cagr(tr))}   unadjusted-price CAGR {pct(cagr(px))}"
          f"   gap {100 * (cagr(tr) - cagr(px)):+.2f} pp/yr  (paid out {100 * dist.sum() / 10:.2f}%/yr)")
    px_ret = np.diff(px) / px[:-1]
    worst = np.argsort(px_ret)[:10] + 1
    n_ex = sum(dist[i] > 0 for i in worst)
    print(f"   of the 10 worst days in the unadjusted series, {n_ex} are ex-dates;"
          f" the capital-gains day prints {pct(px_ret[cg_day - 1])} vs a true"
          f" {pct(r[cg_day])}")
    cash, roc = 1.00, 0.60
    print(f"   return of capital: a {cash:.2f}/share payout with {roc:.2f} ROC moves the price by"
          f" the full {cash:.2f}; only {cash - roc:.2f} is income, and cost basis falls by {roc:.2f}")
    # An income product: low vol, monthly payout.  Own generator so later sections are unchanged.
    rng2 = np.random.default_rng(SEED + 1)
    vol2, pay2 = 0.048, 0.0035
    r2 = rng2.normal(0.045 / DAYS, vol2 / np.sqrt(DAYS), n)
    dist2 = np.zeros(n)
    month_starts = [i for i in range(1, n) if dates[i].month != dates[i - 1].month]
    dist2[month_starts] = pay2                          # 4.2%/yr paid monthly, bond-fund-like
    px2 = np.cumprod(1 + r2 - dist2)
    px2_ret = np.diff(px2) / px2[:-1]
    worst2 = np.argsort(px2_ret)[:10] + 1
    n_ex2 = sum(dist2[i] > 0 for i in worst2)
    print(f"   income product ({vol2:.1%} vol, {pay2:.2%} monthly payout ="
          f" {pay2 / (vol2 / np.sqrt(DAYS)):.1f} daily sigmas): {n_ex2} of the 10 worst"
          " unadjusted days are ex-dates")


def section_i(rng: np.random.Generator) -> None:
    print("\nI. Expense drag; tracking difference (TD) vs tracking error (TE), 10 years")
    n = DAYS * 10
    idx = rng.normal(0.08 / DAYS, 0.16 / np.sqrt(DAYS), n)
    funds = {
        "A: ER 0.95%, full replication": idx - 0.0095 / DAYS,
        "B: ER 0.03%, sampled (noise 0.50%/yr)": idx - 0.0003 / DAYS
        + rng.normal(0.0, 0.005 / np.sqrt(DAYS), n),
        "C: ER 0.20%, +0.05% lending, noise 0.10%": idx - 0.0015 / DAYS
        + rng.normal(0.0, 0.001 / np.sqrt(DAYS), n),
    }
    ann = lambda x: np.prod(1 + x) ** (DAYS / n) - 1
    print(f"   {'fund':<42} {'TD /yr':>8} {'TE /yr':>7} {'10y wealth vs index':>20}")
    for name, f in funds.items():
        td = ann(f) - ann(idx)
        te = np.std(f - idx, ddof=1) * np.sqrt(DAYS)
        rel = np.prod(1 + f) / np.prod(1 + idx) - 1
        print(f"   {name:<42} {pct(td):>8} {100 * te:>6.2f}% {pct(rel):>20}")
    print("   TD is the money.  TE is dispersion around it and can be large while TD is tiny"
          " (B), or zero while TD is a full expense ratio (A).")
    print(f"   TE also makes the realized TD noisy: a 0.50% TE over 10 years is"
          f" +/-{100 * 0.005 / np.sqrt(10):.2f}%/yr of noise in the TD you measure")
    print(f"   0.95%/yr for 10 years at a 0% index return: {pct((1 - 0.0095) ** 10 - 1)}"
          " of terminal wealth")


def section_j(rng: np.random.Generator) -> None:
    print("\nJ. Today's holdings file used as the historical universe (look-ahead)")
    n_stocks, months, top = 300, 120, 100
    mret = rng.normal(0.06 / 12, 0.30 / np.sqrt(12), size=(months, n_stocks))
    cap0 = np.exp(rng.normal(0.0, 1.0, n_stocks))          # dispersed starting caps
    caps = cap0 * np.cumprod(1 + mret, axis=0)              # cap at the END of each month
    caps_prev = np.vstack([cap0, caps[:-1]])                # cap known at the START of month
    members_today = np.argsort(caps[-1])[-top:]             # the file you download today
    pit, today = [], []
    for t in range(months):
        c = caps_prev[t]
        m_pit = np.argsort(c)[-top:]
        for members, out in ((m_pit, pit), (members_today, today)):
            w = c[members] / c[members].sum()
            out.append(float(np.dot(w, mret[t, members])))
    ann = lambda x: np.prod(1 + np.array(x)) ** (12 / months) - 1
    survivors = len(set(members_today) & set(np.argsort(caps_prev[0])[-top:]))
    print(f"   {n_stocks} stocks, identical expected returns, 30% idio vol, cap-weighted top"
          f" {top}, monthly")
    print(f"   point-in-time membership CAGR {pct(ann(pit))}   today's members held"
          f" throughout {pct(ann(today))}   gap {100 * (ann(today) - ann(pit)):+.2f} pp/yr")
    print(f"   {top - survivors} of today's {top} members were not in the index at the start;"
          " the gap is pure selection on realized return.  Real churn and dispersion"
          " differ, so measure it on your index, but the sign never does.")


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"ETF mechanics, seed={SEED}, numpy {np.__version__}, pandas {pd.__version__}\n")
    section_a()
    r_flat16 = section_bc(rng)
    section_d(rng)
    section_e(rng)
    section_f(r_flat16)
    section_g(rng)
    section_h(rng)
    section_i(rng)
    section_j(rng)
    print("\nRule: 'k times the index' is a one-day statement.  Over any longer holding"
          " period the answer depends on the path, and the sign of the gap is not fixed.")


if __name__ == "__main__":
    main()
