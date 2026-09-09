#!/usr/bin/env python3
"""Seasonal adjustment is a second, silent vintage.

No network. The series is synthetic and seeded; the agency facts below were read at their
primary sources on 2026-09-09.

  1. A ratio-to-moving-average (X-11 core) seasonal adjustment in ~40 lines of numpy, so
     the mechanism is visible rather than delegated to a binary that is not installed.
  2. CONCURRENT adjustment: recompute the factors every month from all data through that
     month, exactly as CES does. Measure how far a PAST month's published seasonally
     adjusted value moves when no new unadjusted data for it has arrived.
  3. The phantom signal: how much of the month-over-month change in today's published SA
     history was put there by re-adjustment rather than by data, and how often the SIGN of
     the change flips between the first print and the settled value.
  4. The alternative: freeze the factors point-in-time (or work in NSA), and measure what
     that costs.
  5. X-13ARIMA-SEATS in Python, and what happens without the Census binary.

VERIFIED SOURCE DATA (read 2026-09-09):

  bls.gov/web/empsit/cesseasadjtn.htm, "Technical note on seasonal adjustment of CES
  estimates", verbatim:
    "The CES program employs a concurrent seasonal adjustment methodology to seasonally
     adjust its national estimates of employment, hours, and earnings."
    "Under concurrent methodology, new seasonal factors are calculated each month using
     all relevant data up to and including the current month period."
    "Once a year, BLS seasonally adjusts CES series again, replacing the most recent 5
     years of historical estimates with newly seasonally adjusted estimates."
    "...using the U.S. Census Bureau's X-13ARIMA-SEATS application."

  bls.gov/web/empsit/cesvininfo.htm: "CES estimates are subject to revisions for up to 2
  months from their original publication due to ongoing receipt of sample data."  So any
  movement in a published SA value for a month older than that is NOT new data.

  bls.gov/web/empsit/cesbmart.htm: "Twenty-one months of not seasonally adjusted CES
  estimates for all data types are revised based on this new March level, prior to
  seasonal adjustment."  March 2025 benchmark: seasonally adjusted -898,000 (-0.6%), not
  seasonally adjusted -862,000 (-0.5%).

  census.gov/data/software/x13as.html: "X-13ARIMA-SEATS is seasonal adjustment software
  produced, distributed, and maintained by the Census Bureau", offering "ARIMA model-based
  seasonal adjustment using a version of the SEATS software" as well as "nonparametric
  adjustments from the X-11 procedure", distributed "for Windows(R) PC and Linux/Unix
  platforms".  Page updated 2025-07-10.

  statsmodels 0.15.0, statsmodels/tsa/x13.py, read in the installed package:
      BINARY_NAMES = ("x13as", "x12a", "x13as_ascii", "x13as_html")
      BINARY_NAMES += tuple(f"{name}.exe" for name in BINARY_NAMES)  # windows
  `_find_x12` looks for those on X13PATH, then X12PATH, then the given `x12path`; if none
  is found `_check_x12` raises X13NotFoundError("x12a and x13as not found on path. Give
  the path, put them on PATH, or set the X12PATH or X13PATH environmental variable.").
  Confirmed by running x13_arima_analysis on this machine.

Run:  python seasonal_adjustment.py    (numpy + pandas, seed 20260909, about 6 s)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

SEED = 20260909
PERIOD = 12

# --------------------------------------------------------------------------------------
# 1. The X-11 core, in numpy
# --------------------------------------------------------------------------------------


def centered_ma(x: np.ndarray, period: int = PERIOD) -> np.ndarray:
    """2x12 centred moving average - the trend estimate X-11 starts from."""
    w = np.r_[0.5, np.ones(period - 1), 0.5] / period
    n, h = x.size, period // 2
    out = np.full(n, np.nan)
    for t in range(h, n - h):
        out[t] = float(np.dot(w, x[t - h:t + h + 1]))
    return out


def _fill_ends(v: np.ndarray) -> np.ndarray:
    """Carry the nearest finite value into the leading and trailing gaps, interp inside."""
    ok = np.isfinite(v)
    if not ok.any():
        return np.ones_like(v)
    out = v.copy()
    first, last = int(np.flatnonzero(ok)[0]), int(np.flatnonzero(ok)[-1])
    out[:first] = out[first]
    out[last + 1:] = out[last]
    f = np.isfinite(out)
    if not f.all():
        out = np.interp(np.arange(out.size), np.flatnonzero(f), out[f])
    return out


def seasonal_factors(x: np.ndarray, period: int = PERIOD) -> np.ndarray:
    """TIME-VARYING multiplicative factors, one per observation - X-11's ratio-to-MA core.

    detrend with a 2x12 centred moving average -> SI ratios -> smooth each calendar
    month's SI ratios across years with a 3x3 moving average -> normalise each year to a
    geometric mean of 1.

    The important structural feature, and the whole reason concurrent adjustment revises:
    the centred trend does not exist for the LAST six months, so the factors at the end of
    the sample are extrapolated from one side. When six more months arrive they become
    centred, and they change - without anything about the old unadjusted data changing.
    This is the part of X-13 that produces the effect measured below, not a substitute for
    X-13 (no RegARIMA pre-adjustment, no outlier detection, no SEATS).
    """
    n = x.size
    trend = centered_ma(x, period)
    with np.errstate(invalid="ignore", divide="ignore"):
        si = np.where(np.isfinite(trend) & (trend != 0), x / trend, np.nan)
    fac = np.ones(n)
    w = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0            # 3x3 moving average
    for k in range(period):
        idx = np.arange(k, n, period)
        v = _fill_ends(si[idx])
        pad = np.r_[np.repeat(v[0], 2), v, np.repeat(v[-1], 2)]
        fac[idx] = np.convolve(pad, w, mode="valid")
    # normalise so that a centred 12-term average of the factors is 1 LOCALLY. Doing this
    # in year-long blocks instead would force the trailing partial year to average 1,
    # which is wrong and, at the end of a concurrent sample, wrong by a lot.
    lf = np.log(np.clip(fac, 1e-8, None))
    return np.exp(lf - _fill_ends(centered_ma(lf, period)))


def latest_by_month(fac: np.ndarray, period: int = PERIOD) -> np.ndarray:
    """The most recent factor for each calendar position - what a projection carries."""
    out = np.ones(period)
    for k in range(period):
        idx = np.arange(k, fac.size, period)
        if idx.size:
            out[k] = fac[idx[-1]]
    return out


def adjust(x: np.ndarray, period: int = PERIOD) -> np.ndarray:
    """Seasonally adjusted series: x divided by its own estimated factors."""
    return x / seasonal_factors(x, period)


# --------------------------------------------------------------------------------------
# 2. A seeded unadjusted series
# --------------------------------------------------------------------------------------

def simulate_nsa(n_months: int = 300, seed: int = SEED, *, level: float = 150_000.0,
                 drift: float = 130.0, seas_amp: float = 0.055,
                 seas_drift: float = 0.0025, noise: float = 0.0035) -> dict:
    """Employment-like NSA level: trend, an evolving 12-month seasonal, irregular noise.

    `seas_drift` is the point of the exercise: the true seasonal pattern EVOLVES, which is
    precisely why an agency recomputes its factors, and therefore why the published SA
    history keeps moving.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    base = np.array([np.sin(2 * np.pi * k / PERIOD) + 0.45 * np.cos(4 * np.pi * k / PERIOD)
                     for k in range(PERIOD)])
    base = base - base.mean()
    seas = np.empty(n_months)
    for i in range(n_months):
        amp = seas_amp * (1.0 + seas_drift * (i - n_months / 2) / 12.0)
        seas[i] = amp * base[i % PERIOD]
    trend = level + drift * t + rng.normal(0.0, 300.0, n_months).cumsum() * 0.6
    irregular = rng.normal(0.0, noise, n_months)
    nsa = trend * (1.0 + seas) * (1.0 + irregular)
    truth_sa = trend * (1.0 + irregular)
    return {"nsa": nsa, "truth_sa": truth_sa, "trend": trend, "seas": seas}


# --------------------------------------------------------------------------------------
# 3. Publication regimes
# --------------------------------------------------------------------------------------

def publish_concurrent(nsa: np.ndarray, start: int = 96) -> np.ndarray:
    """P[m, t] = the SA value for month t as published in month m, CES-style.

    New factors every month from all data through month m, applied to the whole history.
    Nothing about month t's unadjusted value changes; its published SA value does.
    """
    n = nsa.size
    P = np.full((n, n), np.nan)
    for m in range(start, n):
        P[m, :m + 1] = adjust(nsa[:m + 1])
    return P


def publish_projected(nsa: np.ndarray, start: int = 96) -> np.ndarray:
    """The older convention: factors computed once a year and PROJECTED forward.

    At each annual anchor the history is re-adjusted with fresh factors and the latest
    factor per calendar month is carried forward for the next twelve. Published history
    then moves once a year rather than twelve times - at the cost of a factor that can be
    up to twelve months stale on the newest observations.
    """
    n = nsa.size
    P = np.full((n, n), np.nan)
    fac = proj = anchor = None
    for m in range(start, n):
        if fac is None or m % PERIOD == 0:
            fac = seasonal_factors(nsa[:m + 1])
            proj = latest_by_month(fac)
            anchor = m
        row = np.empty(m + 1)
        row[:anchor + 1] = nsa[:anchor + 1] / fac[:anchor + 1]
        fwd = np.arange(anchor + 1, m + 1)
        row[anchor + 1:] = nsa[fwd] / proj[fwd % PERIOD]
        P[m, :m + 1] = row
    return P


def publish_frozen(nsa: np.ndarray, start: int = 96) -> np.ndarray:
    """Factors estimated ONCE, at `start`, and never touched. History never moves."""
    n = nsa.size
    proj = latest_by_month(seasonal_factors(nsa[:start + 1]))
    base = nsa / proj[np.arange(n) % PERIOD]
    P = np.full((n, n), np.nan)
    for m in range(start, n):
        P[m, :m + 1] = base[:m + 1]
    return P


# --------------------------------------------------------------------------------------
# 4. How much a settled month's published value moves
# --------------------------------------------------------------------------------------
NSA_REVISION_WINDOW = 2          # "revisions for up to 2 months from their original
                                 # publication" - beyond that, movement is not new data


def rewrite_profile(P: np.ndarray, start: int, ages=(1, 2, 3, 6, 12, 24)) -> pd.DataFrame:
    """Month-on-month movement in the published SA value for a month `age` months back.

    Expressed in percent of the level. Ages beyond NSA_REVISION_WINDOW are months whose
    unadjusted value is settled: every basis point of movement there was manufactured by
    re-adjustment.
    """
    n = P.shape[0]
    rows = []
    for age in ages:
        d = []
        for m in range(start + 1, n):
            t = m - age
            if t >= 0 and np.isfinite(P[m, t]) and np.isfinite(P[m - 1, t]):
                d.append(abs(P[m, t] / P[m - 1, t] - 1.0) * 100.0)
        d = np.asarray(d)
        rows.append({"age_months": age, "n": d.size, "mean_abs_pct": float(d.mean()),
                     "p95_pct": float(np.percentile(d, 95)), "max_pct": float(d.max()),
                     "new_data_possible": age <= NSA_REVISION_WINDOW})
    return pd.DataFrame(rows).set_index("age_months")


def settled_drift(P: np.ndarray, start: int, age: int = 24) -> dict:
    """Total distance a settled month's published SA value travels after it settles."""
    n = P.shape[0]
    firsts, lasts, paths = [], [], []
    for t in range(start, n - age):
        row = P[t + NSA_REVISION_WINDOW + 1:, t]
        row = row[np.isfinite(row)]
        if row.size < 3:
            continue
        firsts.append(row[0])
        lasts.append(row[-1])
        paths.append(np.abs(np.diff(row)).sum() / row[0] * 100.0)
    firsts, lasts = np.asarray(firsts), np.asarray(lasts)
    return {"n": firsts.size,
            "mean_abs_net_pct": float(np.abs(lasts / firsts - 1.0).mean() * 100),
            "max_abs_net_pct": float(np.abs(lasts / firsts - 1.0).max() * 100),
            "mean_path_pct": float(np.mean(paths))}


# --------------------------------------------------------------------------------------
# 5. The phantom signal
# --------------------------------------------------------------------------------------

def phantom_signal(P: np.ndarray, start: int) -> dict:
    """Decompose the change in TODAY's published SA history into data and re-adjustment.

    For each month t the analyst reads `d_final[t] = P[-1, t] - P[-1, t-1]`, the
    month-over-month change in the series they downloaded. What existed at the time is
    `d_rt[t] = P[t, t] - P[t, t-1]`, both values as published then. The difference is
    revision. Only the first `NSA_REVISION_WINDOW` months of it can be new data.
    """
    n = P.shape[0]
    fin = P[n - 1]
    ts = [t for t in range(start + 1, n) if np.isfinite(P[t, t]) and np.isfinite(P[t, t - 1])]
    d_final = np.array([fin[t] - fin[t - 1] for t in ts])
    d_rt = np.array([P[t, t] - P[t, t - 1] for t in ts])
    rev = d_final - d_rt
    lvl = np.array([fin[t] for t in ts])
    corr = float(np.corrcoef(d_final, d_rt)[0, 1])
    return {"n": len(ts),
            "sd_final": float(d_final.std(ddof=1)),
            "sd_realtime": float(d_rt.std(ddof=1)),
            "sd_revision": float(rev.std(ddof=1)),
            "rev_over_final": float(rev.std(ddof=1) / d_final.std(ddof=1)),
            "unexplained": float(1.0 - corr ** 2),
            "corr": corr,
            "sign_flip": float(np.mean(np.sign(d_final) != np.sign(d_rt))),
            "mean_abs_rev_pct": float(np.abs(rev / lvl).mean() * 100),
            "mean_abs_rev": float(np.abs(rev).mean())}


def accuracy(P: np.ndarray, truth_sa: np.ndarray, start: int) -> dict:
    """What each regime costs in ACCURACY, so the comparison is not one-sided."""
    n = P.shape[0]
    fin, rt = P[n - 1], np.array([P[t, t] for t in range(n)])
    idx = np.arange(start, n)
    err_f = (fin[idx] / truth_sa[idx] - 1.0) * 100
    err_r = (rt[idx] / truth_sa[idx] - 1.0) * 100
    return {"rmse_final_pct": float(np.sqrt((err_f ** 2).mean())),
            "rmse_realtime_pct": float(np.sqrt((err_r ** 2).mean()))}


# --------------------------------------------------------------------------------------
# 6. X-13 in Python
# --------------------------------------------------------------------------------------

def x13_status() -> dict:
    """Is statsmodels' X-13 wrapper usable here? Report exactly what it says if not."""
    try:
        from statsmodels.tsa.x13 import BINARY_NAMES, _find_x12
    except Exception as exc:                                     # pragma: no cover
        return {"statsmodels": False, "detail": f"{type(exc).__name__}: {exc}"}
    found = _find_x12()
    out = {"statsmodels": True, "binary_names": list(BINARY_NAMES),
           "binary_found": bool(found), "detail": str(found) if found else ""}
    if not found:
        try:
            from statsmodels.tsa.x13 import x13_arima_analysis
            idx = pd.period_range("2000-01", periods=120, freq="M")
            s = pd.Series(np.arange(120) * 0.1 + 10 * np.sin(np.arange(120) * np.pi / 6),
                          index=idx)
            x13_arima_analysis(s)
        except Exception as exc:
            out["exception"] = f"{type(exc).__module__}.{type(exc).__name__}"
            out["message"] = str(exc)
    return out


def compare_with_x13(nsa: np.ndarray) -> dict | None:
    """If the Census binary IS installed, check this script's core against it."""
    st = x13_status()
    if not st.get("binary_found"):
        return None
    from statsmodels.tsa.x13 import x13_arima_analysis
    idx = pd.period_range("2000-01", periods=nsa.size, freq="M")
    res = x13_arima_analysis(pd.Series(nsa, index=idx))
    ours = adjust(nsa)
    theirs = res.seasadj.to_numpy()
    rel = np.abs(ours / theirs - 1.0)
    return {"mean_abs_rel_diff_pct": float(rel.mean() * 100),
            "max_abs_rel_diff_pct": float(rel.max() * 100),
            "corr_of_changes": float(np.corrcoef(np.diff(ours), np.diff(theirs))[0, 1])}


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    START = 96

    print("=" * 78)
    print("1. What the agency says it does")
    print("=" * 78)
    print("   BLS, bls.gov/web/empsit/cesseasadjtn.htm, verbatim:")
    print("     'Under concurrent methodology, new seasonal factors are calculated each")
    print("      month using all relevant data up to and including the current month")
    print("      period.'")
    print("     'Once a year, BLS seasonally adjusts CES series again, replacing the most")
    print("      recent 5 years of historical estimates with newly seasonally adjusted")
    print("      estimates.'")
    print("   And bls.gov/web/empsit/cesvininfo.htm: 'CES estimates are subject to")
    print("   revisions for up to 2 months from their original publication due to ongoing")
    print("   receipt of sample data.'")
    print("   Put those together: past the 2-month window, the unadjusted number is")
    print("   settled. Anything the published SA value does after that is arithmetic.")

    sim = simulate_nsa()
    nsa, truth = sim["nsa"], sim["truth_sa"]
    n = nsa.size
    print(f"\n   Synthetic NSA series: {n} months, seasonal amplitude "
          f"{np.abs(sim['seas']).max() * 100:.1f}% of level at its widest,")
    print(f"   evolving slowly over the sample; irregular noise 0.35% of level.")

    print("\n" + "=" * 78)
    print("2. How far a settled month's PUBLISHED value moves, with no new data")
    print("=" * 78)
    Pc = publish_concurrent(nsa, START)
    Pp = publish_projected(nsa, START)
    Pf = publish_frozen(nsa, START)
    prof_c = rewrite_profile(Pc, START)
    prof_p = rewrite_profile(Pp, START)
    print("   Month-on-month movement in the published SA value for a month `age` months")
    print("   back, in percent of level. Ages past 2 cannot be new data.")
    print(f"   {'age (months)':>13}{'concurrent mean':>18}{'p95':>9}{'max':>9}"
          f"{'projected mean':>17}{'new data?':>11}")
    for age in prof_c.index:
        c, p = prof_c.loc[age], prof_p.loc[age]
        print(f"   {age:>13}{c['mean_abs_pct']:>18.4f}{c['p95_pct']:>9.4f}"
              f"{c['max_pct']:>9.4f}{p['mean_abs_pct']:>17.4f}"
              f"{('yes' if c['new_data_possible'] else 'NO'):>11}")
    sd = settled_drift(Pc, START)
    print("   Ages 1-3 move by the same amount here because this simplified filter gives")
    print("   every month inside the un-centred tail one shared end-normaliser; X-13's")
    print("   asymmetric end filters differ in that detail. The rows that carry the point")
    print("   are 6 and beyond, where the month is settled and the CENTRED trend filter")
    print(f"   has finally reached it: {prof_c.loc[6, 'mean_abs_pct']:.4f}% on average, "
          f"{prof_c.loc[6, 'max_pct']:.4f}% at worst,")
    print("   in a single month, on a number nobody has collected any new data about.")
    print(f"   Over its life, a settled month's published SA value ends "
          f"{sd['mean_abs_net_pct']:.3f}% from")
    print(f"   where it started (max {sd['max_abs_net_pct']:.3f}%) and travels "
          f"{sd['mean_path_pct']:.3f}% getting there.")
    print("   On a 158,000,000-job level 0.01% is 15,800 jobs, so the 12-month-old row")
    print(f"   ({prof_c.loc[12, 'mean_abs_pct']:.4f}%) is about "
          f"{prof_c.loc[12, 'mean_abs_pct'] / 0.01 * 15800 / 1000:.0f}k jobs a month "
          f"arriving from nowhere.")
    print("   The projected-factor regime spreads the same total rewrite over one event a")
    print("   year instead of twelve. It pays for that with a factor that can be twelve")
    print("   months stale on the newest observation (section 4).")

    print("\n" + "=" * 78)
    print("3. The phantom signal")
    print("=" * 78)
    ph = phantom_signal(Pc, START)
    print(f"   For each of {ph['n']} months, the over-the-month change as it was published")
    print("   at the time, and as it appears in the series you download today:")
    print(f"     sd of the change, as published today   {ph['sd_final']:>12,.0f}")
    print(f"     sd of the change, in real time         {ph['sd_realtime']:>12,.0f}")
    print(f"     sd of the difference (revision)        {ph['sd_revision']:>12,.0f}")
    print(f"     revision sd / published-today sd      {ph['rev_over_final']:>13.2f}")
    print(f"     correlation between the two           {ph['corr']:>13.4f}")
    print(f"     so {ph['unexplained']:.1%} of the variation in the change you read today "
          f"was NOT in the")
    print(f"     change that existed at the time, and the SIGN differs in "
          f"{ph['sign_flip']:.1%} of months.")
    print("   None of that revision is new data - every month here is at least three")
    print("   months old by the time it settles. A momentum, surprise or acceleration")
    print("   feature computed on today's SA history is partly reading a factor update")
    print("   that arrived after the decision it is supposed to inform.")
    acc = accuracy(Pc, truth, START)
    print(f"   The trade is real, not free: against the true adjusted series the settled")
    print(f"   concurrent estimate has RMSE {acc['rmse_final_pct']:.3f}% and the real-time")
    print(f"   one {acc['rmse_realtime_pct']:.3f}%. Re-adjustment does make the history")
    print("   more accurate. It just makes it a different series.")

    print("\n" + "=" * 78)
    print("4. Freezing the factors: what it costs and what it buys")
    print("=" * 78)
    prof_f = rewrite_profile(Pf, START)
    accs = {"concurrent": accuracy(Pc, truth, START),
            "projected (annual)": accuracy(Pp, truth, START),
            "frozen at the start": accuracy(Pf, truth, START)}
    print(f"   {'regime':<22}{'RMSE vs truth, settled':>24}{'RMSE, real time':>18}"
          f"{'12-mo-old rewrite':>20}")
    for name, P, prof in (("concurrent", Pc, prof_c), ("projected (annual)", Pp, prof_p),
                          ("frozen at the start", Pf, prof_f)):
        a = accs[name]
        print(f"   {name:<22}{a['rmse_final_pct']:>24.3f}{a['rmse_realtime_pct']:>18.3f}"
              f"{prof.loc[12, 'mean_abs_pct']:>20.4f}")
    print("   Frozen factors never rewrite a published value - the rewrite column is")
    print("   exactly zero - and on a series whose seasonality is drifting they are the")
    print("   least accurate. That is the whole trade, in one table: accuracy of the")
    print("   history against stability of the history. A backtest needs the second one.")
    print("   The practical form: keep the NSA vintage series, estimate your own factors")
    print("   on data available at each decision date, and never let a later factor")
    print("   estimate touch an earlier published value.")

    print("\n" + "=" * 78)
    print("5. X-13ARIMA-SEATS in Python")
    print("=" * 78)
    st = x13_status()
    print("   Census: 'X-13ARIMA-SEATS is seasonal adjustment software produced,")
    print("   distributed, and maintained by the Census Bureau' - a command-line program")
    print("   with Windows and Linux/Unix binaries, offering both the X-11 nonparametric")
    print("   filters and ARIMA-model-based SEATS adjustment.")
    print(f"   statsmodels present: {st['statsmodels']}")
    if st.get("statsmodels"):
        print(f"   binaries statsmodels looks for: {', '.join(st['binary_names'])}")
        print(f"   found on this machine: {st['binary_found']}")
        if not st["binary_found"]:
            print(f"   calling x13_arima_analysis raises {st.get('exception', '?')}:")
            print(f"     \"{st.get('message', '')}\"")
            print("   pip install statsmodels does NOT install the adjustment engine. The")
            print("   wrapper writes a spec file, shells out to the Census binary and")
            print("   parses its output; with no binary it raises before doing any work.")
            print("   Download it from census.gov/data/software/x13as.html and set")
            print("   X13PATH, or pass x12path= explicitly.")
    cmp = compare_with_x13(nsa)
    if cmp:
        print(f"   This script's X-11 core vs X-13: mean abs relative difference "
              f"{cmp['mean_abs_rel_diff_pct']:.3f}%, max "
              f"{cmp['max_abs_rel_diff_pct']:.3f}%, correlation of month-over-month "
              f"changes {cmp['corr_of_changes']:.4f}")
    else:
        print("   (With the binary installed this script checks its own X-11 core against")
        print("   X-13 and prints the comparison instead of asserting agreement.)")

    print(f"\n   (total runtime {time.time() - t0:.1f}s)")
    print("\nRule: treat the seasonally adjusted series as a vintage object like any other"
          " - a settled month's published SA value keeps moving with no new data, so back"
          "test on the NSA vintage with factors estimated point-in-time, or carry the SA"
          " vintages and say which you used.")
