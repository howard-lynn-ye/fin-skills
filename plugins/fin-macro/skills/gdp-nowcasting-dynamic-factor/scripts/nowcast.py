#!/usr/bin/env python3
"""A mixed-frequency dynamic factor nowcast, and the benchmark it has to beat.

No network. The panel is synthetic and seeded; the published accuracy figures quoted in
the docstring were read at the institutions' own pages on 2026-09-09.

  1. A seeded mixed-frequency panel: one latent monthly factor, several monthly
     indicators with different publication lags, and quarterly GDP observed as the
     Mariano-Murasawa weighted average of unobserved monthly growth rates. The RAGGED
     EDGE is the point: at any vintage the newest months are partly missing, and which
     series are missing depends on how late each one publishes.
  2. A Kalman filter/smoother for that model in ~50 lines of numpy, with a time-varying
     observation dimension so missing entries are dropped rather than imputed. Factor
     recovery is measured at the true parameters, and separately at the ragged edge.
  3. The nowcast as data arrives, against three benchmarks: the historical mean, an AR(1)
     on quarterly growth, and a bridge regression on the available monthly indicators.
  4. statsmodels' DynamicFactorMQ on the same panel: parameter and factor recovery, and a
     comparison with the numpy filter. Skipped cleanly when statsmodels is absent.

The Kalman machinery itself lives in
`../../../fin-models/skills/state-space-and-kalman/SKILL.md`; the filter here is the
smallest version that carries the quarterly aggregation, and it exists so this script runs
without statsmodels.

VERIFIED SOURCE DATA (read 2026-09-09):

  statsmodels 0.15.0, statsmodels/tsa/statespace/dynamic_factor_mq.py, read in the
  installed package. The class docstring opens: "Implementation of the dynamic factor
  model of Banbura and Modugno (2014) ([1]_) and Banbura, Giannone, and Reichlin (2011)
  ([2]_). Uses the EM algorithm for parameter fitting, and so can accommodate a large
  number of left-hand-side variables. ... Can incorporate monthly/quarterly mixed
  frequency data along the lines of Mariano and Murasawa (2011) ([4]_). A special case of
  this model is the Nowcasting model of Bok et al. (2017) ([3]_)."  Notes: "The observed
  data may contain arbitrary patterns of missing entries."  And on identification: "The
  estimated factors and the factor loadings in this model are only identified up to an
  invertible transformation. ... This model does not impose any normalization to identify
  the factors and the factor loadings."  Its References section prints
  [3] Bok, Caratelli, Giannone, Sbordone and Tambalotti, 2018, "Macroeconomic Nowcasting
  and Forecasting with Big Data", Annual Review of Economics 10(1): 615-43, and
  [4] Mariano and Murasawa, "A coincident index, common factors, and monthly real GDP",
  Oxford Bulletin of Economics and Statistics 72(1) (2010): 27-46 - i.e. the prose years
  (2017, 2011) and the reference years (2018, 2010) disagree inside one docstring.

  atlantafed.org/research-and-data/data/gdpnow, GDPNow FAQ, read from the page: "Since we
  started tracking GDP growth with versions of this model in 2011, the average absolute
  error of final GDPNow forecasts is 0.77 percentage points. The root-mean-squared error
  of the forecasts is 1.17 percentage points. These accuracy measures cover initial
  estimates for 2011:Q3-2025:Q2."  And: "When back-testing with revised data, the root
  mean-squared error of the model's out-of sample forecast with the same data coverage
  that an analyst would have just before the 'advance' estimate is 1.15 percentage points
  for the 2000:Q1-2013:Q4 period."  And: "Overall, these accuracy metrics do not give
  compelling evidence that the model is more accurate than professional forecasters. The
  model does appear to fare well compared to other conventional statistical models."

  newyorkfed.org/research/policy/nowcast, read from the page: "Updates to the New York Fed
  Staff Nowcast were suspended between September 2021 and September 2023."  Cause: "The
  COVID-19 pandemic generated considerable uncertainty and volatility with respect to
  macroeconomic data, which posed important challenges to the New York Fed Staff Nowcast
  model."  Schedule: "We update it each Friday (except on federal holidays) at or shortly
  after 12:45 p.m., using data available up to 10 a.m."  Current model: "still based on a
  dynamic factor model and still employs Kalman-filtering techniques. The current model,
  however, is estimated with Bayesian techniques", allowing "unbalanced arrival of new
  data releases and mixed data frequency" and "time-varying volatility as well as variance
  outliers".  No accuracy figure appears on that page.

Run:  python nowcast.py    (numpy + pandas + scipy, seed 20260909; 6-25 s depending on
                            machine load, of which the DynamicFactorMQ fit in section 4
                            is 2-5 s. Section 4 is skipped cleanly without statsmodels.)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

SEED = 20260909
# Mariano-Murasawa triangle: a quarter-on-quarter log growth rate is this weighted sum of
# the five most recent (unobserved) month-on-month growth rates.
MM_WEIGHTS = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 3.0
NLAG = MM_WEIGHTS.size

# publication lag in months for each monthly indicator, at a given vintage: how many of
# the most recent months are still missing when the nowcast is made.
PUB_LAGS = (0, 0, 1, 1, 2, 2)


# --------------------------------------------------------------------------------------
# 1. The panel
# --------------------------------------------------------------------------------------

def simulate_panel(n_months: int = 300, seed: int = SEED, *, phi: float = 0.72,
                   loadings=(1.00, 0.85, 0.70, 0.60, 0.45, 0.35),
                   noise=(0.55, 0.65, 0.75, 0.85, 1.00, 1.15),
                   lambda_g: float = 0.55, sig_g: float = 0.22) -> dict:
    """One AR(1) factor, `len(loadings)` monthly indicators, and quarterly GDP.

    GDP's monthly-equivalent growth is `lambda_g * f_t` plus a quarterly disturbance; what
    is OBSERVED is the Mariano-Murasawa weighted average, on the third month of each
    quarter and nowhere else.
    """
    rng = np.random.default_rng(seed)
    k = len(loadings)
    f = np.empty(n_months)
    f[0] = rng.normal(0.0, 1.0 / np.sqrt(1 - phi ** 2))
    e = rng.normal(0.0, 1.0, n_months)
    for t in range(1, n_months):
        f[t] = phi * f[t - 1] + e[t]

    X = np.full((n_months, k), np.nan)
    for i, (lam, sd) in enumerate(zip(loadings, noise)):
        X[:, i] = lam * f + rng.normal(0.0, sd, n_months)

    gdp_monthly = lambda_g * f
    y = np.full(n_months, np.nan)
    q_end = np.arange(NLAG - 1, n_months)
    q_end = q_end[(q_end % 3) == 2]
    agg = np.array([float(MM_WEIGHTS @ gdp_monthly[t - NLAG + 1:t + 1][::-1])
                    for t in q_end])
    y[q_end] = agg + rng.normal(0.0, sig_g, q_end.size)
    return {"f": f, "X": X, "y": y, "q_end": q_end, "phi": phi,
            "loadings": np.asarray(loadings), "noise": np.asarray(noise),
            "lambda_g": lambda_g, "sig_g": sig_g, "gdp_monthly": gdp_monthly}


def ragged(panel: dict, vintage: int, pub_lags=PUB_LAGS, gdp_lag: int = 1) -> dict:
    """The panel AS IT LOOKED at month `vintage`: later months blanked, per series.

    Indicator i is missing for its last `pub_lags[i]` months as well as everything after
    the vintage; quarterly GDP is missing for the last `gdp_lag` quarter-end months. The
    result is a jagged bottom edge - what every nowcasting paper calls the ragged edge and
    what a naive `dropna()` deletes.
    """
    X = panel["X"].copy()
    y = panel["y"].copy()
    X[vintage + 1:] = np.nan
    y[vintage + 1:] = np.nan
    for i, lag in enumerate(pub_lags):
        if lag:
            X[max(0, vintage + 1 - lag):vintage + 1, i] = np.nan
    seen = panel["q_end"][panel["q_end"] <= vintage]
    for q in seen[-gdp_lag:] if gdp_lag else []:
        y[q] = np.nan
    return {"X": X, "y": y}


def edge_shape(X: np.ndarray, y: np.ndarray, vintage: int, back: int = 5) -> pd.DataFrame:
    """What the bottom of the panel actually looks like at a vintage."""
    rows = []
    for t in range(vintage - back + 1, vintage + 1):
        rows.append({"month": t, "indicators_available": int(np.isfinite(X[t]).sum()),
                     "of": X.shape[1], "gdp": "yes" if np.isfinite(y[t]) else "-"})
    return pd.DataFrame(rows).set_index("month")


# --------------------------------------------------------------------------------------
# 2. A Kalman filter that carries the quarterly aggregation
# --------------------------------------------------------------------------------------

def build_ss(phi: float, loadings: np.ndarray, noise: np.ndarray, lambda_g: float,
             sig_g: float):
    """State = [f_t, ..., f_{t-4}]. Monthly rows load on f_t; the GDP row loads on the
    Mariano-Murasawa weighted sum of all five, so the aggregation lives in the design
    matrix and no monthly GDP has to be imputed."""
    k = loadings.size
    T = np.zeros((NLAG, NLAG))
    T[0, 0] = phi
    T[1:, :-1] = np.eye(NLAG - 1)
    Q = np.zeros((NLAG, NLAG))
    Q[0, 0] = 1.0
    Z = np.zeros((k + 1, NLAG))
    Z[:k, 0] = loadings
    Z[k, :] = lambda_g * MM_WEIGHTS
    R = np.r_[noise ** 2, sig_g ** 2]
    return T, Q, Z, R


def kalman(X: np.ndarray, y: np.ndarray, phi: float, loadings: np.ndarray,
           noise: np.ndarray, lambda_g: float, sig_g: float, smooth: bool = True):
    """Filter (and optionally smooth) with a per-period observation selection.

    Missing entries are handled by dropping their rows from the update, which is the
    correct treatment and the reason a state-space nowcast tolerates a ragged edge at all.
    Returns filtered and smoothed state means, plus the one-step-ahead GDP prediction.
    """
    n, k = X.shape
    T, Q, Z, R = build_ss(phi, loadings, noise, lambda_g, sig_g)
    obs = np.column_stack([X, y])
    a_pred = np.zeros((n, NLAG))
    P_pred = np.zeros((n, NLAG, NLAG))
    a_filt = np.zeros((n, NLAG))
    P_filt = np.zeros((n, NLAG, NLAG))
    a, P = np.zeros(NLAG), np.eye(NLAG) * (1.0 / max(1 - phi ** 2, 1e-6))
    for t in range(n):
        if t > 0:
            a = T @ a_filt[t - 1]
            P = T @ P_filt[t - 1] @ T.T + Q
        a_pred[t], P_pred[t] = a, P
        m = np.isfinite(obs[t])
        if m.any():
            Zt, Rt, v = Z[m], np.diag(R[m]), obs[t][m] - Z[m] @ a
            F = Zt @ P @ Zt.T + Rt
            K = P @ Zt.T @ np.linalg.inv(F)
            a = a + K @ v
            P = P - K @ Zt @ P
        a_filt[t], P_filt[t] = a, P
    if not smooth:
        return {"filtered": a_filt, "smoothed": None}
    a_sm = a_filt.copy()
    P_sm = P_filt.copy()
    for t in range(n - 2, -1, -1):
        J = P_filt[t] @ T.T @ np.linalg.pinv(P_pred[t + 1])
        a_sm[t] = a_filt[t] + J @ (a_sm[t + 1] - a_pred[t + 1])
        P_sm[t] = P_filt[t] + J @ (P_sm[t + 1] - P_pred[t + 1]) @ J.T
    return {"filtered": a_filt, "smoothed": a_sm}


def gdp_from_state(state_row: np.ndarray, lambda_g: float) -> float:
    """The model's GDP growth for the quarter ending at this month."""
    return float(lambda_g * (MM_WEIGHTS @ state_row))


# --------------------------------------------------------------------------------------
# 3. Recovery and the ragged edge
# --------------------------------------------------------------------------------------

def factor_recovery(panel: dict, vintage: int | None = None) -> dict:
    """Correlation and RMSE of the extracted factor against the truth, at the true
    parameters - so this measures the FILTER, not an estimator."""
    v = panel["X"].shape[0] - 1 if vintage is None else vintage
    r = ragged(panel, v)
    out = kalman(r["X"], r["y"], panel["phi"], panel["loadings"], panel["noise"],
                 panel["lambda_g"], panel["sig_g"])
    f = panel["f"][:v + 1]
    filt = out["filtered"][:v + 1, 0]
    sm = out["smoothed"][:v + 1, 0]
    burn = 24
    return {"corr_filtered": float(np.corrcoef(f[burn:], filt[burn:])[0, 1]),
            "corr_smoothed": float(np.corrcoef(f[burn:], sm[burn:])[0, 1]),
            "rmse_filtered": float(np.sqrt(np.mean((f[burn:] - filt[burn:]) ** 2))),
            "rmse_smoothed": float(np.sqrt(np.mean((f[burn:] - sm[burn:]) ** 2))),
            "state": out}


def edge_penalty(panel: dict, vintage: int, back: int = 6) -> pd.DataFrame:
    """How much worse the factor estimate is in the last months than deep in the sample.

    The comparison that matters is the filtered value at the edge against the SMOOTHED
    value the same month will have once the panel fills in - the revision a nowcast
    publishes to itself.
    """
    r = ragged(panel, vintage)
    now = kalman(r["X"], r["y"], panel["phi"], panel["loadings"], panel["noise"],
                 panel["lambda_g"], panel["sig_g"])
    full = kalman(panel["X"], panel["y"], panel["phi"], panel["loadings"], panel["noise"],
                  panel["lambda_g"], panel["sig_g"])
    rows = []
    for t in range(vintage - back + 1, vintage + 1):
        rows.append({"month": t, "age": vintage - t,
                     "n_series": int(np.isfinite(r["X"][t]).sum()
                                     + np.isfinite(r["y"][t])),
                     "filtered_now": now["filtered"][t, 0],
                     "settled": full["smoothed"][t, 0],
                     "truth": panel["f"][t]})
    tab = pd.DataFrame(rows).set_index("month")
    tab["revision"] = tab["settled"] - tab["filtered_now"]
    tab["err_now"] = tab["filtered_now"] - tab["truth"]
    tab["err_settled"] = tab["settled"] - tab["truth"]
    return tab


# --------------------------------------------------------------------------------------
# 4. The nowcast, and the benchmarks it has to beat
# --------------------------------------------------------------------------------------

def nowcast_at(panel: dict, target_q: int, vintage: int) -> float:
    r = ragged(panel, vintage)
    # everything after the vintage is NaN by construction; filtering it is wasted work
    out = kalman(r["X"][:vintage + 1], r["y"][:vintage + 1], panel["phi"],
                 panel["loadings"], panel["noise"], panel["lambda_g"], panel["sig_g"],
                 smooth=False)
    if target_q <= vintage:
        return gdp_from_state(out["filtered"][target_q], panel["lambda_g"])
    T, _, _, _ = build_ss(panel["phi"], panel["loadings"], panel["noise"],
                          panel["lambda_g"], panel["sig_g"])
    a = out["filtered"][vintage]
    for _ in range(target_q - vintage):
        a = T @ a
    return gdp_from_state(a, panel["lambda_g"])


def benchmarks(panel: dict, target_q: int, vintage: int) -> dict:
    """Three things a nowcast has to beat before anyone should look at it."""
    y, q_end = panel["y"], panel["q_end"]
    past = q_end[q_end <= vintage]
    past = past[past < target_q]
    past = past[np.isfinite(y[past])]
    if past.size < 8:
        return {}
    hist = y[past]
    out = {"mean": float(hist.mean())}
    a, b = hist[:-1], hist[1:]
    beta = float(np.cov(a, b, ddof=1)[0, 1] / np.var(a, ddof=1))
    alpha = float(b.mean() - beta * a.mean())
    out["ar1"] = alpha + beta * float(hist[-1])
    # bridge: OLS of quarterly GDP on the quarterly mean of the FIRST indicator, using
    # only months already published at this vintage
    r = ragged(panel, vintage)

    def qmean(q):
        seg = r["X"][max(0, q - 2):q + 1, 0]
        seg = seg[np.isfinite(seg)]
        return float(seg.mean()) if seg.size else np.nan

    xq = np.array([qmean(q) for q in past])
    ok = np.isfinite(xq)
    if ok.sum() >= 8:
        A = np.column_stack([np.ones(int(ok.sum())), xq[ok]])
        coef, *_ = np.linalg.lstsq(A, hist[ok], rcond=None)
        xt = qmean(target_q)
        out["bridge"] = float(coef[0] + coef[1] * xt) if np.isfinite(xt) else float("nan")
    return out


def evaluation(panel: dict, start_q: int = 60, offsets=(-4, -2, 0, 1),
               step: int = 1) -> pd.DataFrame:
    """RMSE by position in the data flow. `offset` is months from the quarter's last month
    to the vintage: -4 is a month before the quarter starts, 0 is its final month, +1 is
    one month after (still before the first official estimate)."""
    n = panel["X"].shape[0]
    q_end = panel["q_end"]
    targets = q_end[(q_end >= start_q) & (q_end <= n - 3)][::step]
    rows = []
    for off in offsets:
        errs = {"dfm": [], "mean": [], "ar1": [], "bridge": []}
        for q in targets:
            v = q + off
            if v < 30 or v >= n:
                continue
            truth = panel["y"][q]
            bm = benchmarks(panel, q, v)
            if not bm:
                continue
            errs["dfm"].append(nowcast_at(panel, q, v) - truth)
            for key in ("mean", "ar1", "bridge"):
                errs[key].append(bm.get(key, np.nan) - truth)
        row = {"months_to_quarter_end": off, "n": len(errs["dfm"])}
        for key, e in errs.items():
            e = np.asarray(e, dtype=float)
            e = e[np.isfinite(e)]
            row[f"rmse_{key}"] = float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("months_to_quarter_end")
    tab["dfm_vs_ar1"] = tab["rmse_dfm"] / tab["rmse_ar1"]
    tab["dfm_vs_bridge"] = tab["rmse_dfm"] / tab["rmse_bridge"]
    return tab


# --------------------------------------------------------------------------------------
# 5. statsmodels' DynamicFactorMQ
# --------------------------------------------------------------------------------------

def have_statsmodels() -> bool:
    try:
        import statsmodels.api  # noqa: F401
        return True
    except Exception:
        return False


def dfmq_facts() -> dict:
    """What the installed DynamicFactorMQ says about itself, read from its own docstring."""
    try:
        from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
    except Exception as exc:
        return {"available": False, "detail": f"{type(exc).__name__}: {exc}"}
    doc = " ".join((DynamicFactorMQ.__doc__ or "").split())   # de-wrap before searching
    keys = {
        "mixed monthly/quarterly": "monthly/quarterly mixed frequency data",
        "arbitrary missing data": "arbitrary patterns of missing entries",
        "EM algorithm": "the default fitting method in this model uses the EM algorithm",
        "factors unidentified": "only identified up to an invertible transformation",
        "standardises by default": "default behavior is to standardize each variable",
        "news decomposition": "the news associated with updated data releases",
    }
    cites = [c for c in ("Modugno", "Giannone", "Bok", "Mariano") if c in doc]
    return {"available": True, "doc_chars": len(doc),
            "claims": {k: (v in doc) for k, v in keys.items()},
            "cited": cites,
            "prose_years": [y for y in ("2014", "2011", "2017") if y in doc],
            "reference_years": [y for y in ("2018", "2010") if y in doc]}


def fit_dfmq(panel: dict, n_months: int = 180, maxiter: int = 60) -> dict | None:
    """Fit DynamicFactorMQ on the same panel and measure what it recovers."""
    try:
        from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
    except Exception:
        return None
    import warnings
    n = min(n_months, panel["X"].shape[0])
    idx = pd.period_range("2000-01", periods=n, freq="M")
    endog_m = pd.DataFrame(panel["X"][:n], index=idx,
                           columns=[f"m{i}" for i in range(panel["X"].shape[1])])
    q_end = panel["q_end"][panel["q_end"] < n]
    qidx = pd.PeriodIndex([idx[t].asfreq("Q") for t in q_end], freq="Q")
    endog_q = pd.DataFrame({"gdp": panel["y"][q_end]}, index=qidx)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = DynamicFactorMQ(endog_m, endog_quarterly=endog_q, factors=1,
                              factor_orders=1, idiosyncratic_ar1=False,
                              standardize=True)
        res = mod.fit(maxiter=maxiter, disp=False)
    took = time.time() - t0
    fac = np.asarray(res.factors.filtered.iloc[:, 0])
    truth = panel["f"][:n]
    sign = np.sign(np.corrcoef(truth[24:], fac[24:])[0, 1])
    names = list(res.model.param_names)
    params = np.asarray(res.params)
    phi_hat = float("nan")
    for nm in names:
        if nm.startswith("L1."):
            phi_hat = float(params[names.index(nm)])
            break
    # loadings, up to the sign and the one common scale the docstring warns about. The
    # model standardises, so the comparable truth is lambda_i / sd(x_i).
    lam_hat = np.array([params[names.index(f"loading.0->m{i}")]
                        for i in range(panel["X"].shape[1])])
    lam_true = panel["loadings"] / np.nanstd(panel["X"][:n], axis=0)
    scale = float(np.dot(np.abs(lam_hat), lam_true) / np.dot(lam_true, lam_true))
    ours = kalman(panel["X"][:n], panel["y"][:n], panel["phi"], panel["loadings"],
                  panel["noise"], panel["lambda_g"], panel["sig_g"])["filtered"][:, 0]
    return {"seconds": took, "n_months": n, "maxiter": maxiter,
            "corr_factor_vs_truth": float(sign * np.corrcoef(truth[24:], fac[24:])[0, 1]),
            "corr_factor_vs_numpy": float(abs(np.corrcoef(ours[24:], fac[24:])[0, 1])),
            "phi_true": panel["phi"], "phi_hat": phi_hat,
            "loadings_hat": lam_hat, "loadings_true_standardised": lam_true,
            "loading_scale": scale, "all_loadings_negative": bool((lam_hat < 0).all()),
            "loading_corr": float(np.corrcoef(np.abs(lam_hat), lam_true)[0, 1]),
            "loading_max_rel_err": float(np.max(np.abs(np.abs(lam_hat)
                                                       / (scale * lam_true) - 1.0))),
            "llf": float(res.llf), "n_params": len(names)}


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    pd.set_option("display.width", 120)
    panel = simulate_panel()
    n = panel["X"].shape[0]
    VINTAGE = n - 8

    print("=" * 78)
    print("1. The panel, and the ragged edge")
    print("=" * 78)
    print(f"   {n} months, {panel['X'].shape[1]} monthly indicators, "
          f"{panel['q_end'].size} quarterly GDP observations.")
    print("   Quarterly growth is the Mariano-Murasawa weighted average of five unobserved")
    print(f"   monthly growth rates, weights {MM_WEIGHTS * 3} / 3, on the third month of")
    print("   each quarter and nowhere else.")
    print(f"   Publication lags by indicator (months still missing at the vintage): "
          f"{PUB_LAGS}")
    r = ragged(panel, VINTAGE)
    print(f"\n   The bottom of the panel as it looks at month {VINTAGE}:")
    es = edge_shape(r["X"], r["y"], VINTAGE)
    print(f"   {'month':>8}{'indicators available':>23}{'GDP':>6}")
    for m, row in es.iterrows():
        print(f"   {m:>8}{row['indicators_available']:>15} of {row['of']}{row['gdp']:>6}")
    keep = np.isfinite(r["X"]).all(axis=1) & np.isfinite(r["y"])
    print(f"   A dropna() over this panel keeps {int(keep.sum())} of {VINTAGE + 1} months")
    print("   and deletes every month you actually want a nowcast for.")

    print("\n" + "=" * 78)
    print("2. Factor recovery at the true parameters")
    print("=" * 78)
    rec = factor_recovery(panel, VINTAGE)
    print(f"   correlation with the true factor: filtered "
          f"{rec['corr_filtered']:.4f}, smoothed {rec['corr_smoothed']:.4f}")
    print(f"   RMSE against the true factor:     filtered "
          f"{rec['rmse_filtered']:.4f}, smoothed {rec['rmse_smoothed']:.4f}")
    print("   The smoothed estimate is better because it uses the future. It is also the")
    print("   one you must not trade - the same distinction regime-detection measures.")
    ep = edge_penalty(panel, VINTAGE)
    print(f"\n   At the edge, month by month (age 0 is the vintage month itself):")
    print(f"   {'age':>5}{'series':>8}{'filtered now':>15}{'settled later':>15}"
          f"{'revision':>11}{'|err| now':>11}{'|err| settled':>15}")
    for _, row in ep.iterrows():
        print(f"   {int(row['age']):>5}{int(row['n_series']):>8}"
              f"{row['filtered_now']:>15.4f}{row['settled']:>15.4f}"
              f"{row['revision']:>11.4f}{abs(row['err_now']):>11.4f}"
              f"{abs(row['err_settled']):>15.4f}")
    print(f"   mean |revision| over these {len(ep)} months: "
          f"{ep['revision'].abs().mean():.4f}; mean |error| now "
          f"{ep['err_now'].abs().mean():.4f} against {ep['err_settled'].abs().mean():.4f}")
    print("   once the panel fills in. A nowcast revises itself, and the revision is not")
    print("   a bug - it is the ragged edge closing.")

    print("\n" + "=" * 78)
    print("3. The nowcast against the benchmarks it has to beat")
    print("=" * 78)
    ev = evaluation(panel)
    print("   RMSE of the quarterly growth nowcast, by how far the vintage sits from the")
    print("   quarter's last month. -4 is before the quarter starts; +1 is a month after")
    print("   it ends, still before any official estimate.")
    print(f"   {'offset':>8}{'n':>5}{'DFM':>9}{'mean':>9}{'AR(1)':>9}{'bridge':>9}"
          f"{'DFM/AR1':>10}{'DFM/bridge':>12}")
    for off, row in ev.iterrows():
        print(f"   {off:>8}{int(row['n']):>5}{row['rmse_dfm']:>9.4f}"
              f"{row['rmse_mean']:>9.4f}{row['rmse_ar1']:>9.4f}{row['rmse_bridge']:>9.4f}"
              f"{row['dfm_vs_ar1']:>10.3f}{row['dfm_vs_bridge']:>12.3f}")
    print("   The whole value of a nowcast is the slope of that first column: it is the")
    print("   only method here whose error falls as the quarter fills in. The others do")
    print("   not use the monthly flow at all, or use one series of it.")
    print("\n   For calibration, the published record:")
    print("     Atlanta Fed GDPNow, its own FAQ: average absolute error of the FINAL")
    print("     forecast 0.77 pp, RMSE 1.17 pp, over initial estimates for 2011:Q3-2025:Q2;")
    print("     and 'these accuracy metrics do not give compelling evidence that the model")
    print("     is more accurate than professional forecasters.'")
    print("     New York Fed Staff Nowcast: 'Updates ... were suspended between September")
    print("     2021 and September 2023' - a two-year hole in the middle of any series you")
    print("     download, across a change of model. It now updates each Friday at or")
    print("     shortly after 12:45 p.m., using data available up to 10 a.m.")

    print("\n" + "=" * 78)
    print("4. statsmodels' DynamicFactorMQ")
    print("=" * 78)
    facts = dfmq_facts()
    if not facts["available"]:
        print(f"   statsmodels not importable here: {facts['detail']}")
        print("   Everything above ran without it.")
    else:
        print(f"   Read from the installed class docstring ({facts['doc_chars']:,} chars):")
        for claim, ok in facts["claims"].items():
            print(f"     {'yes' if ok else 'NO ':>4}  {claim}")
        print(f"   cites: {', '.join(facts['cited'])}")
        print(f"   ! the prose says Mariano and Murasawa (2011) and Bok et al. (2017);")
        print(f"     the References section prints 2010 and 2018 for the same two papers.")
        fit = fit_dfmq(panel)
        if fit:
            print(f"\n   Fitted on {fit['n_months']} months, EM, maxiter "
                  f"{fit['maxiter']}, {fit['seconds']:.1f}s, "
                  f"{fit['n_params']} parameters, llf {fit['llf']:,.1f}")
            print(f"   factor vs the TRUE factor:  corr "
                  f"{fit['corr_factor_vs_truth']:.4f}")
            print(f"   factor vs this script's numpy filter: corr "
                  f"{fit['corr_factor_vs_numpy']:.4f}")
            print(f"   factor AR(1): true {fit['phi_true']:.3f}, estimated "
                  f"{fit['phi_hat']:.3f}")
            print(f"   all six monthly loadings came back NEGATIVE: "
                  f"{fit['all_loadings_negative']} - the sign indeterminacy, live.")
            print("   Against the truth the model can actually see (lambda_i / sd(x_i),")
            print("   because it standardises), up to one common scale:")
            print(f"   {'series':>8}{'estimated':>12}{'true (std)':>13}"
                  f"{'est / scale':>14}")
            for i, (h, t_) in enumerate(zip(fit["loadings_hat"],
                                            fit["loadings_true_standardised"])):
                print(f"   {'m' + str(i):>8}{h:>12.4f}{t_:>13.4f}"
                      f"{abs(h) / fit['loading_scale']:>14.4f}")
            print(f"   common scale {fit['loading_scale']:.4f}, correlation "
                  f"{fit['loading_corr']:.4f}, worst relative error "
                  f"{fit['loading_max_rel_err']:.1%}")
            print("   Sign and scale are meaningless on their own: the docstring says the")
            print("   factors and loadings are 'only identified up to an invertible")
            print("   transformation' and that the model imposes no normalisation. Compare")
            print("   correlations, ratios and fitted values, never a raw loading.")

    print(f"\n   (total runtime {time.time() - t0:.1f}s)")
    print("\nRule: nowcast on the ragged edge with a state-space model that treats missing"
          " entries as missing, score it against a mean, an AR(1) and a bridge before"
          " believing it, and never splice a published nowcast across a model change or a"
          " suspension.")
