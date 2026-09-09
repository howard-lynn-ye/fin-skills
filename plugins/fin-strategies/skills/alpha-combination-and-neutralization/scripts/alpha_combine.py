#!/usr/bin/env python3
"""Combining alphas, neutralizing them, and the three ways an IC table lies.

One seeded cross-sectional panel - 200 names, 8 sectors, a market factor with time-varying
betas - and five alphas whose skill is planted in known places: one in the idiosyncratic
return, one in a slow idiosyncratic component, one in the sector return, one in the market
exposure, one pure noise. Because the generator knows where each alpha's information lives,
neutralization can be CHECKED rather than assumed: sector-neutralizing must destroy the
sector alpha and leave the idiosyncratic ones alone.

The three traps, all of which leave a plausible research note and no error:

  1. an IC t-stat computed on OVERLAPPING forward returns, when the alpha is persistent.
     The IC series is then autocorrelated, the i.i.d. standard error is wrong, and the
     t-stat overstates by several times on the same edge. (The section measures the case
     where it does NOT: a near-i.i.d. alpha's overlapping t-stat is roughly honest.)
  2. neutralizing against a beta the market has already told you - a window that reaches
     past t - and its everyday cousin, one full-sample beta per name applied to every date;
  3. winsorizing AFTER ranking, which clips exactly nothing.

Definitions used here (conventions, stated so they can be checked):
  * IC per period = cross-sectional correlation between the alpha known at t and the
    forward return from t. "rank IC" is the Spearman version (Pearson on ranks).
  * ICIR = mean(IC) / std(IC), annualised by sqrt(periods per year).
  * one-way turnover = 0.5 * sum_i |w_{t,i} - w_{t-1,i}| on a book of unit gross - the
    convention backtest-validation/scripts/cost_curve.py consumes.
  * Newey-West standard error of a mean, Bartlett kernel, L lags.

Run:  python alpha_combine.py       (numpy + pandas + scipy, fixed seed, under 10 s)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260909
PERIODS_PER_YEAR = 252
N_NAMES = 200
N_DAYS = 1200
N_SECTORS = 8


# ------------------------------------------------------------------------ synthetic panel
def simulate_panel(n_names: int = N_NAMES, n_days: int = N_DAYS, n_sectors: int = N_SECTORS,
                   seed: int = SEED) -> dict:
    """A cross-sectional panel whose alphas have their skill planted in known components.

    r_{t,i} = beta_{t,i} * mkt_t + sector_{t,s(i)} + eps_fast_{t,i} + eps_slow_{t,i}

    beta drifts slowly, so a single full-sample beta is genuinely stale and a window that
    reaches past t genuinely knows something. Each alpha at t is a noisy view of ONE
    component of r_{t+1}, so the right answer to "what survives sector neutralization" is
    known before anything is measured.
    """
    if n_names < 20 or n_days < 100 or n_sectors < 2:
        raise ValueError("need at least 20 names, 100 days and 2 sectors")
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    cols = [f"S{i:03d}" for i in range(n_names)]
    sector = pd.Series([f"SEC{i % n_sectors}" for i in range(n_names)], index=cols, name="sector")

    # betas drift: an AR(1) around 1.0, half-life ~138 days, cross-sectional sd 0.35, so a
    # single full-sample beta is genuinely stale by the end of the sample
    phi_b = 0.995
    b = 1.0 + rng.normal(0.0, 0.35, n_names)
    shock = rng.normal(0.0, 0.35 * np.sqrt(1 - phi_b ** 2), (n_days, n_names))
    bpath = np.empty((n_days, n_names))
    for t in range(n_days):
        b = 1.0 + phi_b * (b - 1.0) + shock[t]
        bpath[t] = b
    beta_t = pd.DataFrame(np.clip(bpath, 0.1, 2.5), idx, cols)

    mkt = pd.Series(rng.normal(0.0003, 0.011, n_days), index=idx)
    sec_ret = pd.DataFrame(rng.normal(0.0, 0.006, (n_days, n_sectors)), idx,
                           [f"SEC{i}" for i in range(n_sectors)])
    sec_by_name = sec_ret[sector.to_numpy()]
    sec_by_name.columns = cols

    eps_fast = pd.DataFrame(rng.normal(0.0, 0.012, (n_days, n_names)), idx, cols)
    phi_s, s = 0.95, np.zeros(n_names)
    s_shock = rng.normal(0.0, 0.004, (n_days, n_names))
    slow = np.empty((n_days, n_names))
    for t in range(n_days):
        s = phi_s * s + s_shock[t]
        slow[t] = s
    eps_slow = pd.DataFrame(slow, idx, cols)

    mkt_exposure = beta_t.mul(mkt, axis=0)
    ret = mkt_exposure + sec_by_name + eps_fast + eps_slow

    def ar1_noise(phi: float) -> pd.DataFrame:
        """Unit-variance AR(1) noise. phi sets how fast the alpha - and so the book - churns."""
        e = rng.normal(0.0, 1.0, (n_days, n_names))
        if phi <= 0.0:
            return pd.DataFrame(e, idx, cols)
        out = np.empty_like(e)
        v = e[0]
        scale = np.sqrt(1.0 - phi ** 2)
        for t in range(n_days):
            v = phi * v + scale * e[t]
            out[t] = v
        return pd.DataFrame(out, idx, cols)

    def view(component, skill: float, noise_phi: float = 0.0) -> pd.DataFrame:
        """A forecast made at t of the day t+1 component: skill * z(component) + AR(1) noise.

        `noise_phi` is what makes one alpha expensive and another cheap: it controls how
        much of the signal is refreshed each day, and therefore the book's turnover.
        """
        fwd = component.shift(-1)
        arr = fwd.to_numpy(dtype=float)
        z = (fwd - np.nanmean(arr)) / np.nanstd(arr)
        return skill * z + ar1_noise(noise_phi)

    # a persistent tilt AGAINST high-beta names - the shape a low-vol or quality alpha has,
    # and the only kind whose book carries a beta the neutralization has to actually remove
    low_beta = -cs_zscore(beta_t) + 0.6 * ar1_noise(0.98) + view(eps_fast, 0.030, 0.98)

    alphas = {
        "idio_fast": view(eps_fast, 0.055, 0.0),      # survives every neutralization
        "idio_slow": view(eps_slow, 0.075, 0.97),     # persistent, so cheap to trade
        "sector_bet": view(sec_by_name, 0.090, 0.90),  # dies under sector neutralization
        "low_beta": low_beta,                          # a persistent, real beta exposure
        "pure_noise": view(eps_fast, 0.0, 0.0),        # the control
    }
    return {"returns": ret, "alphas": alphas, "sector": sector, "beta_t": beta_t,
            "market": mkt, "sector_returns": sec_ret}


def forward_return(returns: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Compound return from t to t + horizon. Row t is knowable only after t + horizon."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    level = (1.0 + returns).cumprod()
    return level.shift(-horizon) / level - 1.0


# -------------------------------------------------------------- cross-sectional transforms
def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Demean and scale each row across names."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1, ddof=0).replace(0.0, np.nan), axis=0)


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank mapped to [-0.5, 0.5]. Scale-free and outlier-IMMUNE by
    construction, which is exactly why winsorizing it afterwards does nothing."""
    return df.rank(axis=1, pct=True) - 0.5


def winsorize(df: pd.DataFrame, n_sd: float = 3.0) -> pd.DataFrame:
    """Clip each row at +/- n_sd cross-sectional standard deviations."""
    if n_sd <= 0:
        raise ValueError("n_sd must be positive")
    mu, sd = df.mean(axis=1), df.std(axis=1, ddof=0)
    return df.clip(lower=mu - n_sd * sd, upper=mu + n_sd * sd, axis=0)


def clipped_fraction(df: pd.DataFrame, n_sd: float = 3.0) -> float:
    """Share of finite cells a winsorize at n_sd actually moves. 0.0 means the step is a no-op."""
    before = df.to_numpy(dtype=float)
    after = winsorize(df, n_sd).to_numpy(dtype=float)
    ok = np.isfinite(before) & np.isfinite(after)
    return float(np.mean(np.abs(before[ok] - after[ok]) > 1e-12)) if ok.any() else 0.0


# -------------------------------------------------------------------- IC and its t-stat
def _row_corr(a: np.ndarray, b: np.ndarray, min_names: int) -> np.ndarray:
    """Row-wise Pearson correlation of two aligned (T, N) arrays, NaN-aware and vectorised."""
    ok = np.isfinite(a) & np.isfinite(b)
    n = ok.sum(axis=1)
    az, bz = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
    cnt = np.maximum(n, 1)
    da = np.where(ok, az - (az.sum(axis=1) / cnt)[:, None], 0.0)
    db = np.where(ok, bz - (bz.sum(axis=1) / cnt)[:, None], 0.0)
    den = np.sqrt((da * da).sum(axis=1) * (db * db).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (da * db).sum(axis=1) / den
    return np.where((n >= min_names) & (den > 0), out, np.nan)


def ic_series(alpha: pd.DataFrame, fwd: pd.DataFrame, method: str = "spearman",
              min_names: int = 20) -> pd.Series:
    """Per-period cross-sectional IC between the alpha known at t and the forward return.

    Spearman is Pearson on within-row ranks; pandas' default 'average' tie handling matches
    scipy.stats.spearmanr, and the demo checks a sample of rows against scipy directly.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")
    a, f = alpha.align(fwd, join="inner")
    if method == "spearman":
        a, f = a.rank(axis=1), f.rank(axis=1)
    vals = _row_corr(a.to_numpy(dtype=float), f.to_numpy(dtype=float), min_names)
    return pd.Series(vals, index=a.index, name=f"{method}_ic").dropna()


def icir(ic: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """mean(IC)/std(IC), annualised. The information ratio of the IC series itself."""
    ic = pd.Series(ic).dropna()
    sd = ic.std(ddof=1)
    return float(ic.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan


def plain_tstat(x) -> float:
    """t of the mean assuming i.i.d. observations - what every IC table prints."""
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(x.size))) if sd > 0 else np.nan


def newey_west_se(x, lags: int) -> float:
    """HAC standard error of a mean, Bartlett kernel:

    S = g_0 + 2 * sum_{l=1..L} (1 - l/(L+1)) * g_l ,   se = sqrt(S / n)
    """
    if lags < 0:
        raise ValueError("lags must be >= 0")
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    n = x.size
    e = x - x.mean()
    s = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        s += 2.0 * (1.0 - lag / (lags + 1.0)) * float(e[lag:] @ e[:-lag]) / n
    return float(np.sqrt(max(s, 0.0) / n))


def newey_west_tstat(x, lags: int) -> float:
    se = newey_west_se(x, lags)
    return float(pd.Series(x).dropna().mean() / se) if se > 0 else np.nan


def overlap_tstats(alpha: pd.DataFrame, returns: pd.DataFrame, horizon: int,
                   method: str = "spearman") -> dict:
    """The same IC three ways: overlapping, non-overlapping, and HAC-corrected."""
    ic = ic_series(alpha, forward_return(returns, horizon), method)
    independent = ic.iloc[::horizon]
    return {
        "horizon": horizon, "mean_ic": float(ic.mean()),
        "n_overlapping": int(ic.size), "t_overlapping": plain_tstat(ic),
        "n_independent": int(independent.size), "t_non_overlapping": plain_tstat(independent),
        "t_newey_west": newey_west_tstat(ic, lags=horizon - 1),
        "ic_autocorr1": float(ic.autocorr(1)) if ic.size > 2 else np.nan,
    }


# --------------------------------------------------------------------------- neutralization
def _design(sector_codes: np.ndarray | None, beta_col: np.ndarray | None, n: int) -> np.ndarray:
    cols = [np.ones(n)]
    if sector_codes is not None:
        for c in np.unique(sector_codes)[1:]:        # drop one level against the intercept
            cols.append((sector_codes == c).astype(float))
    if beta_col is not None:
        cols.append(beta_col)
    return np.column_stack(cols)


def neutralize(alpha: pd.DataFrame, sector: pd.Series | None = None,
               beta: pd.DataFrame | pd.Series | None = None) -> pd.DataFrame:
    """Cross-sectional OLS residuals of the alpha on sector dummies and/or beta.

    This is what "sector-neutral" and "beta-neutral" mean operationally: per period,
    regress the alpha on the exposures and keep what is left. The residual has zero
    cross-sectional correlation with every column of the design, by construction.
    """
    if sector is None and beta is None:
        raise ValueError("neutralize needs a sector map, a beta frame, or both")
    names = alpha.columns
    codes = None
    if sector is not None:
        codes = pd.Series(sector).reindex(names).to_numpy()
        if pd.isna(codes).any():
            raise ValueError("sector map does not cover every name in the alpha")
    if isinstance(beta, pd.Series):
        beta = pd.DataFrame(np.tile(beta.reindex(names).to_numpy(), (len(alpha), 1)),
                            alpha.index, names)
    a_np = alpha.to_numpy(dtype=float)
    b_np = (beta.reindex(index=alpha.index, columns=names).to_numpy(dtype=float)
            if beta is not None else None)
    out = np.full(a_np.shape, np.nan)
    for t in range(a_np.shape[0]):
        y = a_np[t]
        ok = np.isfinite(y)
        if b_np is not None:
            ok = ok & np.isfinite(b_np[t])
        if ok.sum() < 5:
            continue
        x = _design(codes[ok] if codes is not None else None,
                    b_np[t][ok] if b_np is not None else None, int(ok.sum()))
        coef, *_ = np.linalg.lstsq(x, y[ok], rcond=None)
        out[t, ok] = y[ok] - x @ coef
    return pd.DataFrame(out, alpha.index, names)


def rolling_beta(returns: pd.DataFrame, market: pd.Series, window: int = 252,
                 lag: int = 1) -> pd.DataFrame:
    """Trailing OLS beta to the market. `lag=1` is causal; `lag=0` lets row t see r_t."""
    if window < 10 or lag < 0:
        raise ValueError("window must be >= 10 and lag >= 0")
    m = market.reindex(returns.index)
    return returns.rolling(window).cov(m).div(m.rolling(window).var(), axis=0).shift(lag)


def full_sample_beta(returns: pd.DataFrame, market: pd.Series) -> pd.Series:
    """ONE beta per name from the whole sample - the everyday version of trap 2."""
    m = market.reindex(returns.index)
    return returns.apply(lambda c: c.cov(m) / m.var())


def realized_beta(pnl: pd.Series, market: pd.Series) -> float:
    """OLS beta of a book's realised P&L on the market. What neutralization was FOR."""
    p, m = pd.Series(pnl).align(pd.Series(market), join="inner")
    ok = p.notna() & m.notna()
    return float(np.cov(p[ok], m[ok])[0, 1] / np.var(m[ok], ddof=1))


# --------------------------------------------------------------- weights, turnover, P&L
def to_weights(signal: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """Dollar-neutral book of unit gross: demean across names, scale so sum|w| = gross."""
    a = signal.sub(signal.mean(axis=1), axis=0)
    return a.div(a.abs().sum(axis=1).replace(0.0, np.nan), axis=0) * gross


def turnover(weights: pd.DataFrame) -> float:
    """Mean ONE-WAY turnover per period, 0.5 * sum_i |dw_i|. 1.0 means the book flipped."""
    return float((0.5 * weights.diff().abs().sum(axis=1)).iloc[1:].mean())


def ls_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """P&L of the book held over the NEXT period: weights.iloc[t] earns returns.iloc[t+1]."""
    w, r = weights.align(returns, join="inner")
    return (w.shift(1) * r).sum(axis=1, min_count=1).dropna()


def net_of_cost(gross_pnl: pd.Series, weights: pd.DataFrame, bps: float) -> pd.Series:
    """Charge `bps` on one-way turnover - the convention cost_curve.py uses."""
    cost = 0.5 * weights.diff().abs().sum(axis=1) * bps / 1e4
    return (gross_pnl - cost.reindex(gross_pnl.index)).dropna()


def sharpe(r, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    r = pd.Series(r).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan


# ------------------------------------------------------------------------------ combination
def combine_zscore(alphas: dict, weights: dict | None = None) -> pd.DataFrame:
    """Cross-sectionally z-score each alpha, then take a weighted sum. Keeps magnitude."""
    w = weights or {k: 1.0 for k in alphas}
    parts = [cs_zscore(a) * float(w.get(k, 0.0)) for k, a in alphas.items()]
    return sum(parts[1:], parts[0])


def combine_rank(alphas: dict, weights: dict | None = None) -> pd.DataFrame:
    """Cross-sectionally rank each alpha, then take a weighted sum. Discards magnitude."""
    w = weights or {k: 1.0 for k in alphas}
    parts = [cs_rank(a) * float(w.get(k, 0.0)) for k, a in alphas.items()]
    return sum(parts[1:], parts[0])


def ic_weights(alphas: dict, fwd: pd.DataFrame, method: str = "spearman") -> dict:
    """Weight each alpha by its mean IC, floored at zero."""
    return {k: max(float(ic_series(a, fwd, method).mean()), 0.0) for k, a in alphas.items()}


def cost_aware_weights(alphas: dict, fwd: pd.DataFrame, returns: pd.DataFrame, bps: float,
                       method: str = "spearman") -> dict:
    """Weight by mean IC discounted by what the alpha's own turnover costs.

    Scaling each IC by (net Sharpe / gross Sharpe) of that alpha traded alone charges it
    for its own turnover without needing a utility function. An alpha whose edge does not
    survive its own costs gets weight zero.
    """
    out = {}
    for k, a in alphas.items():
        w = to_weights(cs_zscore(a))
        g = ls_returns(w, returns)
        gs, ns = sharpe(g), sharpe(net_of_cost(g, w, bps))
        keep = 0.0 if not np.isfinite(gs) or gs <= 0 else max(ns / gs, 0.0)
        out[k] = max(float(ic_series(a, fwd, method).mean()), 0.0) * keep
    return out


def smooth(signal: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """EWMA the combined signal - the cheapest turnover control there is."""
    if halflife <= 0:
        raise ValueError("halflife must be positive")
    return signal.ewm(halflife=halflife).mean()


# ---------------------------------------------------------------------------------- demo
if __name__ == "__main__":
    p = simulate_panel()
    ret, alphas, sector, mkt = p["returns"], p["alphas"], p["sector"], p["market"]
    fwd1 = forward_return(ret, 1)
    BPS = 10.0
    print("=" * 92)
    print(f"ALPHA COMBINATION AND NEUTRALIZATION  --  {ret.shape[1]} names, {ret.shape[0]} days, "
          f"{sector.nunique()} sectors, seed {SEED}")
    print("=" * 92)

    # ---- 0. the vectorised IC agrees with scipy
    a0 = alphas["idio_fast"].iloc[100:110]
    f0 = fwd1.iloc[100:110]
    ours = ic_series(a0, f0).to_numpy()
    theirs = np.array([stats.spearmanr(a0.iloc[i], f0.iloc[i])[0] for i in range(len(ours))])
    print(f"\n0. rank IC vs scipy.stats.spearmanr on 10 rows: max abs diff "
          f"{np.max(np.abs(ours - theirs)):.2e}")

    # ---- 1. the five alphas
    print("\n1. Five alphas, each with its skill planted in a known component of the return")
    print(f"   {'alpha':<24}{'rank IC':>9}{'ICIR':>7}{'turnover':>10}{'gross SR':>10}"
          f"{'net SR':>9}   (cost {BPS:.0f} bps one-way)")
    for k, a in alphas.items():
        w = to_weights(cs_zscore(a))
        g = ls_returns(w, ret)
        ic = ic_series(a, fwd1)
        print(f"   {k:<24}{ic.mean():>9.4f}{icir(ic):>7.2f}{turnover(w):>10.1%}"
              f"{sharpe(g):>10.2f}{sharpe(net_of_cost(g, w, BPS)):>9.2f}")
    print("   The noise control shows an IC indistinguishable from zero, which is the point of")
    print("   having one. Note the turnover column: idio_fast has the second-best rank IC and")
    print("   the worst net Sharpe, because it refreshes the whole book every day.")

    # ---- 2. TRAP 1 - overlapping forward returns
    print("\n2. TRAP 1 - the IC t-stat on OVERLAPPING forward returns")
    for name in ("idio_slow", "idio_fast"):
        kind = "persistent" if name.endswith("slow") else "near-i.i.d."
        print(f"   {name}  ({kind} alpha)")
        print(f"     {'horizon':>7}{'mean IC':>9}{'AC(1) of IC':>13}{'t overlap':>11}"
              f"{'t indep':>9}{'t Newey-West':>14}{'NW / indep':>12}")
        for h in (1, 5, 21, 63):
            r = overlap_tstats(alphas[name], ret, h)
            print(f"     {h:>7}{r['mean_ic']:>9.4f}{r['ic_autocorr1']:>13.2f}"
                  f"{r['t_overlapping']:>11.1f}{r['t_non_overlapping']:>9.1f}"
                  f"{r['t_newey_west']:>14.1f}"
                  f"{r['t_newey_west'] / r['t_non_overlapping']:>11.2f}x")
    print("   READ THE AC(1) COLUMN FIRST. Overlap by itself inflates nothing: it is the")
    print("   AUTOCORRELATION of the IC series that breaks the i.i.d. standard error, and that")
    print("   needs a persistent ALPHA as well as an overlapping return. The near-i.i.d. alpha")
    print("   sampled daily against a 63-day return really does carry ~63x the observations and")
    print("   its naive t-stat is roughly honest; the persistent one's is not.")
    slow_ic = ic_series(alphas["idio_slow"], fwd1)
    print(f"   And the usual rule 'lags = h-1' is about the OVERLAP, not the alpha: at h=1 it")
    print(f"   prescribes 0 lags, so the daily IC of idio_slow keeps t = "
          f"{plain_tstat(slow_ic):.1f} even though its")
    print(f"   IC autocorrelation is {slow_ic.autocorr(1):.2f}. With 21 lags the same series gives t = "
          f"{newey_west_tstat(slow_ic, 21):.1f}.")

    # ---- 3. neutralization does what it says
    print("\n3. Neutralization: rank IC of each alpha, raw and residual")
    beta_ok = rolling_beta(ret, mkt, window=126, lag=1)
    print(f"   {'alpha':<24}{'raw':>10}{'sector-neutral':>16}{'beta-neutral':>14}{'both':>10}")
    for k in ("idio_fast", "idio_slow", "sector_bet", "low_beta"):
        a = alphas[k]
        vals = [ic_series(a, fwd1).mean(),
                ic_series(neutralize(a, sector=sector), fwd1).mean(),
                ic_series(neutralize(a, beta=beta_ok), fwd1).mean(),
                ic_series(neutralize(a, sector=sector, beta=beta_ok), fwd1).mean()]
        print(f"   {k:<24}{vals[0]:>10.4f}{vals[1]:>16.4f}{vals[2]:>14.4f}{vals[3]:>10.4f}")
    print("   sector_bet loses ALL of its IC to eight dummies; the idiosyncratic alphas barely")
    print("   move. That is the check to run: neutralization must remove the exposure you named")
    print("   and nothing else, and if your 'idiosyncratic' alpha dies too it was never")
    print("   idiosyncratic. low_beta is the instructive one - it keeps most of its IC, because")
    print("   its edge is only partly the tilt. What beta-neutralizing changes for low_beta is")
    print("   the BOOK's exposure, not the IC, and that is what section 4 measures.")

    # ---- 4. TRAP 2 - which beta you neutralized with
    print("\n4. TRAP 2 - neutralizing with a beta the market has already told you")
    beta_full = full_sample_beta(ret, mkt)
    beta_ahead = rolling_beta(ret, mkt, window=126, lag=0).shift(-63)
    raw_w = to_weights(cs_zscore(alphas["low_beta"]))
    print(f"   {'beta used to neutralize':<41}{'rank IC':>9}{'realized book beta':>20}"
          f"{'gross SR':>10}{'net SR':>9}")
    print(f"   {'(no neutralization at all)':<41}{ic_series(alphas['low_beta'], fwd1).mean():>9.4f}"
          f"{realized_beta(ls_returns(raw_w, ret), mkt):>20.4f}"
          f"{sharpe(ls_returns(raw_w, ret)):>10.2f}"
          f"{sharpe(net_of_cost(ls_returns(raw_w, ret), raw_w, BPS)):>9.2f}")
    for label, b in (("trailing 126d, shifted 1 day (causal)", beta_ok),
                     ("ONE full-sample beta per name", beta_full),
                     ("126d window ending 63 days AHEAD", beta_ahead)):
        resid = neutralize(alphas["low_beta"], beta=b)
        w = to_weights(cs_zscore(resid))
        g = ls_returns(w, ret)
        print(f"   {label:<41}{ic_series(resid, fwd1).mean():>9.4f}"
              f"{realized_beta(g, mkt):>20.4f}{sharpe(g):>10.2f}"
              f"{sharpe(net_of_cost(g, w, BPS)):>9.2f}")
    print("   Read the REALIZED-BETA column, not the IC column. Neutralization is a promise")
    print("   about the book's exposure, and the IC hardly notices whether it was kept.")
    print("\n   Does that ordering hold on other panels? |realized book beta|:")
    print(f"   {'panel':>18}{'none':>10}{'causal':>10}{'full-sample':>13}{'63d ahead':>11}")
    for kw in (dict(n_names=60, n_days=500, n_sectors=4, seed=5),
               dict(n_names=80, n_days=700, n_sectors=5, seed=11),
               dict()):
        q = simulate_panel(**kw)
        qr, qm, qa = q["returns"], q["market"], q["alphas"]["low_beta"]

        def bb(b=None, qa=qa, qr=qr, qm=qm):
            sig = qa if b is None else neutralize(qa, beta=b)
            return abs(realized_beta(ls_returns(to_weights(cs_zscore(sig)), qr), qm))

        tag = f"{qr.shape[1]}x{qr.shape[0]}"
        print(f"   {tag:>18}{bb():>10.4f}{bb(rolling_beta(qr, qm, 126, 1)):>10.4f}"
              f"{bb(full_sample_beta(qr, qm)):>13.4f}"
              f"{bb(rolling_beta(qr, qm, 126, 0).shift(-63)):>11.4f}")
    print("   INVARIANT: the window reaching 63 days past t removes ~85% of the book's beta")
    print("   on every panel, where every causal hedge removes about half. That gap is the")
    print("   leak, and it is not a property of one seed.")
    print("   NOT INVARIANT: whether ONE full-sample beta beats a noisy trailing one flips")
    print("   with the panel - a constant cannot track drift, but a short window is noisy,")
    print("   and which loses depends on your data. So you cannot reason your way to the")
    print("   right estimator; you have to regress the book's P&L on the factor and look.")
    print("   The gross-Sharpe column is not reliably signed either, which is exactly why")
    print("   'the Sharpe went up' is not how you detect this. Neither case errors.")

    # ---- 5. TRAP 3 - winsorizing after ranking
    print("\n5. TRAP 3 - winsorizing AFTER ranking clips nothing")
    fat = alphas["idio_fast"].copy()
    rows = fat.index[::37]
    hit = np.random.default_rng(SEED).choice(fat.columns, size=len(rows))
    for t, c in zip(rows, hit):
        fat.loc[t, c] *= 60.0                                   # a data-error-sized outlier
    kurt = float(stats.kurtosis(fat.to_numpy().ravel(), nan_policy="omit"))
    print(f"   raw alpha excess kurtosis {kurt:.1f} after injecting one 60x outlier every 37 days")
    print(f"   {'pipeline':<40}{'clip moves':>12}{'max |w|':>10}{'IC (pearson)':>14}"
          f"{'gross SR':>10}")
    pipes = {
        "z-score only, no winsorize": (None, cs_zscore(fat)),
        "winsorize -> z-score": (fat, cs_zscore(winsorize(fat, 3.0))),
        "z-score -> winsorize": (cs_zscore(fat), winsorize(cs_zscore(fat), 3.0)),
        "rank -> winsorize  (the no-op)": (cs_rank(fat), winsorize(cs_rank(fat), 3.0)),
    }
    for label, (clip_input, sig) in pipes.items():
        moved = "-" if clip_input is None else f"{clipped_fraction(clip_input, 3.0):.2%}"
        w = to_weights(sig)
        print(f"   {label:<40}{moved:>12}{float(w.abs().max().max()):>10.4f}"
              f"{ic_series(sig, fwd1, 'pearson').mean():>14.4f}"
              f"{sharpe(ls_returns(w, ret)):>10.2f}")
    print("   Ranking maps every row onto [-0.5, 0.5], whose cross-sectional sd is 0.289, so a")
    print("   3-sd clip sits at +/-0.87 - wider than the data can ever be. The step moves 0.00%")
    print("   of the cells: it is decoration that a reviewer reads as protection.")
    wz, zw = cs_zscore(winsorize(fat, 3.0)), winsorize(cs_zscore(fat), 3.0)
    d_frame = float(np.nanmax(np.abs(wz.to_numpy() - zw.to_numpy())))
    d_book = float(np.nanmax(np.abs(to_weights(wz).to_numpy() - to_weights(zw).to_numpy())))
    print("   Two things this does NOT say. (a) Ranking is fine, and here it scores best - the")
    print("   objection is to the dead winsorize line after it, not to the rank.")
    print(f"   (b) winsorize -> z-score and z-score -> winsorize print IDENTICAL rows above, and")
    print(f"   that is a fact about the METRICS, not the frames: the two frames differ by up to")
    print(f"   {d_frame:.3f}, but they are affine images of each other, so after to_weights they")
    print(f"   agree to {d_book:.2e} and every scale-free statistic downstream is the same. Order")
    print("   starts to matter the moment a step is NOT affine - which is what a rank is.")
    print("   The number that should worry you is max |w| with no clip at all: one name takes")
    print("   44% of a 200-name book because one cell of the input was 60x too large.")

    # ---- 6. turnover-aware combination
    print("\n6. Combining, and paying for it")
    tradeable = {k: alphas[k] for k in ("idio_fast", "idio_slow", "low_beta", "pure_noise")}
    icw = ic_weights(tradeable, fwd1)
    caw = cost_aware_weights(tradeable, fwd1, ret, BPS)
    print("   IC weights : " + "  ".join(f"{k}={v:.4f}" for k, v in icw.items()))
    print("   cost-aware : " + "  ".join(f"{k}={v:.4f}" for k, v in caw.items()))
    print(f"   {'combination':<36}{'rank IC':>10}{'turnover':>10}{'gross SR':>10}{'net SR':>9}")
    combos = {
        "equal-weight z-score": combine_zscore(tradeable),
        "IC-weighted z-score": combine_zscore(tradeable, icw),
        "IC-weighted rank": combine_rank(tradeable, icw),
        "cost-aware z-score": combine_zscore(tradeable, caw),
    }
    for label, sig in combos.items():
        w = to_weights(sig)
        g = ls_returns(w, ret)
        print(f"   {label:<36}{ic_series(sig, fwd1).mean():>10.4f}{turnover(w):>10.1%}"
              f"{sharpe(g):>10.2f}{sharpe(net_of_cost(g, w, BPS)):>9.2f}")
    print(f"\n   Smoothing the IC-weighted combination, halflife in days:")
    print(f"   {'halflife':>10}{'rank IC':>10}{'turnover':>10}{'gross SR':>10}{'net SR':>9}")
    base = combine_zscore(tradeable, icw)
    curve = {}
    for hl in (None, 0.5, 1.0, 2.0, 5.0, 20.0, 60.0):
        sig = base if hl is None else smooth(base, hl)
        w = to_weights(sig)
        g = ls_returns(w, ret)
        n = sharpe(net_of_cost(g, w, BPS))
        curve[hl] = n
        label = "none" if hl is None else f"{hl:g}"
        print(f"   {label:>10}{ic_series(sig, fwd1).mean():>10.4f}"
              f"{turnover(w):>10.1%}{sharpe(g):>10.2f}{n:>9.2f}")
    bestx = max(curve, key=lambda k: curve[k])
    interior = bestx is not None and bestx not in (0.5, 60.0)
    print(f"   net Sharpe peaks at halflife {'none' if bestx is None else bestx:g}"
          f" ({curve[bestx]:.2f}); the optimum is "
          f"{'INTERIOR' if interior else 'at an endpoint of this grid'}.")
    print("   Too little smoothing pays the spread, too much throws away the signal. Which")
    print("   side you are on depends on the cost, so it is not a constant of the strategy -")
    print("   pick it on a cost curve: backtest-validation/scripts/cost_curve.py.")

    print("\nRule: correct the IC t-stat for overlap or sample non-overlapping, neutralize with"
          " exposures known strictly before the return, and never winsorize a rank.")
