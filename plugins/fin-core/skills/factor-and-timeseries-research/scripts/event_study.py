#!/usr/bin/env python3
"""Market-model event study with Boehmer-Musumeci-Poulsen standardisation.

WHY this exists: there is no usable Python event-study library. The only PyPI package,
`eventstudy`, is 0.1a12 -- an alpha from 2021, GPL-3.0, 69 stars -- and the alternatives
top out at 12 stars. So everyone writes their own ~40 lines, and the ~40 lines almost
always stop at a test whose denominator comes from the ESTIMATION window. That is the
error, and it is worth being precise about which test it breaks.

MEASURED IN `__main__` (300 replications, true abnormal return = 0, event-induced
variance multiplier U(1,4) on days [-1,+1]):

    test                                     denominator          rejection @ nominal 5%
    Patell / standardised-residual           estimation window            ~46%
    Corrado rank, classic time-series SE     estimation window            ~15%
    BMP (Boehmer-Musumeci-Poulsen)           cross-section                 ~5%
    generalised sign                         cross-section                 ~5%

Volatility rises BECAUSE of the event, so estimation-window variance understates
event-window variance, and every statistic that divides by the former reports a
one-in-two false positive as a one-in-twenty. BMP (1991) is the fix: divide each firm's
CAR by its own Patell forecast-error sd -- which puts every firm on a common scale -- and
then take the standard error from the CROSS-SECTION of those standardised values, so an
event-induced variance jump inflates the denominator instead of the rejection rate.

⚠️ ONE COMMON CLAIM DOES NOT SURVIVE MEASUREMENT. The plain cross-sectional t-test on
RAW, unstandardised CARs is often described as over-rejecting too. On this generator it
does not: it comes in at 3-5%, i.e. correctly sized to slightly conservative, because a
symmetric scale mixture inflates the numerator and the denominator together. Its real
failure is POWER -- see panel A, where it needs a ~2.5x larger effect than BMP to reach
the same t-statistic, because one 9%-vol microcap contributes as much to sd(CAR) as fifty
1.2%-vol names. Report BMP because the raw test is weak, and report Patell never.

What BMP does NOT fix: cross-sectional correlation from calendar clustering (an industry
shock, a regulatory date, one earnings season). If events pile up on the same dates the
effective N is far below the event count and no per-firm standardisation rescues it --
use calendar-time portfolios or a Kolari-Pynnonen correction, and say which.

Usage:
    from event_study import event_study, resolve_event_date

    res = event_study(returns_df, market_series, events=[("AAPL", "2024-02-02"), ...],
                      window=(-1, 1), est_window=250, gap=30)
    print(res.report())
    res.tests.loc["BMP (standardised)", "p_value"]      # <- the one to quote

METHOD (from ../references/_event-study-method.md):
    estimation window            gap      event window
    [ t-250 ............ t-31 ] [ ... ] [ t-1, t0, t+1 ... t+k ]

    AR_it  = R_it - (alpha_i + beta_i * R_mt)
    CAR_i  = sum(AR_it) over the window
    BHAR_i = prod(1+R_it) - prod(1+E[R_it])          # long horizons, badly skewed

STATE THE WINDOW IN ADVANCE. Choosing it after seeing results is p-hacking, and it counts
as a trial in any deflated-Sharpe / multiple-testing accounting downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------------------
# event timing -- §1 of the method note, where most event studies fail before any maths
# --------------------------------------------------------------------------------------
def resolve_event_date(timestamp: pd.Timestamp | str,
                       sessions: pd.DatetimeIndex,
                       close_hour: int = 16) -> pd.Timestamp | None:
    """Map an event TIMESTAMP to the session that could first trade on it.

    The event time is `available_at` (for an SEC filing, `acceptanceDateTime` converted to
    exchange local time), not the period the event describes. If it lands at or after the
    close, day 0 is the NEXT session -- Apple's earnings 8-Ks cluster at 16:30-17:30 ET,
    and treating those as same-day events trades on information that did not exist yet.

    A date-only timestamp (midnight) is treated as pre-close, since it carries no time of
    day to judge. Returns None if no session on or after the event remains in `sessions`.
    """
    ts = pd.Timestamp(timestamp)
    day = ts.normalize()
    after_close = (ts != day) and (ts.hour >= close_hour)
    lo = day + pd.Timedelta(days=1) if after_close else day
    pos = int(sessions.searchsorted(lo, side="left"))
    return sessions[pos] if pos < len(sessions) else None


# --------------------------------------------------------------------------------------
# market model
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Fit:
    """OLS market model over the estimation window (numpy; no statsmodels dependency)."""
    alpha: float
    beta: float
    sigma: float          # residual sd, sqrt(SSE / (L1 - 2))
    xbar: float           # estimation-window mean market return
    ss_x: float           # estimation-window sum of squared market deviations
    n_est: int
    resid: np.ndarray     # estimation-window abnormal returns (feeds the sign/rank tests)


def _market_model(y: np.ndarray, x: np.ndarray) -> _Fit:
    n = len(y)
    xbar, ybar = float(x.mean()), float(y.mean())
    ss_x = float(((x - xbar) ** 2).sum())
    if ss_x <= 0.0:
        raise ValueError("market return has zero variance over the estimation window")
    beta = float(((x - xbar) * (y - ybar)).sum() / ss_x)
    alpha = ybar - beta * xbar
    resid = y - alpha - beta * x
    sigma = float(np.sqrt(float(resid @ resid) / (n - 2)))
    return _Fit(alpha, beta, sigma, xbar, ss_x, n, resid)


def _car_prediction_sd(fit: _Fit, x_win: np.ndarray) -> float:
    """Patell forecast-error sd of the CUMULATIVE abnormal return over `x_win`.

        Var(sum AR) = sigma^2 * [ L2 + L2^2/L1 + L2^2 * (xbar_win - xbar_est)^2 / SS_x ]

    The second and third terms are parameter uncertainty: alpha and beta were estimated,
    so out-of-sample forecast errors are larger than in-sample residuals, and the excess
    grows when the event window's market returns sit far from the estimation-window mean.
    Dividing by the raw estimation-window sd instead -- the usual shortcut -- inflates
    every standardised statistic built on top of it.
    """
    l2 = len(x_win)
    l1 = fit.n_est
    drift = (float(x_win.mean()) - fit.xbar) ** 2
    var = fit.sigma ** 2 * (l2 + l2 ** 2 / l1 + l2 ** 2 * drift / fit.ss_x)
    return float(np.sqrt(var))


# --------------------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------------------
def naive_cross_sectional_test(car: np.ndarray) -> tuple[float, float]:
    """t = mean(CAR)/(sd(CAR)/sqrt(N)) on RAW CARs.

    Correctly sized on symmetric data, but weak: sd(CAR) is dominated by whichever handful
    of names has the widest idiosyncratic vol, so the effective sample size is far below N.
    """
    car = np.asarray(car, dtype=float)
    n = len(car)
    sd = float(car.std(ddof=1))
    if n < 2 or sd == 0.0:
        return float("nan"), float("nan")
    t = float(car.mean() / (sd / np.sqrt(n)))
    return t, float(2.0 * stats.t.sf(abs(t), n - 1))


def patell_test(scar: np.ndarray, n_est: int) -> tuple[float, float]:
    """Patell (1976) standardised-residual test. THE ONE THAT BLOWS UP -- never report it.

        Z = sum(SCAR_i) / sqrt( N * (L1-2)/(L1-4) )

    It assumes Var(SCAR_i) equals its estimation-window value (L1-2)/(L1-4). Event-induced
    variance makes the true variance several times that, and the statistic scales directly
    with the error. Included so the contrast with `bmp_test` is visible in every report.
    """
    scar = np.asarray(scar, dtype=float)
    n = len(scar)
    if n < 2 or n_est <= 4:
        return float("nan"), float("nan")
    z = float(scar.sum() / np.sqrt(n * (n_est - 2) / (n_est - 4)))
    return z, float(2.0 * stats.norm.sf(abs(z)))


def bmp_test(scar: np.ndarray) -> tuple[float, float]:
    """Boehmer-Musumeci-Poulsen (1991): cross-sectional t on STANDARDISED CARs.

    Same numerator as Patell, different denominator. `scar` is CAR_i divided by its own
    forecast-error sd; taking the standard error from the cross-section of those is what
    absorbs event-induced variance -- if the event triples everyone's volatility the SCARs
    triple too and the statistic is unchanged, where Patell's would triple.
    """
    scar = np.asarray(scar, dtype=float)
    n = len(scar)
    sd = float(scar.std(ddof=1))
    if n < 2 or sd == 0.0:
        return float("nan"), float("nan")
    t = float(scar.mean() / (sd / np.sqrt(n)))
    return t, float(2.0 * stats.t.sf(abs(t), n - 1))


def generalized_sign_test(car: np.ndarray, p_positive: float) -> tuple[float, float]:
    """Cowan (1992) generalised sign test on the sign of each event's CAR.

    `p_positive` is the fraction of POSITIVE abnormal returns in the estimation windows,
    not 0.5 -- benchmarking against 0.5 mis-sizes the test whenever the market model
    leaves a skewed residual, which it usually does. Continuity-corrected: without it the
    normal approximation to a binomial over-rejects at the event counts studies actually
    have (measured ~8% at a nominal 5% for N=120).
    """
    car = np.asarray(car, dtype=float)
    n = len(car)
    p = float(min(max(p_positive, 1e-6), 1.0 - 1e-6))
    w = int((car > 0).sum())
    denom = np.sqrt(n * p * (1.0 - p))
    if denom == 0.0:
        return float("nan"), float("nan")
    raw = w - n * p
    z = float((raw - np.sign(raw) * 0.5) / denom) if raw != 0.0 else 0.0
    return z, float(2.0 * stats.norm.sf(abs(z)))


def _pooled_ranks(ar_est: np.ndarray, ar_win: np.ndarray) -> tuple[np.ndarray, int]:
    """Rank each firm's ARs within its OWN pooled estimation+event series, centred on 0.5.

    Ranking within the firm removes firm-level scale entirely, which is what makes the
    rank tests the outlier-robust cross-check on the parametric ones.
    """
    ar_est = np.atleast_2d(np.asarray(ar_est, dtype=float))
    ar_win = np.atleast_2d(np.asarray(ar_win, dtype=float))
    n, l2 = ar_win.shape
    pooled = np.hstack([ar_est, ar_win])
    total = pooled.shape[1]
    order = pooled.argsort(axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(1, total + 1, dtype=float)
    return ranks / (total + 1.0) - 0.5, l2


def corrado_rank_test_classic(ar_est: np.ndarray, ar_win: np.ndarray) -> tuple[float, float]:
    """Corrado (1989) as originally published -- a TIME-SERIES standard error.

    S(K) is the sd of the cross-sectional mean rank deviation over the whole pooled period,
    so ~98% of it comes from estimation days. Event-induced variance pushes event-day ARs
    to the extreme ranks, where |K| concentrates near 0.5 instead of spreading uniformly,
    so the event-day dispersion is larger than S(K) admits. Measured rejection at a nominal
    5% on the `__main__` generator: ~15%. Same disease as Patell, different notation.
    """
    k, l2 = _pooled_ranks(ar_est, ar_win)
    kbar = k.mean(axis=0)
    s_k = float(np.sqrt((kbar ** 2).mean()))
    if s_k == 0.0 or k.shape[0] < 2:
        return float("nan"), float("nan")
    return (float(kbar[-l2:].sum() / (np.sqrt(l2) * s_k)),
            float(2.0 * stats.norm.sf(abs(kbar[-l2:].sum() / (np.sqrt(l2) * s_k)))))


def corrado_rank_test(ar_est: np.ndarray, ar_win: np.ndarray) -> tuple[float, float]:
    """Corrado ranks with a CROSS-SECTIONAL standard error -- the rank analogue of BMP.

    Per firm, U_i = sum of its centred event-window rank deviations / sqrt(L2); then a
    cross-sectional t on the U_i. Under the null the K_it are exchangeable within firm, so
    E[U_i] = 0 whatever the event does to the firm's volatility, and the cross-sectional sd
    absorbs the rank concentration that breaks the classic form.
    """
    k, l2 = _pooled_ranks(ar_est, ar_win)
    n = k.shape[0]
    u = k[:, -l2:].sum(axis=1) / np.sqrt(l2)
    sd = float(u.std(ddof=1))
    if n < 2 or sd == 0.0:
        return float("nan"), float("nan")
    t = float(u.mean() / (sd / np.sqrt(n)))
    return t, float(2.0 * stats.t.sf(abs(t), n - 1))


# --------------------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------------------
@dataclass
class EventStudyResult:
    ar: pd.DataFrame                 # events x relative day, abnormal returns
    aar: pd.Series                   # average abnormal return per relative day
    caar: pd.Series                  # cumulative AAR across the display window
    car: pd.Series                   # per-event CAR over the TEST window
    scar: pd.Series                  # per-event forecast-error-standardised CAR
    bhar: pd.Series                  # per-event buy-and-hold abnormal return
    beta: pd.Series
    tests: pd.DataFrame
    window: tuple[int, int]
    est_window: int
    gap: int
    n_events_in: int
    attrition: dict[str, int] = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return len(self.car)

    def report(self, title: str = "EVENT STUDY") -> str:
        w = 84
        lines = [title, "=" * w,
                 f"window {self.window}  |  estimation {self.est_window} sessions ending "
                 f"{self.gap} sessions before t0  |  N = {self.n_events}",
                 "-" * w, "ATTRITION",
                 f"  {'events supplied':<50} {self.n_events_in:>6}"]
        for reason, k in self.attrition.items():
            lines.append(f"  {'dropped: ' + reason:<50} {-k:>6}")
        lines += [f"  {'events used':<50} {self.n_events:>6}", "-" * w,
                  f"CAAR{list(self.window)} = {self.car.mean():+.3%}    "
                  f"mean BHAR = {self.bhar.mean():+.3%}    "
                  f"median CAR = {self.car.median():+.3%}    "
                  f"mean beta = {self.beta.mean():.2f}",
                  "-" * w,
                  f"{'test':<34} {'statistic':>10} {'p_value':>10}   note"]
        for name, row in self.tests.iterrows():
            lines.append(f"{name:<34} {row['statistic']:>10.3f} {row['p_value']:>10.4f}"
                         f"   {row['note']}")
        lines += ["-" * w,
                  "AAR / CAAR by relative day. The pre-event days are shown on purpose: "
                  "drift before t0 is",
                  "either timestamp leakage or genuine anticipation, and you need to see "
                  "which."]
        lines.append("  day " + "".join(f"{d:>8}" for d in self.aar.index))
        lines.append("  AAR " + "".join(f"{v:>8.2%}" for v in self.aar.values))
        lines.append("  CAAR" + "".join(f"{v:>8.2%}" for v in self.caar.values))
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# the study
# --------------------------------------------------------------------------------------
def event_study(returns: pd.DataFrame,
                market: pd.Series,
                events: Sequence[tuple[str, object]],
                window: tuple[int, int] = (-1, 1),
                est_window: int = 250,
                gap: int = 30,
                pre_days: int = 5,
                post_days: int = 20,
                min_nonzero: int = 30) -> EventStudyResult:
    """Market-model event study over `events` = [(ticker, day0), ...].

    returns    : DataFrame of simple returns, dates x tickers.
    market     : Series of market returns on the same calendar.
    window     : (a, b) INCLUSIVE relative-day test window. Declare it before you look.
    est_window : estimation-window length in sessions.
    gap        : sessions between the end of the estimation window and t0, so the event's
                 own volatility and any anticipation stay out of the betas. >= 10-30.
    pre_days,
    post_days  : display window for AAR/CAAR reporting; widened to contain `window`.
    min_nonzero: minimum non-zero returns required in the estimation window. Thin trading
                 breaks OLS beta estimation; this is the cheap guard (the expensive one is
                 Scholes-Williams or Dimson betas).

    Events with incomplete data anywhere in the estimation or display window are DROPPED,
    not patched: equal-length windows are what make the rank tests valid, and a
    forward-filled return is a fabricated observation. Every drop is counted in
    `.attrition` -- report that table.

    Survivorship warning: `returns` must still contain the firms that later delisted,
    especially for distress-related events. See ../../research-integrity-guards.
    """
    a, b = int(window[0]), int(window[1])
    if a > b:
        raise ValueError(f"window must be (low, high); got {window}")
    if gap < 0 or est_window < 10:
        raise ValueError("need gap >= 0 and est_window >= 10")
    disp_a, disp_b = min(a, -abs(pre_days)), max(b, abs(post_days))
    rel_days = list(range(disp_a, disp_b + 1))

    idx = returns.index
    mkt = market.reindex(idx).to_numpy(dtype=float)
    pos_of_date = {ts: i for i, ts in enumerate(idx)}
    col_of_ticker = {c: j for j, c in enumerate(returns.columns)}
    mat = returns.to_numpy(dtype=float)

    attrition: dict[str, int] = {}

    def _drop(reason: str) -> None:
        attrition[reason] = attrition.get(reason, 0) + 1

    labels: list[str] = []
    ar_rows: list[np.ndarray] = []
    est_rows: list[np.ndarray] = []
    cars: list[float] = []
    scars: list[float] = []
    bhars: list[float] = []
    betas: list[float] = []

    for ticker, day0 in events:
        j = col_of_ticker.get(ticker)
        if j is None:
            _drop("ticker not in the returns panel")
            continue
        i = pos_of_date.get(pd.Timestamp(day0))
        if i is None:
            _drop("event date is not a session in the panel")
            continue
        est_lo, est_hi = i - gap - est_window, i - gap
        if est_lo < 0:
            _drop("insufficient pre-event history")
            continue
        if i + disp_b >= len(idx) or i + disp_a < 0:
            _drop("insufficient data around the event window")
            continue

        y, x = mat[est_lo:est_hi, j], mkt[est_lo:est_hi]
        if not (np.isfinite(y).all() and np.isfinite(x).all()):
            _drop("missing returns in the estimation window")
            continue
        if int(np.count_nonzero(y)) < min_nonzero:
            _drop("thin trading in the estimation window")
            continue

        y_win = mat[i + disp_a: i + disp_b + 1, j]
        x_win = mkt[i + disp_a: i + disp_b + 1]
        if not (np.isfinite(y_win).all() and np.isfinite(x_win).all()):
            _drop("missing returns in the event window")
            continue

        try:
            fit = _market_model(y, x)
        except ValueError:
            _drop("degenerate market model")
            continue

        expected = fit.alpha + fit.beta * x_win
        ar = y_win - expected
        k0, k1 = a - disp_a, b - disp_a + 1          # slice of the display window on test
        car = float(ar[k0:k1].sum())
        sd_car = _car_prediction_sd(fit, x_win[k0:k1])
        if not np.isfinite(sd_car) or sd_car <= 0.0:
            _drop("degenerate forecast-error sd")
            continue

        labels.append(f"{ticker}@{pd.Timestamp(day0).date()}")
        ar_rows.append(ar)
        est_rows.append(fit.resid)
        cars.append(car)
        scars.append(car / sd_car)
        bhars.append(float(np.prod(1.0 + y_win[k0:k1]) - np.prod(1.0 + expected[k0:k1])))
        betas.append(fit.beta)

    if not ar_rows:
        raise ValueError(f"no events survived the filters; attrition = {attrition}")

    ar_df = pd.DataFrame(np.vstack(ar_rows), index=labels, columns=rel_days)
    car_s = pd.Series(cars, index=labels, name="CAR")
    scar_s = pd.Series(scars, index=labels, name="SCAR")
    est_mat = np.vstack(est_rows)
    k0, k1 = a - disp_a, b - disp_a + 1
    win_mat = ar_df.to_numpy()[:, k0:k1]

    t_naive, p_naive = naive_cross_sectional_test(car_s.to_numpy())
    z_pat, p_pat = patell_test(scar_s.to_numpy(), est_window)
    t_bmp, p_bmp = bmp_test(scar_s.to_numpy())
    p_pos = float((est_mat > 0).mean())
    z_sign, p_sign = generalized_sign_test(car_s.to_numpy(), p_pos)
    t_rk_c, p_rk_c = corrado_rank_test_classic(est_mat, win_mat)
    t_rank, p_rank = corrado_rank_test(est_mat, win_mat)

    tests = pd.DataFrame(
        [{"statistic": t_naive, "p_value": p_naive,
          "note": "raw CARs: correctly sized but WEAK"},
         {"statistic": z_pat, "p_value": p_pat,
          "note": "!! estimation-window SE -- OVER-REJECTS"},
         {"statistic": t_bmp, "p_value": p_bmp,
          "note": "<- REPORT THIS (parametric)"},
         {"statistic": z_sign, "p_value": p_sign,
          "note": f"nonparametric, p0={p_pos:.3f} from estimation windows"},
         {"statistic": t_rk_c, "p_value": p_rk_c,
          "note": "!! time-series SE -- OVER-REJECTS"},
         {"statistic": t_rank, "p_value": p_rank,
          "note": "<- REPORT THIS (nonparametric, outlier-robust)"}],
        index=["naive cross-sectional (raw CAR)", "Patell standardised-residual",
               "BMP (standardised)", "generalised sign",
               "Corrado rank (classic SE)", "Corrado rank (cross-sectional SE)"])

    aar = ar_df.mean(axis=0)
    return EventStudyResult(
        ar=ar_df, aar=aar, caar=aar.cumsum(), car=car_s, scar=scar_s,
        bhar=pd.Series(bhars, index=labels, name="BHAR"),
        beta=pd.Series(betas, index=labels, name="beta"),
        tests=tests, window=(a, b), est_window=est_window, gap=gap,
        n_events_in=len(events), attrition=attrition,
    )


# --------------------------------------------------------------------------------------
# offline demo
# --------------------------------------------------------------------------------------
def _synthetic_events(rng: np.random.Generator,
                      n_names: int = 120,
                      n_days: int = 320,
                      true_ar: float = 0.0,
                      frac_high_vol: float = 0.15,
                      high_vol_ratio: float = 7.5,
                      induced: tuple[float, float] = (1.0, 4.0)
                      ) -> tuple[pd.DataFrame, pd.Series, list[tuple[str, pd.Timestamp]]]:
    """A panel with a KNOWN abnormal return and KNOWN event-induced variance.

    Three features are deliberate, and all three are facts about real event samples:
      * idiosyncratic vol is a two-point mixture -- 85% of names at 1.2% daily, 15% at 9%
        (large caps and distressed microcaps in the same study);
      * on days [-1, +1] each firm's idiosyncratic shock is multiplied by k_i ~ U(1, 4):
        event-induced variance, firm-specific, exactly what Patell assumes away;
      * event dates are STAGGERED, so the demo isolates event-induced variance from
        calendar clustering -- which BMP does not fix and this file does not claim to.
    """
    idx = pd.bdate_range("2021-01-04", periods=n_days)
    names = [f"N{i:03d}" for i in range(n_names)]
    mkt = rng.normal(0.0003, 0.010, n_days)
    beta = rng.uniform(0.6, 1.6, n_names)
    hi = rng.random(n_names) < frac_high_vol
    sigma = np.where(hi, 0.012 * high_vol_ratio, 0.012)

    shocks = rng.standard_normal((n_days, n_names)) * sigma
    rets = mkt[:, None] * beta[None, :] + shocks

    k = rng.uniform(induced[0], induced[1], n_names)
    e_idx = 230 + rng.integers(0, 40, n_names)
    events: list[tuple[str, pd.Timestamp]] = []
    for j, nm in enumerate(names):
        i0 = int(e_idx[j])
        sl = slice(i0 - 1, i0 + 2)                       # the [-1, +1] test window
        rets[sl, j] = mkt[sl] * beta[j] + shocks[sl, j] * k[j]
        rets[i0, j] += true_ar
        events.append((nm, idx[i0]))
    return (pd.DataFrame(rets, index=idx, columns=names),
            pd.Series(mkt, index=idx, name="MKT"), events)


def _run(seed: int, true_ar: float) -> EventStudyResult:
    rng = np.random.default_rng(seed)
    rets, mkt, events = _synthetic_events(rng, true_ar=true_ar)
    return event_study(rets, mkt, events, window=(-1, 1), est_window=150, gap=15,
                       pre_days=4, post_days=6, min_nonzero=20)


if __name__ == "__main__":
    TRUE_AR = 0.04

    print("=== A. POWER: 120 events carrying a KNOWN +4.00% abnormal return on day 0 ===")
    res = _run(seed=7, true_ar=TRUE_AR)
    print(res.report(title="SYNTHETIC EARNINGS-STYLE EVENTS (true CAR[-1,+1] = +4.00%)"))
    t_bmp = res.tests.loc["BMP (standardised)", "statistic"]
    t_raw = res.tests.loc["naive cross-sectional (raw CAR)", "statistic"]
    print(f"\n  recovered CAAR[-1,+1] = {res.car.mean():+.3%} against a planted "
          f"{TRUE_AR:+.3%}.")
    print(f"  Both find it, but BMP's t is {t_bmp:.2f} against the raw test's {t_raw:.2f} "
          f"-- {t_bmp / t_raw:.1f}x the")
    print("  statistic on identical data, because standardising stops the ~15% of names "
          "carrying")
    print("  9% daily vol from setting the standard error for all 120 events. That is the "
          "cost")
    print("  of not standardising: not a false positive, a missed true one.")

    print("\n" + "=" * 84)
    print("=== B. SIZE: the same generator with the effect switched OFF (true AR = 0) ===")
    reps = 300
    print(f"Under a true null a 5% test must reject 5% of the time. {reps} replications:")
    names = list(res.tests.index)
    pvals: dict[str, list[float]] = {nm: [] for nm in names}
    for s in range(reps):
        t = _run(seed=10_000 + s, true_ar=0.0).tests
        for nm in names:
            pvals[nm].append(float(t.loc[nm, "p_value"]))

    print("-" * 84)
    print(f"{'test':<34} {'reject @5%':>11} {'reject @1%':>11}   verdict")
    print("-" * 84)
    rates: dict[str, float] = {}
    for nm in names:
        v = np.asarray(pvals[nm], dtype=float)
        r5, r1 = float((v < 0.05).mean()), float((v < 0.01).mean())
        rates[nm] = r5
        verdict = ("OVER-REJECTS - do not use" if r5 > 0.09 else
                   "conservative" if r5 < 0.03 else "correctly sized")
        print(f"{nm:<34} {r5:>10.1%} {r1:>11.1%}   {verdict}")
    print("-" * 84)

    pat = rates["Patell standardised-residual"]
    bmp = rates["BMP (standardised)"]
    print(f"On data containing NO effect whatsoever, the Patell test -- the standardised "
          f"test\nwith an estimation-window standard error -- fires {pat:.0%} of the time "
          f"at a nominal 5%.")
    print(f"BMP, same numerator and a cross-sectional standard error, fires {bmp:.0%}. "
          f"That is\n{pat / max(bmp, 1e-9):.0f}x the false-discovery rate, and it is the "
          f"whole reason this file exists.")
    print("The classic Corrado rank test fails the same way for the same reason; the "
          "cross-sectional")
    print("variant does not. Any denominator taken from the estimation window is wrong "
          "when the")
    print("event itself moves volatility -- which is the definition of an event.")
    print("\nNote what did NOT happen: the raw cross-sectional t-test is correctly sized "
          "here.")
    print("Its failure is power (panel A), not size. Both parametric tests still assume "
          "cross-")
    print("sectional INDEPENDENCE -- cluster the events in calendar time and both go "
          "wrong together.")
