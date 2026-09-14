#!/usr/bin/env python3
"""Validate a delivered bar panel BEFORE computing anything on it.

WHY: a delivered panel can carry plausible-looking defects: a repeated close can be
stale, a missing session has no row to inspect, and inverted OHLC values remain numeric.
Validate against an explicit schema and calendar before calculating signals.

The examples distinguish repeated quotes on real sessions from prices inserted on
non-sessions. Deleting repeated values changes the observation clock; it is not a
universal correction for volatility or beta. Missing observations are reported without
filling or deleting any input. A fill without provenance can look like an observed bar.

Pure numpy / pandas, fixed seed, no network, no file writes.

Usage:
    from data_quality import validate_panel, vol_bias, stale_runs

    report = validate_panel(bars, sessions=nyse_2024)
    print(report.summary())
    if not report.passed:
        raise SystemExit(1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

SEED = 20260910
REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


# --------------------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Check:
    """One check's verdict. `count` is how many rows failed it, 0 meaning it passed."""

    name: str
    severity: str            # "error" | "warning" | "info"
    count: int
    detail: str
    where: tuple[str, ...] = ()

    def __str__(self) -> str:
        head = f"{self.severity.upper():<7} {self.name:<20} {self.count:>6}  {self.detail}"
        return head + (f"  e.g. {', '.join(self.where)}" if self.where else "")


@dataclass
class PanelReport:
    """What `validate_panel` returns. `passed` is True iff no check has severity 'error'."""

    n_rows: int
    columns: tuple[str, ...]
    checks: list[Check] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if c.severity == "error"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def get(self, name: str) -> Check | None:
        return next((c for c in self.checks if c.name == name), None)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"check": c.name, "severity": c.severity, "count": c.count,
                              "detail": c.detail} for c in self.checks])

    def summary(self) -> str:
        head = (f"{'PASS' if self.passed else 'FAIL'}  {self.n_rows} rows x "
                f"{len(self.columns)} columns  "
                f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)")
        return "\n".join([head] + [f"  {c}" for c in self.checks])

    def __str__(self) -> str:
        return self.summary()


def _stamps(index: Iterable, limit: int = 3) -> tuple[str, ...]:
    out = [str(pd.Timestamp(t).date()) for t in list(index)[:limit]]
    return tuple(out)


# --------------------------------------------------------------------------------------
# individual checks, each usable on its own
# --------------------------------------------------------------------------------------
def stale_runs(close: pd.Series) -> pd.DataFrame:
    """Runs of consecutive IDENTICAL closes: start, end, length (rows repeated).

    `length` counts the repeated rows, not the run including its first fresh print, so a
    close printed on Monday and repeated Tuesday and Wednesday is length 2.
    """
    c = pd.Series(close).astype(float)
    same = c.eq(c.shift()).to_numpy()
    if len(same):
        same[0] = False
    rows = []
    i = 1
    while i < len(same):
        if same[i]:
            j = i
            while j < len(same) and same[j]:
                j += 1
            rows.append({"start": c.index[i], "end": c.index[j - 1], "length": j - i})
            i = j
        else:
            i += 1
    return pd.DataFrame(rows, columns=["start", "end", "length"])


def stale_mask(close: pd.Series) -> np.ndarray:
    c = pd.Series(close).astype(float)
    m = c.eq(c.shift()).to_numpy()
    if len(m):
        m[0] = False
    return m


def annualised_vol(close: pd.Series | np.ndarray, periods_per_year: float = 252.0) -> float:
    v = np.asarray(pd.Series(close).astype(float).to_numpy())
    if np.any(~np.isfinite(v)) or np.any(v <= 0):
        return float("nan")
    lr = np.diff(np.log(v))
    if lr.size < 2:
        return float("nan")
    return float(np.std(lr, ddof=1) * np.sqrt(periods_per_year))


def vol_bias(close: pd.Series, periods_per_year: float = 252.0) -> dict:
    """Annualised vol with the stale rows retained vs dropped, and the ratio between them.

    The ratio is what a volatility-target sizer would change its leverage by. It is
    approximately sqrt(N / N_fresh) whatever produced the stale rows - see
    `mechanism_table()` for why that does NOT tell you which figure is the right one.
    """
    c = pd.Series(close).astype(float)
    m = stale_mask(c)
    fresh = c[~m]
    retained = annualised_vol(c, periods_per_year)
    dropped = annualised_vol(fresh, periods_per_year)
    runs = stale_runs(c)
    ratio = dropped / retained if retained > 0 else float("nan")
    return {"n_rows": int(len(c)), "n_stale": int(m.sum()),
            "stale_frac": round(float(m.mean()), 6) if len(m) else 0.0,
            "n_runs": int(len(runs)),
            "longest_run": int(runs["length"].max()) if len(runs) else 0,
            "vol_retained": round(retained, 6), "vol_dropped": round(dropped, 6),
            "ratio_dropped_over_retained": round(ratio, 6),
            "sqrt_n_over_nfresh": round(float(np.sqrt(len(c) / max(len(fresh), 1))), 6),
            "leverage_multiple": round(ratio, 6)}


def beta_bias(close: pd.Series, market_returns: pd.Series) -> dict:
    """Compare contemporaneous betas; dropping repeated closes does not recover true beta."""
    c = pd.Series(close).astype(float)
    frame = pd.DataFrame({"asset": np.log(c.where(c > 0)).diff(),
                          "market": pd.Series(market_returns).astype(float),
                          "stale": pd.Series(stale_mask(c), index=c.index)}).dropna()
    def beta(part):
        if len(part) < 2 or part["market"].var() <= 0:
            return float("nan")
        return float(part["asset"].cov(part["market"]) / part["market"].var())
    observed, fresh = beta(frame), beta(frame[~frame["stale"].astype(bool)])
    return {"beta_observed": round(observed, 6), "beta_on_fresh_rows": round(fresh, 6),
            "fresh_fraction": round(float(1 - frame["stale"].mean()), 6),
            "shrinkage": round(observed / fresh, 6) if fresh else float("nan")}


def _session_dates(values) -> pd.DatetimeIndex:
    """Local date labels; callers convert instants to exchange timezone before passing."""
    result = pd.DatetimeIndex(values)
    if result.hasnans:
        raise ValueError("session labels must not contain NaT")
    return result.tz_localize(None).normalize() if result.tz is not None else result.normalize()


def missing_sessions(index: pd.DatetimeIndex,
                     sessions: Sequence | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Declared sessions with no row in the panel. REPORT these; never fill them."""
    s = _session_dates(sessions)
    have = _session_dates(index)
    return s.difference(have)


def non_session_rows(index: pd.DatetimeIndex,
                     sessions: Sequence | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Rows the panel carries on dates that are not sessions - fabricated bars."""
    s = _session_dates(sessions)
    have = _session_dates(index)
    return have.difference(s)


def ohlc_violations(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """Rows where high < low, or open/close sit outside [low, high]."""
    b = bars
    lo, hi = b["low"].astype(float), b["high"].astype(float)
    op, cl = b["open"].astype(float), b["close"].astype(float)
    bad = (hi < lo) | (op > hi) | (op < lo) | (cl > hi) | (cl < lo)
    return pd.DatetimeIndex(b.index[bad.fillna(False)])


def non_positive_prices(bars: pd.DataFrame,
                        columns: Sequence[str] = PRICE_COLUMNS) -> pd.DatetimeIndex:
    cols = [c for c in columns if c in bars.columns]
    if not cols:
        return pd.DatetimeIndex([])
    bad = (bars[cols].astype(float) <= 0).any(axis=1)
    return pd.DatetimeIndex(bars.index[bad.fillna(False)])


def extreme_returns(close: pd.Series, cap: float = 0.5,
                    sigma: float = 8.0) -> pd.DataFrame:
    """Flag absolute SIMPLE returns over cap or centred robust log-return deviations.

    Invalid prices stay missing: no return is bridged across them. A flag identifies a
    move to investigate, not proof of a vendor error or permission to clip the return.
    """
    c = pd.Series(close).astype(float)
    c = c.where(np.isfinite(c) & (c > 0))
    lr = np.log(c).diff()
    valid = lr.dropna()
    centre = float(valid.median()) if len(valid) else 0.0
    mad = float((valid - centre).abs().median()) if len(valid) else 0.0
    robust_sd = 1.4826 * mad if mad > 0 else float(valid.std(ddof=0))
    simple = c.pct_change(fill_method=None)
    over_cap = simple.abs() > cap
    deviation = (lr - centre).abs()
    over_sigma = deviation > sigma * robust_sd if robust_sd > 0 else deviation > np.inf
    mask = (over_cap | over_sigma).fillna(False)
    hit = lr[mask]
    return pd.DataFrame({"date": hit.index, "log_move": hit.to_numpy(),
                         "pct_move": 100 * simple[mask].to_numpy(),
                         "over_cap": over_cap[mask].to_numpy(),
                         "robust_sigmas": (deviation[mask] / robust_sd).to_numpy()
                         if robust_sd > 0 else np.zeros(len(hit))}).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# the contract
# --------------------------------------------------------------------------------------
def validate_panel(bars: pd.DataFrame,
                   sessions: Sequence | pd.DatetimeIndex | None = None,
                   *, columns: Sequence[str] = REQUIRED_COLUMNS,
                   max_stale_run: int = 3, max_stale_frac: float = 0.10,
                   max_abs_return: float = 0.5, outlier_sigma: float = 8.0,
                   require_volume: bool = True,
                   periods_per_year: float = 252.0) -> PanelReport:
    """Run every check over one instrument's bar panel and return a PanelReport.

    `sessions` is the calendar the panel CLAIMS to be on. Without it the calendar checks
    are skipped and reported as skipped - not as passing, because a missing session cannot
    be seen without knowing which sessions there should be.
    """
    if isinstance(max_stale_run, bool) or not isinstance(max_stale_run, (int, np.integer)) or max_stale_run < 0:
        raise ValueError("max_stale_run must be a nonnegative integer")
    if not np.isfinite(max_stale_frac) or not 0 <= max_stale_frac <= 1:
        raise ValueError("max_stale_frac must lie in [0, 1]")
    for name, value in (("max_abs_return", max_abs_return), ("outlier_sigma", outlier_sigma),
                        ("periods_per_year", periods_per_year)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(bars, pd.DataFrame):
        raise TypeError(f"bars must be a DataFrame, got {type(bars).__name__}")
    rep = PanelReport(n_rows=len(bars), columns=tuple(bars.columns))
    add = rep.checks.append

    if bars.columns.has_duplicates:
        add(Check("schema", "error", int(bars.columns.duplicated().sum()), "duplicate column labels"))
        return rep
    columns = tuple(dict.fromkeys((*PRICE_COLUMNS, *columns)))
    if not require_volume:
        columns = tuple(c for c in columns if c != "volume")
    elif "volume" not in columns:
        columns = (*columns, "volume")
    missing_cols = [c for c in columns if c not in bars.columns]
    add(Check("schema", "error" if missing_cols else "info", len(missing_cols),
              f"missing column(s) {missing_cols}" if missing_cols
              else f"all of {list(columns)} present"))
    if missing_cols:
        return rep
    checked_columns = tuple(c for c in REQUIRED_COLUMNS if c in bars)
    non_numeric = [c for c in checked_columns if not pd.api.types.is_numeric_dtype(bars[c])
                   or pd.api.types.is_bool_dtype(bars[c]) or pd.api.types.is_complex_dtype(bars[c])]
    if non_numeric:
        add(Check("dtypes", "error", len(non_numeric), f"non-numeric column(s) {non_numeric}"))
        return rep

    if not isinstance(bars.index, pd.DatetimeIndex):
        add(Check("index_type", "error", 1,
                  f"index is {type(bars.index).__name__}, not a DatetimeIndex"))
        return rep
    if bars.index.hasnans:
        add(Check("missing_timestamp", "error", int(bars.index.isna().sum()), "NaT in the index"))
        return rep
    if len(bars) < 3:
        add(Check("length", "error", len(bars), "need at least 3 rows to check anything"))
        return rep

    dupes = bars.index[bars.index.duplicated()]
    add(Check("duplicate_rows", "error" if len(dupes) else "info", len(dupes),
              "duplicated timestamps" if len(dupes) else "no duplicate timestamps",
              _stamps(dupes)))
    sorted_ok = bars.index.is_monotonic_increasing
    add(Check("sorted", "info" if sorted_ok else "error", 0 if sorted_ok else 1,
              "index is sorted" if sorted_ok
              else "index is NOT sorted - every rolling window is wrong"))

    nans = {c: int(bars[c].isna().sum()) for c in checked_columns if bars[c].isna().any()}
    infinite = {c: int(np.isinf(bars[c].astype(float)).sum()) for c in checked_columns
                if np.isinf(bars[c].astype(float)).any()}
    add(Check("infinite_values", "error" if infinite else "info", sum(infinite.values()),
              f"infinite values by column {infinite}" if infinite else "no infinite values"))
    add(Check("nan_values", "error" if nans else "info", sum(nans.values()),
              f"NaN by column {nans}" if nans else "no NaN"))

    bad_px = non_positive_prices(bars, [c for c in PRICE_COLUMNS if c in columns])
    add(Check("non_positive_prices", "error" if len(bad_px) else "info", len(bad_px),
              "price <= 0: log returns are undefined" if len(bad_px)
              else "all prices positive", _stamps(bad_px)))

    bad_ohlc = ohlc_violations(bars)
    add(Check("ohlc_ordering", "error" if len(bad_ohlc) else "info", len(bad_ohlc),
              "high < low, or open/close outside [low, high]" if len(bad_ohlc)
              else "high >= max(open, close) >= min(open, close) >= low", _stamps(bad_ohlc)))

    if "volume" in bars.columns:
        negative = bars.index[bars["volume"].astype(float) < 0]
        add(Check("negative_volume", "error" if len(negative) else "info", len(negative),
                  "volume must be nonnegative", _stamps(negative)))
        zero_vol = pd.DatetimeIndex(bars.index[(bars["volume"].astype(float) == 0)
                                               .fillna(False)])
        sev = "warning" if (len(zero_vol) and require_volume) else "info"
        add(Check("zero_volume", sev, len(zero_vol),
                  "sessions with no volume - a halt, a suspension, or a fabricated bar"
                  if len(zero_vol) else "every row traded", _stamps(zero_vol)))

    vb = vol_bias(bars["close"], periods_per_year)
    if not sorted_ok or len(dupes):
        for key in ("vol_retained", "vol_dropped", "ratio_dropped_over_retained", "leverage_multiple"):
            vb[key] = float("nan")
    rep.stats["vol_bias"] = vb
    too_many = vb["stale_frac"] > max_stale_frac
    too_long = vb["longest_run"] > max_stale_run
    add(Check("stale_closes", "error" if (too_many or too_long) else "info",
              vb["n_stale"],
              f"{vb['n_stale']} repeated closes in {vb['n_runs']} runs "
              f"({100 * vb['stale_frac']:.2f} %, longest {vb['longest_run']}); "
              f"annualised vol {vb['vol_retained']:.4f} retained vs "
              f"{vb['vol_dropped']:.4f} dropped, ratio "
              f"{vb['ratio_dropped_over_retained']:.4f}"))

    ext = extreme_returns(bars["close"], max_abs_return, outlier_sigma)
    over_cap = int(ext["over_cap"].sum()) if len(ext) else 0
    add(Check("extreme_returns", "error" if over_cap else ("warning" if len(ext) else "info"),
              len(ext),
              f"{over_cap} past the {100 * max_abs_return:.0f} % cap, "
              f"{len(ext) - over_cap} past {outlier_sigma} robust sigma" if len(ext)
              else "no extreme one-bar move", _stamps(ext["date"]) if len(ext) else ()))

    if sessions is None:
        add(Check("calendar", "warning", 0,
                  "no session index supplied - missing sessions CANNOT be detected; "
                  "pass sessions= from your pinned trading calendar"))
        return rep
    miss = missing_sessions(bars.index, sessions)
    extra = non_session_rows(bars.index, sessions)
    add(Check("missing_sessions", "error" if len(miss) else "info", len(miss),
              "declared sessions absent from the panel - REPORT them, do not fill them"
              if len(miss) else "every declared session is present", _stamps(miss)))
    add(Check("non_session_rows", "error" if len(extra) else "info", len(extra),
              "rows outside the declared calendar; verify venue, timezone and session labels" if len(extra) else "no row outside the calendar",
              _stamps(extra)))
    rep.stats["n_sessions_declared"] = int(len(pd.DatetimeIndex(pd.to_datetime(
        list(sessions)))))
    return rep


# --------------------------------------------------------------------------------------
# synthetic panels
# --------------------------------------------------------------------------------------
def _run_mask(n: int, frac: float, mean_run: float, rng: np.random.Generator) -> np.ndarray:
    """Stale-row mask with geometric run lengths, targeting `frac` of the rows."""
    mask = np.zeros(n, dtype=bool)
    target = int(round(frac * n))
    placed, i = 0, 1
    while placed < target and i < n - 1:
        if rng.random() < frac:
            run = int(min(max(1, rng.geometric(1.0 / mean_run)), n - 1 - i))
            mask[i:i + run] = True
            placed += run
            i += run + 1
        else:
            i += 1
    return mask


def clean_panel(n_sessions: int = 504, annual_vol: float = 0.32, s0: float = 40.0,
                seed: int = SEED) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """A well-formed OHLCV panel on a declared session index. Nothing wrong with it."""
    rng = np.random.default_rng(seed)
    sessions = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=n_sessions))
    sig = annual_vol / np.sqrt(252.0)
    close = s0 * np.exp(np.cumsum(rng.normal(0.0, sig, n_sessions)))
    op = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0.0, sig / 3, n_sessions))
    span = np.abs(rng.normal(0.0, sig, n_sessions))
    high = np.maximum(op, close) * (1.0 + span)
    low = np.minimum(op, close) * (1.0 - span)
    vol = rng.lognormal(12.0, 0.6, n_sessions).round()
    return sessions, pd.DataFrame({"open": op, "high": high, "low": low, "close": close,
                                   "volume": vol}, index=sessions)


def stale_series(close: np.ndarray, mask: np.ndarray, mechanism: str) -> np.ndarray:
    """Return the same carried-forward quotes under two session interpretations.

    'deferred' interprets all rows as sessions with unobserved prices on masked rows.
    'fabricated' interprets masked rows as inserted non-sessions; its reference clock
    contains only unmasked quotes. The identical outputs do not identify which model holds.
    """
    out = np.asarray(close, dtype=float).copy()
    if mechanism == "deferred":
        for i in range(1, len(out)):
            if mask[i]:
                out[i] = out[i - 1]
        return out
    if mechanism == "fabricated":
        fresh = np.asarray(close, dtype=float)[~mask]
        j = -1
        for i in range(len(out)):
            if not mask[i]:
                j += 1
                out[i] = fresh[j]
            else:
                out[i] = fresh[max(j, 0)]
        return out
    raise ValueError("mechanism must be 'deferred' or 'fabricated'")


def mechanism_table(n: int = 1008, stale_frac: float = 0.35, mean_run: float = 2.4,
                    annual_vol: float = 0.32, seed: int = SEED) -> pd.DataFrame:
    """Identical delivered values under two different session-clock interpretations."""
    rng = np.random.default_rng(seed)
    sig = annual_vol / np.sqrt(252.0)
    mask = _run_mask(n, stale_frac, mean_run, rng)
    latent = 100.0 * np.exp(np.cumsum(rng.normal(0.0, sig, n)))
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=n))
    rows = []
    for mech in ("deferred", "fabricated"):
        obs = pd.Series(stale_series(latent, mask, mech), index=idx)
        truth = (pd.Series(latent, index=idx) if mech == "deferred"
                 else pd.Series(latent[~mask], index=idx[~mask]))
        vb = vol_bias(obs, 252.0)
        rows.append({
            "mechanism": mech, "stale_frac": vb["stale_frac"], "n_runs": vb["n_runs"],
            "longest_run": vb["longest_run"],
            "vol_truth": round(annualised_vol(truth), 4),
            "vol_retained": round(vb["vol_retained"], 4),
            "vol_dropped": round(vb["vol_dropped"], 4),
            "dropped_over_retained": round(vb["ratio_dropped_over_retained"], 4),
            "retained_over_truth": round(vb["vol_retained"] / annualised_vol(truth), 4),
            "dropped_over_truth": round(vb["vol_dropped"] / annualised_vol(truth), 4),
            "observation_clock": "real sessions" if mech == "deferred" else "non-session rows inserted"})
    return pd.DataFrame(rows)


def beta_demo(n: int = 1008, stale_frac: float = 0.35, mean_run: float = 2.4,
              beta: float = 1.0, annual_vol: float = 0.32, seed: int = SEED) -> dict:
    """A no-print name's measured beta, against the fresh fraction that explains it."""
    rng = np.random.default_rng(seed + 7)
    sig = annual_vol / np.sqrt(252.0)
    mask = _run_mask(n, stale_frac, mean_run, rng)
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=n))
    mkt_r = pd.Series(rng.normal(0.0, sig, n), index=idx)
    latent = 100.0 * np.exp(np.cumsum(beta * mkt_r.to_numpy()
                                      + rng.normal(0.0, sig * 0.6, n)))
    obs = pd.Series(stale_series(latent, mask, "deferred"), index=idx)
    out = beta_bias(obs, mkt_r)
    truth = pd.Series(latent, index=idx)
    tr = np.log(truth).diff().dropna()
    out["beta_true"] = round(float(np.cov(tr, mkt_r.reindex(tr.index))[0, 1]
                                   / np.var(mkt_r.reindex(tr.index), ddof=1)), 6)
    out["observed_over_true"] = round(out["beta_observed"] / out["beta_true"], 6)
    out["target_beta"] = beta
    return out


def fill_cost(n_sessions: int = 504, n_missing: int = 24, window: int = 20,
              seed: int = SEED) -> dict:
    """What FILLING the holes costs, against reporting them. Both numbers, one call."""
    sessions, panel = clean_panel(n_sessions, seed=seed)
    rng = np.random.default_rng(seed + 3)
    drop = pd.DatetimeIndex(rng.choice(sessions[window:], size=n_missing, replace=False))
    delivered = panel.drop(index=drop.sort_values())
    filled = delivered.reindex(sessions).ffill()

    truth_sma = panel["close"].rolling(window).mean()
    filled_sma = filled["close"].rolling(window).mean()
    diff = (filled_sma - truth_sma).abs() / truth_sma
    moved = int((diff > 1e-4).sum())

    vb_filled = vol_bias(filled["close"])
    return {"n_sessions": n_sessions, "n_missing": n_missing,
            "reported": n_missing, "runs_created_by_the_fill": vb_filled["n_runs"],
            "stale_frac_after_fill": vb_filled["stale_frac"],
            "vol_truth": round(annualised_vol(panel["close"]), 6),
            "vol_after_fill": vb_filled["vol_retained"],
            "vol_ratio": round(vb_filled["vol_retained"]
                               / annualised_vol(panel["close"]), 6),
            "window": window, "sma_rows_moved": moved,
            "sma_rows_moved_pct": round(100.0 * moved / len(sessions), 2),
            "max_sma_error_bp": round(float(diff.max() * 1e4), 2)}


def delivered_panel(n_sessions: int = 504, seed: int = SEED) -> tuple[pd.DatetimeIndex,
                                                                     pd.DataFrame]:
    """One panel carrying every defect this module checks for, seeded and reproducible."""
    sessions, panel = clean_panel(n_sessions, seed=seed)
    rng = np.random.default_rng(seed + 11)
    bars = panel.copy()

    mask = _run_mask(len(bars), 0.12, 2.2, rng)             # repeated closes
    bars.loc[:, "close"] = stale_series(bars["close"].to_numpy(), mask, "deferred")
    bars.loc[mask, "volume"] = 0.0                          # ... with no volume
    bars.iloc[40, bars.columns.get_loc("close")] = -1.0     # a negative price
    hi = bars.columns.get_loc("high")
    lo = bars.columns.get_loc("low")
    bars.iloc[77, hi], bars.iloc[77, lo] = bars.iloc[77, lo], bars.iloc[77, hi]
    bars.iloc[150, bars.columns.get_loc("close")] *= 3.0    # a fat-finger print
    drop = sessions[[200, 201, 202, 300, 301, 400, 401, 402, 403]]
    bars = bars.drop(index=drop)                            # missing sessions
    weekend = pd.DatetimeIndex([d + pd.Timedelta(days=(5 - d.weekday()) % 7)
                                for d in (sessions[100], sessions[250])])
    extra = bars.reindex(bars.index.append(weekend)).sort_index().ffill()
    extra = pd.concat([extra, extra.iloc[[5]]]).sort_index()   # a duplicated timestamp
    return sessions, extra


# --------------------------------------------------------------------------------------
def _fmt(x) -> str:
    return (x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)).to_string(index=False)


if __name__ == "__main__":
    print("=== 1. a repeated close is a zero-return day, and it moves realised vol ===")
    mt = mechanism_table()
    print(_fmt(mt))
    a, b = mt.iloc[0], mt.iloc[1]
    print(f"    the OBSERVABLE is identical: {int(a['n_runs'])} runs of repeated closes, "
          f"longest {int(a['longest_run'])},")
    print(f"    {100 * a['stale_frac']:.2f} % of rows, and dropped/retained is "
          f"{a['dropped_over_retained']:.4f} vs {b['dropped_over_retained']:.4f} - the "
          f"same number.")
    print(f"    With DEFERRED prints, retained/truth is {a['retained_over_truth']:.4f};")
    print("    neither deleting rows nor keeping stale prices recovers every latent return.")
    print(f"    When the extra rows are NON-SESSIONS, retained is only "
          f"{b['retained_over_truth']:.4f} of it.")
    print(f"    An unconstrained fixed vol target changes size by {b['dropped_over_retained']:.4f}x.")
    print("    A calendar identifies non-session rows; it cannot recover a missing price.")

    print("\n=== 2. the same staleness pulls beta toward zero ===")
    bd = beta_demo()
    print(f"    true beta {bd['beta_true']:.4f}, measured on the delivered series "
          f"{bd['beta_observed']:.4f}")
    print(f"    ratio {bd['observed_over_true']:.4f} against a fresh fraction of "
          f"{bd['fresh_fraction']:.4f}")
    print(f"    fresh-row beta is {bd['beta_on_fresh_rows']:.4f}; filtering is not a general cure.")

    print("\n=== 3. a missing session is REPORTED, never filled ===")
    fc = fill_cost()
    print(f"    {fc['n_missing']} of {fc['n_sessions']} declared sessions absent from the "
          f"delivery.")
    print(f"    reported : {fc['reported']} dates, and nothing else changes.")
    print(f"    filled   : {fc['runs_created_by_the_fill']} runs of repeated closes "
          f"({100 * fc['stale_frac_after_fill']:.2f} % of rows),")
    print(f"               annualised vol {fc['vol_after_fill']:.4f} against a true "
          f"{fc['vol_truth']:.4f} - a ratio of {fc['vol_ratio']:.4f},")
    print(f"               and the {fc['window']}-day moving average moves on "
          f"{fc['sma_rows_moved']} of {fc['n_sessions']} sessions "
          f"({fc['sma_rows_moved_pct']} %),")
    print(f"               by up to {fc['max_sma_error_bp']} bp.")
    print("    Without a provenance flag, a filled row looks like a real one - it has a")
    print("    timestamp, a price and a volume. The report is the only place the hole "
          "survives.")

    print("\n=== 4. the contract: one call, one report, one boolean ===")
    sessions, clean = clean_panel()
    ok = validate_panel(clean, sessions=sessions)
    print(ok.summary())
    print()
    sessions, bad = delivered_panel()
    rep = validate_panel(bad, sessions=sessions)
    print(rep.summary())
    print(f"\n    passed={rep.passed}, {len(rep.errors)} error(s): "
          f"{[c.name for c in rep.errors]}")
    print("    Each line is a count, not an adjective. Nothing here was repaired.")

    print("\nThe rule: validate the panel before you compute on it - a repeated close, a "
          "missing session and an inverted high/low all look like data, so they must be "
          "COUNTED and REPORTED, never quietly filled.")
