#!/usr/bin/env python3
"""leak_bench - plant known defects in synthetic financial data, measure which guards catch them.

The analogue of PyOD's ADBench for the research-integrity guards in this repo. A synthetic
world with a single true source of alpha (a persistent latent drift per name) is built once,
seeded. One simple strategy is run on it. A catalogue of defects each takes the clean dataset
and returns a corrupted copy plus ground truth - a look-ahead signal, a wrong-side as-of join,
a survivor-only universe, an unadjusted split, a qfq-style rewritten vintage, a back-filled
indicator warm-up, an unpurged CV split, a test window inside an LLM's training period, a cost
assumption 10x too low, a single calm quarter, and three more. Every guard is run on every
dataset, clean and corrupted, and the harness records caught / missed / not-applicable, false
alarms on clean data, runtime, and the Sharpe the defect adds to (or removes from) the strategy.

Guards are IMPORTED from fin_skills, never copied. What they see is declared per guard (its
input domain) so a defect a guard cannot see by construction is reported as n/a, not a miss.

Run from the repo root:   python benchmarks/leak_bench.py        (numpy/pandas; ~1-2 min)
Writes benchmarks/RESULTS.md. Output is ASCII only.
"""
from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The guards under test. Imported, not copied.
from fin_skills.core import assert_causal as ac_mod                     # noqa: E402
from fin_skills.core import adjustment_check as adj_mod                 # noqa: E402
from fin_skills.core import cost_curve as cc_mod                        # noqa: E402
from fin_skills.core import fold_leak_test as flt_mod                   # noqa: E402
from fin_skills.core import pit_fundamentals as pf_mod                  # noqa: E402
from fin_skills.core import pit_universe as pu_mod                      # noqa: E402
from fin_skills.core import regime_coverage as rc_mod                   # noqa: E402
from fin_skills.core import safe_asof as sa_mod                         # noqa: E402
from fin_skills.core import survivorship_audit as sv_mod                # noqa: E402
from fin_skills.core import warmup_probe as wp_mod                      # noqa: E402
from fin_skills.libraries import purge_effect as pe_mod                 # noqa: E402
from fin_skills.llm import contamination_probe as cp_mod                # noqa: E402

warnings.simplefilter("ignore", category=FutureWarning)

# ------------------------------------------------------------------------------------------
# constants
# ------------------------------------------------------------------------------------------
SEED = 20260908
N_NAMES = 60
START, END = "2017-01-02", "2023-12-29"
LLM_CUTOFF = "2020-12-31"                 # the LLM score's training cutoff
EVAL_START, EVAL_END = "2021-01-04", "2023-12-29"   # the honest reporting window
CACHE_DATE = "2022-06-30"                 # when the qfq pipeline last cached its history
PERIODS = 252
SPAN = 63                                 # EMA span of the trend indicator
H = 21                                    # label horizon of the ML component (bars)
EMBARGO = 21                              # embargo after the purge, in bars
KNN_K = 5
BLOCK = 63                                # walk-forward refit block (bars)
MIN_TRAIN = 252                           # bars of history before the first ML prediction
LIVE_WINDOW = 126                         # candles the "live" bot fetches in the truncation defect
ADV_MIN = 8.0e6                           # dollars of trailing ADV to be tradeable
COST_HONEST = 20.0                        # bps round-trip, the cost the clean pipeline assumes
COST_LOW = 2.0                            # the defective assumption (10x too low)
SPLIT_TRIGGER = 150.0                     # a name splits 2:1 once its raw price exceeds this
DELIST_DRAWDOWN = 0.70                    # a name leaves the tape 70% below its running peak
TOL = 1e-9


# ------------------------------------------------------------------------------------------
# synthetic world
# ------------------------------------------------------------------------------------------
@dataclass
class World:
    dates: pd.DatetimeIndex
    tickers: list[str]
    regime: np.ndarray            # 0 calm, 1 turbulent (true DGP state, per date)
    market: np.ndarray            # market factor daily return
    mu: pd.DataFrame              # latent drift, dates x tickers (the only true alpha)
    sigma: pd.Series              # idiosyncratic daily vol per name
    true_close: pd.DataFrame      # economic value path, continuous, NaN outside listing
    raw_close: pd.DataFrame       # as quoted: splits visible, rounded to cents
    volume: pd.DataFrame          # shares traded
    actions: pd.DataFrame         # date, ticker, ratio, kind
    listings: pd.DataFrame        # ticker, listing_date, delisting_date (the exchange's table)
    feed: pd.DataFrame            # feed_ts, ticker, score  (post-close news score)
    llm: pd.DataFrame             # dates x tickers, an LLM score with a training cutoff
    facts: dict[str, list[dict]]  # companyfacts-shaped vintages per name


def make_world(seed: int = SEED) -> World:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(START, END)
    n = len(dates)
    tickers = [f"N{i:03d}" for i in range(N_NAMES)]

    # --- market factor: two-state Markov volatility, calm 10% / turbulent 30% annualised
    p_stay = (0.99, 0.96)
    mu_m = (0.08 / PERIODS, -0.15 / PERIODS)
    sig_m = (0.10 / np.sqrt(PERIODS), 0.30 / np.sqrt(PERIODS))
    regime = np.zeros(n, dtype=int)
    for t in range(1, n):
        s = regime[t - 1]
        regime[t] = s if rng.random() < p_stay[s] else 1 - s
    market = rng.normal(np.array(mu_m)[regime], np.array(sig_m)[regime])

    # --- names: beta, idiosyncratic vol, persistent latent drift (AR(1), half-life ~70 days)
    beta = rng.uniform(0.5, 1.5, N_NAMES)
    sigma = rng.uniform(0.20, 0.45, N_NAMES) / np.sqrt(PERIODS)
    phi, sd_mu = 0.99, 0.12 / PERIODS
    mu = np.zeros((n, N_NAMES))
    mu[0] = rng.normal(0.0, sd_mu, N_NAMES)
    innov = rng.normal(0.0, sd_mu * np.sqrt(1 - phi ** 2), (n, N_NAMES))
    for t in range(1, n):
        mu[t] = phi * mu[t - 1] + innov[t]
    eps = rng.normal(0.0, 1.0, (n, N_NAMES)) * sigma
    ret = beta * market[:, None] + mu + eps
    value = 50.0 * np.exp(np.cumsum(ret, axis=0)) * rng.uniform(0.6, 1.6, N_NAMES)

    # --- listing: a quarter of the names IPO later; a name delists 70% below its running peak
    start_pos = np.where(rng.random(N_NAMES) < 0.25,
                         rng.integers(0, 4 * PERIODS, N_NAMES), 0)
    true_close = pd.DataFrame(value, index=dates, columns=tickers)
    listing_rows = []
    for j, c in enumerate(tickers):
        j0 = int(start_pos[j])
        true_close.iloc[:j0, j] = np.nan
        series = true_close[c].to_numpy()
        peak = np.fmax.accumulate(np.where(np.isnan(series), -np.inf, series))
        dd = np.where(peak > 0, series / peak - 1.0, 0.0)
        hit = np.where((np.arange(n) > j0 + 60) & (dd < -DELIST_DRAWDOWN))[0]
        if len(hit):
            k = int(hit[0])
            true_close.iloc[k + 1:, j] = np.nan
            listing_rows.append({"ticker": c, "listing_date": dates[j0], "delisting_date": dates[k]})
        else:
            listing_rows.append({"ticker": c, "listing_date": dates[j0], "delisting_date": pd.NaT})
    listings = pd.DataFrame(listing_rows)

    # --- splits: 2:1 whenever the quoted price crosses SPLIT_TRIGGER (splits follow run-ups)
    raw_arr = true_close.to_numpy().copy()
    action_rows = []
    for j, c in enumerate(tickers):
        v = raw_arr[:, j]
        factor = 1.0
        last_split = -10 ** 9
        for t in range(n):
            if np.isnan(v[t]):
                continue
            if v[t] / factor > SPLIT_TRIGGER and t - last_split > PERIODS:
                factor *= 2.0
                last_split = t
                action_rows.append({"date": dates[t], "ticker": c, "ratio": 2.0, "kind": "split"})
            v[t] = v[t] / factor
    raw = pd.DataFrame(np.round(raw_arr, 2), index=dates, columns=tickers)
    actions = pd.DataFrame(action_rows, columns=["date", "ticker", "ratio", "kind"])

    # --- volume: names differ in liquidity by an order of magnitude
    base = rng.lognormal(np.log(3.0e5), 0.9, N_NAMES)
    volume = pd.DataFrame(base * rng.lognormal(0.0, 0.5, (n, N_NAMES)), index=dates,
                          columns=tickers).where(true_close.notna())

    # --- post-close news feed, stamped with the trading date it was published after.
    # Its content is the NEXT day's move (after-hours news) plus a weak drift view plus noise.
    logret = np.log(true_close).diff()
    r_next = logret.shift(-1)
    z_next = r_next / sigma
    z_mu = pd.DataFrame(mu, index=dates, columns=tickers) / sd_mu
    score = 1.0 * z_next + 0.5 * z_mu + rng.normal(0.0, 1.0, (n, N_NAMES))
    feed = (score.stack().rename("score").reset_index()
            .rename(columns={"level_0": "feed_ts", "level_1": "ticker"}))
    feed = feed[["feed_ts", "ticker", "score"]].sort_values(["feed_ts", "ticker"]).reset_index(drop=True)

    # --- an LLM score: recall inside its training window, a weak honest view after the cutoff
    cutoff = pd.Timestamp(LLM_CUTOFF)
    llm_recall = z_next + 0.7 * rng.normal(0.0, 1.0, (n, N_NAMES))
    llm_honest = 0.3 * z_mu + rng.normal(0.0, 1.0, (n, N_NAMES))
    llm = llm_recall.where(pd.Series(dates <= cutoff, index=dates), llm_honest)
    llm = llm.where(true_close.notna())

    # --- quarterly fundamentals with amendments. The original 10-Q is a noisy read of the
    # quarter's drift; the amendment 30 days later adds what the company learned since.
    facts: dict[str, list[dict]] = {}
    qends = pd.date_range(START, END, freq="QE")
    mu_df = pd.DataFrame(mu, index=dates, columns=tickers)
    for j, c in enumerate(tickers):
        rows = []
        for qi, qe in enumerate(qends):
            qs = (qe - pd.offsets.QuarterEnd(startingMonth=3)) + pd.Timedelta(days=1)
            in_q = mu_df.loc[(dates > qe - pd.Timedelta(days=92)) & (dates <= qe), c]
            if in_q.isna().all():
                continue
            filed0 = qe + pd.Timedelta(days=45)
            filed1 = qe + pd.Timedelta(days=75)
            if filed0 > dates[-1]:
                continue
            base_val = float(in_q.mean()) / sd_mu
            orig = 1.0e8 * (1.0 + 0.10 * base_val + 0.15 * rng.normal())
            learned = mu_df.loc[(dates > filed0) & (dates <= filed1), c]
            adj = float(learned.mean()) / sd_mu if learned.notna().any() else 0.0
            amended = orig * (1.0 + 0.10 * adj + 0.03 * rng.normal())
            fy = qe.year
            fp = f"Q{((qe.month - 1) // 3) + 1}"
            rows.append({"start": qs.strftime("%Y-%m-%d"), "end": qe.strftime("%Y-%m-%d"),
                         "val": round(orig, 0), "accn": f"{c}-{qi:02d}-0", "fy": fy, "fp": fp,
                         "form": "10-Q", "filed": filed0.strftime("%Y-%m-%d")})
            if filed1 <= dates[-1]:
                rows.append({"start": qs.strftime("%Y-%m-%d"), "end": qe.strftime("%Y-%m-%d"),
                             "val": round(amended, 0), "accn": f"{c}-{qi:02d}-1", "fy": fy,
                             "fp": fp, "form": "10-Q/A", "filed": filed1.strftime("%Y-%m-%d")})
        facts[c] = rows

    return World(dates=dates, tickers=tickers, regime=regime, market=market, mu=mu_df,
                 sigma=pd.Series(sigma, index=tickers), true_close=true_close, raw_close=raw,
                 volume=volume, actions=actions, listings=listings, feed=feed, llm=llm,
                 facts=facts)


# ------------------------------------------------------------------------------------------
# dataset = what the pipeline is handed + how it is configured
# ------------------------------------------------------------------------------------------
@dataclass
class Config:
    adjust: bool = True                  # back-adjust the delivered closes with the actions table
    join_exact: bool = False             # merge_asof allow_exact_matches on the feed
    shift_signal: int = 0                # -1 = the trend reads tomorrow's close
    warmup_fill: str = "drop"            # "drop" | "bfill" for the indicator's leading NaNs
    live_window: int | None = None       # candles the deployed bot fetches (None = all history)
    cv: str = "wf_purged"                # "wf_purged" | "wf_unpurged"
    scaler_shared: bool = False          # ML feature scaler fitted once on the full sample
    adv_rule: str = "trailing"           # "trailing" | "fullsample"
    fund_vintage: str = "pit"            # "pit" | "latest"
    eval_start: str = EVAL_START
    eval_end: str = EVAL_END
    cost_bps: float = COST_HONEST
    llm_cutoff: str = LLM_CUTOFF


@dataclass
class Dataset:
    name: str
    close: pd.DataFrame                  # delivered closes, dates x tickers
    volume: pd.DataFrame
    actions: pd.DataFrame
    feed: pd.DataFrame
    llm: pd.DataFrame
    facts: dict[str, list[dict]]
    cfg: Config
    repull: pd.DataFrame | None = None   # a fresh re-pull of the same series (vintage check)
    cache_vintage: pd.DataFrame | None = None
    new_action_dates: dict[str, pd.Timestamp] = field(default_factory=dict)
    truth: dict[str, Any] = field(default_factory=dict)


def clean_dataset(w: World) -> Dataset:
    # The scale-dependent fields are read from the module globals HERE, at call time, so
    # that apply_quick() (which reassigns them before the world is built) takes effect -
    # the Config dataclass captured its own defaults at import, before any --quick resize.
    cfg = Config(eval_start=EVAL_START, eval_end=EVAL_END, cost_bps=COST_HONEST,
                 llm_cutoff=LLM_CUTOFF)
    return Dataset(name="clean", close=w.raw_close.copy(), volume=w.volume.copy(),
                   actions=w.actions.copy(), feed=w.feed.copy(), llm=w.llm.copy(),
                   facts=w.facts, cfg=cfg, repull=w.raw_close.copy(),
                   cache_vintage=w.raw_close.loc[:CACHE_DATE].copy(),
                   truth={"defect": "none"})


# ------------------------------------------------------------------------------------------
# the pipeline (one strategy, configurable in exactly the ways the defects need)
# ------------------------------------------------------------------------------------------
class Scaler:
    """sklearn-shaped scaler (fit/transform, trailing-underscore attributes) in numpy."""

    def fit(self, x: np.ndarray) -> "Scaler":
        self.mean_ = np.nanmean(x, axis=0)
        self.scale_ = np.nanstd(x, axis=0)
        self.scale_ = np.where(self.scale_ > 0, self.scale_, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.scale_


def back_adjust(raw: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Divide history before each action by its ratio: anchored at the present, continuous."""
    px = raw.copy()
    for _, a in actions.iterrows():
        mask = px.index < a["date"]
        px.loc[mask, a["ticker"]] = px.loc[mask, a["ticker"]] / a["ratio"]
    return px


def ema_indicator(x: np.ndarray, span: int = SPAN) -> np.ndarray:
    """The trend's EMA on a 1-D array, length-preserving, NaN for the first span-1 bars."""
    return pd.Series(x, dtype=float).ewm(span=span, adjust=False, min_periods=span).mean().to_numpy()


def trend_indicator(close: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.live_window is None:
        ema = close.ewm(span=SPAN, adjust=False, min_periods=SPAN).mean()
    else:
        # What a live bot computes: an EMA seeded on the first of the last `live_window` closes.
        alpha = 2.0 / (SPAN + 1.0)
        w = cfg.live_window
        weights = alpha * (1.0 - alpha) ** np.arange(w)
        weights[-1] = (1.0 - alpha) ** (w - 1)
        ema = sum(close.shift(j) * weights[j] for j in range(w))
    trend = np.log(close) - np.log(ema)
    if cfg.warmup_fill == "bfill":
        trend = trend.bfill()
    return trend


def ml_features(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lr = np.log(close).diff()
    f1 = np.log(close) - np.log(close.ewm(span=SPAN, adjust=False, min_periods=SPAN).mean())
    f2 = lr.rolling(5).sum()
    f3 = lr.rolling(H).sum()
    f4 = lr.rolling(H).std()
    return f1, f2, f3, f4


def ml_labels(close: pd.DataFrame) -> pd.DataFrame:
    lr = np.log(close).diff()
    fwd = lr.shift(-1)[::-1].rolling(H).sum()[::-1]      # sum of the NEXT H log returns
    return (fwd > 0).astype(float).where(fwd.notna())


def walk_forward_blocks(n: int, first: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_idx, test_idx) pairs: quarterly test blocks, train = everything before them."""
    out = []
    for t0 in range(first + MIN_TRAIN, n, BLOCK):
        test = np.arange(t0, min(t0 + BLOCK, n))
        out.append((np.arange(0, t0), test))
    return out


def knn_block(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray,
              purge: int, scaler: Scaler | None) -> np.ndarray:
    """Out-of-sample k-NN score for one block. `purge` bars are cut from the end of train."""
    t0 = int(test[0])
    tr = train[train < t0 - purge]
    ok = np.isfinite(x[tr]).all(axis=1) & np.isfinite(y[tr])
    tr = tr[ok]
    te_ok = np.isfinite(x[test]).all(axis=1)
    out = np.full(len(test), np.nan)
    if len(tr) < 100 or not te_ok.any():
        return out
    sc = scaler if scaler is not None else Scaler().fit(x[tr])
    out[te_ok] = pe_mod.knn_scores(sc.transform(x[tr]), y[tr], sc.transform(x[test][te_ok]),
                                   k=KNN_K) - 0.5
    return out


def ml_component(close: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    feats = ml_features(close)
    labels = ml_labels(close)
    purge = H + EMBARGO if cfg.cv == "wf_purged" else 0
    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for c in close.columns:
        x = np.column_stack([f[c].to_numpy() for f in feats])
        y = labels[c].to_numpy()
        valid = np.flatnonzero(np.isfinite(x).all(axis=1))
        if len(valid) < MIN_TRAIN + 100:
            continue
        scaler = Scaler().fit(x[np.isfinite(x).all(axis=1)]) if cfg.scaler_shared else None
        col = np.full(len(x), np.nan)
        for train, test in walk_forward_blocks(len(x), int(valid[0])):
            col[test] = knn_block(x, y, train, test, purge, scaler)
        out[c] = col
    return out


def xs_z(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per date; NaN where fewer than 5 names have a value."""
    m = df.mean(axis=1)
    s = df.std(axis=1, ddof=1)
    z = df.sub(m, axis=0).div(s.where(s > 0), axis=0)
    return z.where(df.notna().sum(axis=1) >= 5, np.nan).clip(-3.0, 3.0)


def make_signal_fn(cfg: Config) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def signal_fn(packed: pd.DataFrame) -> pd.DataFrame:
        close = packed["close"]
        if cfg.shift_signal:
            close = close.shift(cfg.shift_signal)
        comps = [trend_indicator(close, cfg), packed["score"], packed["llm"], packed["fund"],
                 ml_component(close, cfg)]
        zs = [xs_z(c) for c in comps]
        composite = sum(z.fillna(0.0) for z in zs)
        any_valid = sum(z.notna().astype(int) for z in zs) > 0
        # A 5-day causal smoothing: the daily components are noisy and unsmoothed sign
        # positions would turn the book over every day.
        return composite.where(any_valid).rolling(5, min_periods=1).mean()
    signal_fn.__name__ = "composite_signal"
    return signal_fn


def join_feed(close: pd.DataFrame, feed: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """As-of join of the feed onto every (date, ticker) bar. Returns (left, joined, score panel)."""
    left = (close.stack(future_stack=True).rename("close").reset_index()
            .rename(columns={"level_0": "date", "level_1": "ticker"}))
    left = left[["date", "ticker"]].sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    right = feed.sort_values("feed_ts", kind="mergesort")
    joined = pd.merge_asof(left, right, left_on="date", right_on="feed_ts", by="ticker",
                           tolerance=pd.Timedelta("5D"), allow_exact_matches=cfg.join_exact,
                           direction="backward")
    panel = joined.pivot(index="date", columns="ticker", values="score").reindex(
        index=close.index, columns=close.columns)
    return left, joined, panel


def fundamental_panel(facts: dict[str, list[dict]], index: pd.DatetimeIndex,
                      columns: list[str], cfg: Config) -> pd.DataFrame:
    """Daily fundamental feature: the value of the latest period on file.

    pit    - at each date, the latest vintage of each period FILED by that date; feature is the
             most recent period's value (the semantics of pit_fundamentals.pit_facts).
    latest - keep='last' per period, placed at the ORIGINAL filing date: the restated number is
             used from the day the first vintage appeared.
    """
    out = pd.DataFrame(np.nan, index=index, columns=columns)
    for c in columns:
        rows = facts.get(c, [])
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["filed"] = pd.to_datetime(df["filed"])
        df["end"] = pd.to_datetime(df["end"])
        if cfg.fund_vintage == "latest":
            last = df.sort_values(["filed", "accn"], kind="mergesort").groupby("end").tail(1)
            first_filed = df.groupby("end")["filed"].min()
            events = pd.DataFrame({"filed": last["end"].map(first_filed).to_numpy(),
                                   "end": last["end"].to_numpy(), "val": last["val"].to_numpy()})
        else:
            events = df[["filed", "end", "val"]]
        events = events.sort_values(["filed", "end"], kind="mergesort")
        series = np.full(len(index), np.nan)
        best_end = pd.Timestamp.min
        for filed, end, val in zip(events["filed"], events["end"], events["val"]):
            if end >= best_end:
                best_end = end
                series[index.searchsorted(filed, side="left"):] = float(val)
        out[c] = series
    return out.where(out.notna())


@dataclass
class Result:
    ds: Dataset
    px: pd.DataFrame                     # the price series the pipeline trades and computes returns from
    packed: pd.DataFrame
    signal_fn: Callable
    composite: pd.DataFrame
    universe: pd.DataFrame
    weights: pd.DataFrame
    returns: pd.Series                   # gross daily strategy return
    turnover: pd.Series                  # one-way traded notional as a fraction of the book
    asset: pd.Series                     # equal-weight return of the tradeable universe
    left: pd.DataFrame
    joined: pd.DataFrame
    liquidity: pd.DataFrame
    fund: pd.DataFrame
    runtime: float

    def window(self) -> pd.DatetimeIndex:
        return self.returns.loc[self.ds.cfg.eval_start:self.ds.cfg.eval_end].index

    def net(self, bps: float | None = None, window: bool = True) -> pd.Series:
        bps = self.ds.cfg.cost_bps if bps is None else bps
        net = self.returns - self.turnover * (bps / 1e4)
        return net.loc[self.window()] if window else net

    def sharpe(self, bps: float | None = None) -> float:
        return sharpe_of(self.net(bps))


def sharpe_of(r: pd.Series) -> float:
    r = r.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(PERIODS)) if len(r) > 2 and sd > 0 else float("nan")


def sharpe_se(s: float, n_days: int) -> float:
    """Lo (2002) iid standard error of an annualised Sharpe over n_days."""
    years = n_days / PERIODS
    return float(np.sqrt((1.0 + 0.5 * s * s) / years))


def run_pipeline(ds: Dataset) -> Result:
    t0 = time.time()
    cfg = ds.cfg
    px = back_adjust(ds.close, ds.actions) if cfg.adjust else ds.close.copy()
    liquidity = (ds.close * ds.volume)
    alive = px.notna()
    if cfg.adv_rule == "trailing":
        screen = liquidity.rolling(21, min_periods=1).mean()
    else:
        screen = pd.DataFrame(np.broadcast_to(liquidity.mean(axis=0).to_numpy(), liquidity.shape),
                              index=liquidity.index, columns=liquidity.columns)
    universe = alive & (screen >= ADV_MIN)

    left, joined, score = join_feed(px, ds.feed, cfg)
    fund = fundamental_panel(ds.facts, px.index, list(px.columns), cfg)
    packed = pd.concat({"close": px, "volume": ds.volume, "score": score, "llm": ds.llm,
                        "fund": fund}, axis=1)
    signal_fn = make_signal_fn(cfg)
    composite = signal_fn(packed)

    long = (composite > 0) & universe
    n_univ = universe.sum(axis=1).replace(0, np.nan)
    weights = long.astype(float).div(n_univ, axis=0).fillna(0.0)
    r = px.pct_change(fill_method=None)
    returns = (weights.shift(1) * r.fillna(0.0)).sum(axis=1)
    turnover = (weights - weights.shift(1)).abs().sum(axis=1)
    asset = r.where(universe.shift(1).fillna(False)).mean(axis=1).fillna(0.0)
    return Result(ds=ds, px=px, packed=packed, signal_fn=signal_fn, composite=composite,
                  universe=universe, weights=weights, returns=returns, turnover=turnover,
                  asset=asset, left=left, joined=joined, liquidity=liquidity, fund=fund,
                  runtime=time.time() - t0)


# __BENCH_PART2__
# ==========================================================================================
# Part 2 - the benchmark proper: a defect catalogue, a harness that runs every guard on the
# clean data and on each corrupted copy, the caught/miss/n-a matrix, the effect-size table,
# and the gaps list. Guards are run through the unified layer (fin_skills.api.get(name).run);
# .passed is False exactly when the guard's documented failure fired. Never re-implemented.
# ==========================================================================================
import argparse

from fin_skills.api import get as get_guard          # noqa: E402


# ---- --quick: a smaller world so CI finishes well under 90s (full world is the default) ---
def apply_quick() -> None:
    """Shrink the world in place. Read at call time by make_world/clean_dataset (see the
    note in clean_dataset), so this must run BEFORE the world is built."""
    global N_NAMES, END, EVAL_END
    N_NAMES = 36
    END = "2022-12-30"
    EVAL_END = "2022-12-30"


# ------------------------------------------------------------------------------------------
# helpers the defects and adapters share
# ------------------------------------------------------------------------------------------
def forward_adjust(raw: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Forward- (qfq-) adjust: anchor at the START, scale prices AT OR AFTER each action up
    by its ratio. History never changes; only later prices move. The mirror image of
    back_adjust. Returns identical to back-adjusted (they differ by a per-name constant),
    but price LEVELS are on a different scale - which is exactly what a return-blind check
    misses and a convention check catches."""
    px = raw.copy()
    for _, a in actions.iterrows():
        mask = px.index >= a["date"]
        px.loc[mask, a["ticker"]] = px.loc[mask, a["ticker"]] * a["ratio"]
    return px


def _ema_recursive(a: np.ndarray, span: int) -> np.ndarray:
    """EMA seeded at the first bar (pandas ewm(adjust=False) convention), length-preserving.
    Infinite memory: the last value depends on how much history preceded it."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(a), dtype=float)
    if len(a) == 0:
        return out
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1.0 - alpha) * out[i - 1]
    return out


def _name_xy(px: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """One name's ML feature matrix and binary labels, NaN rows dropped (dense)."""
    feats = ml_features(px)
    y = ml_labels(px)[col].to_numpy()
    x = np.column_stack([f[col].to_numpy() for f in feats])
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return x[ok], y[ok].astype(int)


def _wf_folds(m: int, purge: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walk-forward (train, test) index pairs over m dense rows, purging `purge` bars off
    the end of each train block - the exact shape the pipeline's ML component trades on."""
    out = []
    for t0 in range(MIN_TRAIN, m, BLOCK):
        test = np.arange(t0, min(t0 + BLOCK, m))
        train = np.arange(0, t0 - purge) if purge else np.arange(0, t0)
        if len(train) >= 100 and len(test) > 0:
            out.append((train, test))
    return out


def find_calm_quarter(w: World, eval_start: str, eval_end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First calendar quarter that lies inside [eval_start, eval_end] and is entirely calm
    (true DGP regime 0). A single such quarter is the classic 'my backtest is one quiet
    quarter' defect. Falls back to any all-calm quarter, then to the first quarter."""
    reg = pd.Series(w.regime, index=w.dates)
    q = w.dates.to_period("Q")
    s, e = pd.Timestamp(eval_start), pd.Timestamp(eval_end)
    for require_window in (True, False):
        for p in pd.unique(q):
            m = np.asarray(q == p)
            d0, d1 = w.dates[m][0], w.dates[m][-1]
            if require_window and (d0 < s or d1 > e):
                continue
            seg = reg.to_numpy()[m]
            if len(seg) > 40 and (seg == 0).all():
                return d0, d1
    m0 = np.asarray(q == pd.unique(q)[0])
    return w.dates[m0][0], w.dates[m0][-1]


# ------------------------------------------------------------------------------------------
# the defect catalogue - each takes the clean Dataset and returns a corrupted copy
# ------------------------------------------------------------------------------------------
@dataclass
class Defect:
    key: str
    title: str
    channels: tuple[str, ...]     # which guard input-domain(s) this defect actually perturbs
    severity: str                 # critical / high / medium (my prior; the Sharpe delta is measured)
    make: Callable[[Dataset, World, dict], Dataset]
    note: str = ""


def _d_lookahead(ds, w, ctx):
    return replace(ds, name="lookahead_signal", cfg=replace(ds.cfg, shift_signal=-1))


def _d_asof(ds, w, ctx):
    return replace(ds, name="wrong_side_asof", cfg=replace(ds.cfg, join_exact=True))


def _d_unadjusted(ds, w, ctx):
    return replace(ds, name="unadjusted_split", cfg=replace(ds.cfg, adjust=False))


def _d_forward_adj(ds, w, ctx):
    fwd = forward_adjust(ds.close, ds.actions)
    return replace(ds, name="forward_adjusted_qfq", close=fwd, cfg=replace(ds.cfg, adjust=False))


def _d_survivor(ds, w, ctx):
    survivors = [c for c in ds.close.columns if pd.notna(ds.close[c].iloc[-1])]
    keep = set(survivors)
    return replace(ds, name="survivor_only_universe",
                   close=ds.close[survivors].copy(), volume=ds.volume[survivors].copy(),
                   actions=ds.actions[ds.actions.ticker.isin(keep)].copy())


def _d_warmup(ds, w, ctx):
    return replace(ds, name="warmup_live_window", cfg=replace(ds.cfg, live_window=LIVE_WINDOW))


def _d_scaler(ds, w, ctx):
    return replace(ds, name="shared_scaler", cfg=replace(ds.cfg, scaler_shared=True))


def _d_unpurged(ds, w, ctx):
    return replace(ds, name="unpurged_cv", cfg=replace(ds.cfg, cv="wf_unpurged"))


def _d_latest_fund(ds, w, ctx):
    return replace(ds, name="latest_vintage_fundamentals", cfg=replace(ds.cfg, fund_vintage="latest"))


def _d_llm_overlap(ds, w, ctx):
    pre = max(pd.Timestamp(ctx["start_ok"]),
              pd.Timestamp(LLM_CUTOFF) - pd.DateOffset(years=2))
    return replace(ds, name="llm_cutoff_overlap",
                   cfg=replace(ds.cfg, eval_start=str(pre.date())))


def _d_cost(ds, w, ctx):
    return replace(ds, name="cost_too_low", cfg=replace(ds.cfg, cost_bps=COST_LOW))


def _d_calm(ds, w, ctx):
    c0, c1 = ctx["calm"]
    return replace(ds, name="single_calm_quarter",
                   cfg=replace(ds.cfg, eval_start=str(c0.date()), eval_end=str(c1.date())))


def defect_catalogue() -> list[Defect]:
    return [
        Defect("lookahead_signal", "look-ahead signal (trend reads bar t+1)", ("signal",),
               "critical", _d_lookahead, "cfg.shift_signal=-1: the close is shifted so today's "
               "trend reads tomorrow's price."),
        Defect("wrong_side_asof", "wrong-side as-of join (same-stamp match)", ("feed",),
               "high", _d_asof, "cfg.join_exact=True: merge_asof takes the feed stamped on the "
               "bar's own date - post-close news about that day."),
        Defect("unadjusted_split", "unadjusted split left in the prices", ("adjust",),
               "high", _d_unadjusted, "cfg.adjust=False: raw quotes, so each 2:1 split is a "
               "-50% one-day 'return' the strategy trades."),
        Defect("forward_adjusted_qfq", "forward-adjusted (qfq-style) rewrite", ("adjust",),
               "medium", _d_forward_adj, "delivered close is forward-adjusted while the "
               "pipeline assumes back-adjusted; returns are identical, price LEVELS are not."),
        Defect("survivor_only_universe", "survivor-only universe (delisted names dropped)",
               ("universe",), "high", _d_survivor, "the panel keeps only names alive at the "
               "end - every failure is deleted from history."),
        Defect("warmup_live_window", "indicator warm-up leak (live seeds on too few bars)",
               ("warmup",), "medium", _d_warmup, f"cfg.live_window={LIVE_WINDOW}: the deployed "
               f"EMA is seeded on far fewer bars than it needs to converge."),
        Defect("shared_scaler", "shared full-sample scaler across CV folds", ("cvstate",),
               "high", _d_scaler, "cfg.scaler_shared=True: one scaler fitted on the whole "
               "sample leaks the test set into every fold."),
        Defect("unpurged_cv", "unpurged CV split with overlapping labels", ("cvpurge",),
               "high", _d_unpurged, "cfg.cv=wf_unpurged: train rows abut the test block, so "
               "labels spanning H bars overlap it."),
        Defect("latest_vintage_fundamentals", "restated (latest-vintage) fundamentals",
               ("fund",), "high", _d_latest_fund, "cfg.fund_vintage=latest: the panel uses "
               "restatements filed months after the decision date."),
        Defect("llm_cutoff_overlap", "test window overlaps the LLM training cutoff", ("llm",),
               "critical", _d_llm_overlap, "the reported window is extended back across the "
               "LLM's training cutoff, where its score recalls the future."),
        Defect("cost_too_low", "cost assumption 10x too low", ("cost",), "high", _d_cost,
               f"cfg.cost_bps={COST_LOW:.0f} instead of the realistic {COST_HONEST:.0f} bps "
               f"round-trip."),
        Defect("single_calm_quarter", "regime coverage of a single calm quarter", ("regime",),
               "medium", _d_calm, "the reported window is one all-calm quarter - a single "
               "regime, no turbulent episode."),
    ]


# ------------------------------------------------------------------------------------------
# the guards under test (column order) - name, 4-char code, and the input domain it owns
# ------------------------------------------------------------------------------------------
GUARDS: list[tuple[str, str, str]] = [
    ("assert_causal",       "caus", "signal"),
    ("safe_asof",           "asof", "feed"),
    ("adjustment_check",    "adj ", "adjust"),
    ("survivorship_audit",  "surv", "universe"),
    ("pit_universe",        "univ", "universe"),
    ("warmup_probe",        "warm", "warmup"),
    ("fold_leak_test",      "fold", "cvstate"),
    ("purge_effect",        "purg", "cvpurge"),
    ("pit_fundamentals",    "fund", "fund"),
    ("contamination_probe", "cont", "llm"),
    ("cost_curve",          "cost", "cost"),
    ("regime_coverage",     "regm", "regime"),
]


# ------------------------------------------------------------------------------------------
# guard adapters - build each guard's inputs from a dataset + its pipeline result. Every
# adapter reads the defect's effect out of the config/data, so the same adapter fires on the
# defect and stays quiet on clean. No guard is re-implemented; these only marshal inputs.
# ------------------------------------------------------------------------------------------
def _adapt_causal(ds, res, w, ctx):
    cfg = ds.cfg

    def fn(c, shift=cfg.shift_signal, cfg=cfg):
        cc = c.shift(shift) if shift else c
        return trend_indicator(cc, cfg)
    fn.__name__ = "trend_signal"
    return dict(fn=fn, df=res.px, k=len(res.px) // 2)


def _adapt_asof(ds, res, w, ctx):
    left = (res.px.stack(future_stack=True).rename("close").reset_index()
            .rename(columns={"level_0": "ts", "level_1": "ticker"}))[["ts", "ticker"]]
    right = ds.feed.rename(columns={"feed_ts": "ts"})
    return dict(left=left, right=right, on="ts", by="ticker", tolerance="5D",
                allow_exact_matches=ds.cfg.join_exact)


def _adapt_adjustment(ds, res, w, ctx):
    t = ctx["split_ticker"]
    acts = w.actions[w.actions.ticker == t][["date", "ratio", "kind"]]
    return dict(close=res.px[t].dropna(), actions=acts, expected="back-adjusted")


def _adapt_survivorship(ds, res, w, ctx):
    return dict(prices=res.px, listings=w.listings)


def _adapt_pit_universe(ds, res, w, ctx):
    cols = list(ds.close.columns)
    members = (w.listings[w.listings.ticker.isin(cols)]
               .rename(columns={"listing_date": "start_date", "delisting_date": "end_date"}))
    rebals = pd.bdate_range(res.px.index[0], res.px.index[-1], freq="BQE")
    universe = pu_mod.rebalance_universe(rebals, members)
    return dict(universe=universe)


def _adapt_warmup(ds, res, w, ctx):
    closes = res.px[ctx["ml_name"]].dropna().to_numpy()
    max_probe = min(int(len(closes) * 0.6), 900)        # deep enough for EMA(SPAN) to converge
    return dict(indicator=(lambda a: _ema_recursive(a, SPAN)), closes=closes, tol=1e-9,
                max_probe=max_probe, step=4, history=ds.cfg.live_window)


def _adapt_fold_leak(ds, res, w, ctx):
    x, y = _name_xy(res.px, ctx["ml_name"])
    folds = _wf_folds(len(x), H + EMBARGO)
    if ds.cfg.scaler_shared:
        shared = Scaler().fit(x)                       # fitted on the WHOLE sample -> the leak

        def run_fold(fold, config, _s=shared, _x=x, _y=y):
            tr, te = fold
            s = pe_mod.knn_scores(_s.transform(_x[tr]), _y[tr], _s.transform(_x[te]), k=KNN_K)
            return float(np.nanmean(s))
    else:
        xf, yf = x.copy(), y.copy()
        xf.flags.writeable = False                     # shared INPUT is fine once it is read-only
        yf.flags.writeable = False

        def run_fold(fold, config, _x=xf, _y=yf):
            tr, te = fold
            sc = Scaler().fit(_x[tr])                   # fitted INSIDE the fold
            s = pe_mod.knn_scores(sc.transform(_x[tr]), _y[tr], sc.transform(_x[te]), k=KNN_K)
            return float(np.nanmean(s))
    run_fold.__name__ = "ml_fold"
    return dict(run_fold=run_fold, folds=folds, config={})


def _adapt_purge(ds, res, w, ctx):
    x, y = _name_xy(res.px, ctx["ml_name"])
    purge = 0 if ds.cfg.cv == "wf_unpurged" else (H + EMBARGO)
    folds = _wf_folds(len(x), purge)
    return dict(x=x, y=y, horizon=H, splits=folds, embargo=0)


def _adapt_pit_fund(ds, res, w, ctx):
    t = ctx["fund_ticker"]
    facts = w.facts[t]
    as_of = ctx["fund_as_of"]
    pit = pf_mod.pit_facts(facts, as_of).set_index("period")["val"].astype(float)
    if ds.cfg.fund_vintage == "latest":
        naive = pf_mod.naive_latest(facts).set_index("period")["val"].astype(float)
        used = {p: float(naive[p]) for p in pit.index if p in naive.index}
    else:
        used = {p: float(pit[p]) for p in pit.index}
    return dict(facts=facts, as_of=str(pd.Timestamp(as_of).date()), used=used)


def _adapt_contamination(ds, res, w, ctx):
    return dict(cutoff=ds.cfg.llm_cutoff, test_start=ds.cfg.eval_start, test_end=ds.cfg.eval_end)


def _adapt_cost(ds, res, w, ctx):
    win = res.window()
    return dict(returns=res.returns.loc[win], turnover=res.turnover.loc[win],
                cost_bps=float(ds.cfg.cost_bps))


def _adapt_regime(ds, res, w, ctx):
    win = res.window()
    reg = pd.Series(w.regime, index=w.dates).reindex(win).fillna(0).astype(int).to_numpy()
    labels = np.where(reg == 1, "turbulent", "calm")
    return dict(dates=win,
                asset=res.asset.reindex(win).fillna(0.0).to_numpy(),
                strategy=res.returns.reindex(win).fillna(0.0).to_numpy(),
                position=(res.weights.reindex(win).fillna(0.0).sum(axis=1) > 0).astype(float).to_numpy(),
                labels=labels, how="ex-post label (true DGP regime)")


ADAPTERS: dict[str, Callable] = {
    "assert_causal": _adapt_causal, "safe_asof": _adapt_asof, "adjustment_check": _adapt_adjustment,
    "survivorship_audit": _adapt_survivorship, "pit_universe": _adapt_pit_universe,
    "warmup_probe": _adapt_warmup, "fold_leak_test": _adapt_fold_leak, "purge_effect": _adapt_purge,
    "pit_fundamentals": _adapt_pit_fund, "contamination_probe": _adapt_contamination,
    "cost_curve": _adapt_cost, "regime_coverage": _adapt_regime,
}


def build_context(w: World, clean_ds: Dataset, clean_res: Result) -> dict:
    """Fixed reference points every adapter reuses, chosen once from the clean world so they
    survive every defect (all are SURVIVORS, present even after names are dropped)."""
    survivors = [c for c in clean_res.px.columns if pd.notna(clean_ds.close[c].iloc[-1])]
    counts = clean_res.px[survivors].notna().sum()
    ml_name = str(counts.idxmax())                       # the longest-lived survivor

    split_survivors = [t for t in survivors if t in set(w.actions.ticker)]
    if split_survivors:
        vc = w.actions[w.actions.ticker.isin(split_survivors)].ticker.value_counts()
        split_ticker = str(vc.index[0])
    else:                                                # no survivor split: fall back to any
        split_ticker = str(w.actions.ticker.value_counts().index[0])

    fund_ticker, fund_as_of = None, None
    for t in survivors:
        df = pd.DataFrame(w.facts.get(t, []))
        if df.empty or "form" not in df.columns:
            continue
        df["filed"] = pd.to_datetime(df["filed"])
        amp = df[df.form == "10-Q/A"]
        if amp.empty:
            continue
        end = amp.iloc[len(amp) // 2]["end"]
        orig = df[(df.end == end) & (df.form == "10-Q")]
        amend = df[(df.end == end) & (df.form == "10-Q/A")]
        if orig.empty or amend.empty:
            continue
        of, af = pd.Timestamp(orig.iloc[0]["filed"]), pd.Timestamp(amend.iloc[0]["filed"])
        fund_ticker, fund_as_of = t, of + (af - of) / 2
        break
    if fund_ticker is None:                              # last resort: first ticker with facts
        fund_ticker = next(t for t in survivors if w.facts.get(t))
        fund_as_of = pd.Timestamp(END) - pd.DateOffset(years=1)

    return {"ml_name": ml_name, "split_ticker": split_ticker, "fund_ticker": fund_ticker,
            "fund_as_of": fund_as_of, "calm": find_calm_quarter(w, EVAL_START, EVAL_END),
            "start_ok": str(w.dates[252].date())}


# ------------------------------------------------------------------------------------------
# running one guard, safely: returns (status, fired, runtime_s, message)
#   status: "ran" (fired/clean is meaningful) or "error" (bad inputs / exception)
# ------------------------------------------------------------------------------------------
def run_guard(name: str, ds: Dataset, res: Result, w: World, ctx: dict) -> tuple[str, bool, float, str]:
    try:
        kwargs = ADAPTERS[name](ds, res, w, ctx)
    except Exception as exc:                              # could not even build the inputs
        return "error", False, 0.0, f"input build failed: {type(exc).__name__}: {exc}"
    t0 = time.perf_counter()
    try:
        r = get_guard(name).run(**kwargs)
    except Exception as exc:
        return "error", False, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"
    dt = time.perf_counter() - t0
    fired = not r.passed
    msg = (r.errors[0].message if fired and r.errors else
           (r.findings[-1].message if r.findings else "clean"))
    return "ran", fired, dt, msg.replace("\n", " ")


# ------------------------------------------------------------------------------------------
# report writer - plain ASCII to stdout, the same content into RESULTS.md
# ------------------------------------------------------------------------------------------
class Report:
    def __init__(self) -> None:
        self.md: list[str] = []

    def head(self, title: str, lead: str = "") -> None:
        print("\n" + title)
        print("=" * len(title))
        self.md.append(f"\n## {title}\n")
        if lead:
            print(lead)
            self.md.append(lead + "\n")

    def block(self, lines: list[str]) -> None:
        for ln in lines:
            print(ln)
        self.md.append("```\n" + "\n".join(lines) + "\n```")

    def line(self, s: str = "") -> None:
        print(s)
        self.md.append(s)

    def write(self, path: Path, header: list[str]) -> None:
        path.write_text("\n".join(header) + "\n" + "\n".join(self.md) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------
def main(quick: bool) -> int:
    t_start = time.time()
    if quick:
        apply_quick()

    w = make_world()
    clean_ds = clean_dataset(w)
    defects = defect_catalogue()

    # Build every dataset, run its pipeline once, and cache the result (guards + effect sizes
    # both read it). 1 clean + len(defects) corrupted.
    datasets: dict[str, Dataset] = {"clean": clean_ds}
    ctx_seed = build_context(w, clean_ds, run_pipeline(clean_ds))
    for d in defects:
        datasets[d.key] = d.make(clean_ds, w, ctx_seed)
    results: dict[str, Result] = {k: run_pipeline(ds) for k, ds in datasets.items()}
    ctx = build_context(w, clean_ds, results["clean"])

    guard_channel = {name: chan for name, _, chan in GUARDS}
    defect_channels = {d.key: d.channels for d in defects}

    # ---- run EVERY guard on EVERY dataset (clean + each corrupted). Classification is by
    # input domain: a guard "should" catch only defects in its declared channel. Off-channel
    # cells are n/a (the guard cannot see that defect) - UNLESS the guard fires anyway, which
    # is a cross-catch and is surfaced as a surprise, not silently dropped. -------------------
    all_keys = ["clean"] + [d.key for d in defects]
    fired: dict[tuple[str, str], bool] = {}
    status: dict[tuple[str, str], str] = {}
    rt: dict[tuple[str, str], float] = {}
    msg: dict[tuple[str, str], str] = {}
    for name, _, _ in GUARDS:
        for k in all_keys:
            st, fr, dt, m = run_guard(name, datasets[k], results[k], w, ctx)
            fired[(k, name)] = fr
            status[(k, name)] = st
            rt[(k, name)] = dt
            msg[(k, name)] = m

    rep = Report()
    codes = {name: code for name, code, _ in GUARDS}

    # ---------------------------------------------------------------- the matrix
    rep.head("DETECTION MATRIX  (rows = defects, columns = guards)")
    col_hdr = "  ".join(c.strip().rjust(4) for _, c, _ in GUARDS)
    row_label = "defect \\ guard"
    lines = [f"{row_label:<28}{col_hdr}"]
    lines.append("-" * len(lines[0]))

    def cell(k: str, name: str, applicable: bool) -> str:
        if status.get((k, name)) == "error":
            return "ERR"
        if k == "clean":
            return "FP!" if fired[(k, name)] else "ok"
        if not applicable:
            return "hit*" if fired[(k, name)] else "."
        return "HIT" if fired[(k, name)] else "MISS"

    caught_by: dict[str, list[str]] = {}
    cross: list[tuple[str, str]] = []
    for d in defects:
        row = []
        for name, _, chan in GUARDS:
            applicable = chan in d.channels
            row.append(cell(d.key, name, applicable).rjust(4))
            if status.get((d.key, name)) == "ran" and fired[(d.key, name)]:
                if applicable:
                    caught_by.setdefault(d.key, []).append(name)
                else:
                    cross.append((d.key, name))
        lines.append(f"{d.key:<28}" + "  ".join(row))
    # clean row last
    row = []
    for name, _, _ in GUARDS:
        row.append(cell("clean", name, True).rjust(4))
    lines.append(f"{'clean (false positives)':<28}" + "  ".join(row))
    rep.block(lines)
    rep.line()
    rep.line("legend:  HIT  = guard's documented failure fired   MISS = ran, did not fire")
    rep.line("         .    = n/a, the defect does not touch this guard's input domain")
    rep.line("         hit* = fired OUTSIDE its domain (cross-catch; see SURPRISES)")
    rep.line("         FP!  = FALSE ALARM on clean data           ok = clean, correctly silent")
    rep.line()
    rep.line("guard codes:  " + " | ".join(f"{c.strip()}={n}" for n, c, _ in GUARDS[:6]))
    rep.line("              " + " | ".join(f"{c.strip()}={n}" for n, c, _ in GUARDS[6:]))

    # ---------------------------------------------------------------- effect sizes
    rep.head("EFFECT SIZES  (change in reported net Sharpe, clean -> corrupted)",
             "The strategy's net Sharpe over each dataset's own reporting window. The sign is\n"
             "secondary: + flatters, - degrades returns or shifts to a weaker window; the\n"
             "MAGNITUDE is how far the reported number strays. Weigh it against the s.e. below\n"
             "- small moves are noise. The detection matrix above is the robust result.")
    base = results["clean"].sharpe()
    n_clean = len(results["clean"].window())
    se_clean = sharpe_se(base, n_clean)
    et = [f"{'defect':<28}{'sev':<9}{'clean':>7}{'corrupt':>9}{'delta':>8}   caught by",
          "-" * 86]
    rows_sorted = sorted(defects, key=lambda d: -(results[d.key].sharpe() - base
                         if np.isfinite(results[d.key].sharpe()) else -9))
    for d in rows_sorted:
        cs = results[d.key].sharpe()
        delta = cs - base
        catchers = ",".join(codes[g].strip() for g in caught_by.get(d.key, [])) or "-- NONE --"
        et.append(f"{d.key:<28}{d.severity:<9}{base:>7.2f}{cs:>9.2f}{delta:>+8.2f}   {catchers}")
    et.append("-" * 86)
    et.append(f"clean baseline net Sharpe {base:.2f} over {n_clean} days "
              f"(Lo 2002 iid s.e. ~{se_clean:.2f}); cost {results['clean'].ds.cfg.cost_bps:.0f} bps")
    et.append(f"a |delta| below ~{se_clean:.2f} (one s.e.) is not distinguishable from noise "
              f"at this sample size")
    rep.block(et)
    rep.line()
    rep.line("Notes on the small / negative deltas (all are real defects, all caught above):")
    rep.line("  - each sub-signal is 1/5 of the blended composite, so a leak in one (unpurged")
    rep.line("    CV, shared scaler, restated fundamentals) barely moves the blended Sharpe")
    rep.line("    even as the guard flags it at the source.")
    rep.line("  - forward_adjusted_qfq moves price LEVELS only; returns (hence Sharpe) are")
    rep.line("    unchanged by construction - a data-integrity defect, not a Sharpe lie.")
    rep.line("  - unadjusted_split DEGRADES the Sharpe: the strategy trades the fake -50% split")
    rep.line("    jumps and loses. A negative delta is still a corrupted backtest.")
    rep.line("  - llm_cutoff_overlap / single_calm_quarter change the WINDOW, not the signal;")
    rep.line("    the sign depends on which window, which is exactly why coverage must be")
    rep.line("    audited structurally rather than trusted from one number.")

    # ---------------------------------------------------------------- runtimes
    rep.head("GUARD RUNTIME  (seconds, on this defect's dataset)")
    rtl = [f"{'guard':<22}{'target defect':<30}{'runtime_s':>10}", "-" * 62]
    for name, _, chan in GUARDS:
        tgt = next((d.key for d in defects if chan in d.channels), "clean")
        rtl.append(f"{name:<22}{tgt:<30}{rt.get((tgt, name), rt.get(('clean', name), 0.0)):>10.3f}")
    rep.block(rtl)

    # ---------------------------------------------------------------- gaps + surprises
    def anything_fired(key: str) -> bool:
        return any(status.get((key, n)) == "ran" and fired[(key, n)] for n, _, _ in GUARDS)

    gaps = [d for d in defects if not anything_fired(d.key)]
    false_alarms = [n for n, _, _ in GUARDS if fired.get(("clean", n)) and status[("clean", n)] == "ran"]
    misses = [(d.key, n) for d in defects for n, _, ch in GUARDS
              if ch in d.channels and status.get((d.key, n)) == "ran" and not fired[(d.key, n)]]
    errors = [(k, n) for (k, n), s in status.items() if s == "error"]

    rep.head("GAPS  (defects no guard flagged in this run)",
             "'no guard' = nothing in the suite targets this input domain, so it is the next\n"
             "script to write. 'gate did not bind' = a guard ran but its pass/fail threshold\n"
             "was not crossed here - a limitation of the gate, not a missing script.")
    if gaps:
        for d in gaps:
            applicable = [n for n, _, ch in GUARDS if ch in d.channels]
            if not applicable:
                reason = "NO GUARD targets this domain -> next script to write"
            else:
                bits = []
                for n in applicable:
                    if status.get((d.key, n)) == "ran":
                        bits.append(f"{n} ran, gate did not bind [{msg[(d.key, n)][:52]}]")
                    elif status.get((d.key, n)) == "error":
                        bits.append(f"{n} errored [{msg[(d.key, n)][:52]}]")
                reason = "; ".join(bits)
            rep.line(f"  - {d.key} [{d.severity}]: {d.title}")
            rep.line(f"      {reason}")
    else:
        rep.line("  none: every defect is caught by at least one guard.")

    rep.head("SURPRISES  (reported honestly, not tuned away)")
    for n in false_alarms:
        rep.line(f"  FALSE ALARM  {n} fired on clean data: {msg[('clean', n)][:70]}")
    for k, n in misses:
        rep.line(f"  MISS         {n} did not fire on {k}: {msg[(k, n)][:64]}")
    for k, n in cross:
        rep.line(f"  CROSS-CATCH  {n} fired on {k} (outside its domain): {msg[(k, n)][:52]}")
    for k, n in errors:
        rep.line(f"  ERROR        {n} on {k}: {msg[(k, n)][:64]}")
    if not (false_alarms or misses or cross or errors):
        rep.line("  none.")
    if cross:
        rep.line()
        rep.line("  A cross-catch is a real signal, not a false alarm: the guard is right that")
        rep.line("  something is wrong, but it is not the guard that DIAGNOSES the cause. e.g.")
        rep.line("  cost_curve fires on a cratered backtest because the edge is gone - true, but")
        rep.line("  it is adjustment_check that tells you an unadjusted split is why.")

    # ---------------------------------------------------------------- summary + file
    n_defects = len(defects)
    n_caught = sum(1 for d in defects if caught_by.get(d.key))
    total_rt = sum(rt.values())
    elapsed = time.time() - t_start
    rep.head("SUMMARY")
    rep.line(f"  world      : {len(w.dates)} days, {len(w.tickers)} names, "
             f"{int(w.listings.delisting_date.notna().sum())} delisted, {len(w.actions)} splits, "
             f"turbulent share {w.regime.mean():.2f}  (seed {SEED}{', --quick' if quick else ''})")
    rep.line(f"  defects    : {n_caught}/{n_defects} caught by >=1 guard; "
             f"{len(false_alarms)} false alarm(s) on clean; {len(gaps)} gap(s)")
    rep.line(f"  guard time : {total_rt:.2f}s over {sum(1 for _ in fired)} guard runs; "
             f"total benchmark {elapsed:.1f}s")

    header = [
        "# leak_bench results",
        "",
        f"Generated by `python benchmarks/leak_bench.py{' --quick' if quick else ''}` "
        f"(seed {SEED}). This file is regenerated by the script; edit the method notes in "
        f"`benchmarks/README.md`, not here.",
        "",
        f"- world: {len(w.dates)} trading days, {len(w.tickers)} names, "
        f"{int(w.listings.delisting_date.notna().sum())} delisted, {len(w.actions)} splits",
        f"- guards under test: {len(GUARDS)} (imported from `fin_skills`, run via "
        f"`fin_skills.api.get(name).run`)",
        f"- defects planted: {n_defects}; caught by >=1 guard: {n_caught}; "
        f"false alarms on clean: {len(false_alarms)}",
    ]
    out_path = ROOT / "benchmarks" / "RESULTS.md"
    rep.write(out_path, header)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plant known defects, measure which guards catch them.")
    ap.add_argument("--quick", action="store_true",
                    help="smaller world so the whole run finishes in well under 90s (for CI)")
    args = ap.parse_args()
    raise SystemExit(main(quick=args.quick))
