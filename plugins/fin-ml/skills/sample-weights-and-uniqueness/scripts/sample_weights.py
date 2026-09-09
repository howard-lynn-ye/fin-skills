#!/usr/bin/env python3
"""Concurrency, average uniqueness and sample weights - how many observations you really have.

Advances in Financial Machine Learning (Lopez de Prado 2018), chapter 4. A label that spans
[t0, t1] shares its outcome with every other label whose span overlaps it. Fit on 2,000 such
labels and the fitter believes it saw 2,000 independent draws; it saw far fewer, and every
standard error, t-statistic and accuracy confidence interval computed from that count is too
small. Nothing warns you - the arrays have the right shape.

Measured here, all seeded:

  1. concurrency and average uniqueness (Snippets 4.1 and 4.2), against the analytic value
     `min(1, step / span)` for regularly spaced spans;
  2. the effective sample size: `sum(uniqueness)` and the Kish effective size of the weights,
     against the nominal count;
  3. what unweighted fitting does to a test: the Monte Carlo size of a nominal 5 % t-test on
     overlapping labels, unweighted and uniqueness-weighted;
  4. return-attribution weights (Snippet 4.10) and time decay (Snippet 4.11), and the fact that
     the decay clock runs on cumulative uniqueness rather than on calendar time;
  5. sequential bootstrap (Snippets 4.5-4.6) against the standard one, by drawn uniqueness;
  6. the bar-t0 return the reference weight-by-return implementation includes.

Cross-validation of overlapping labels is a different problem with a different fix - purging and
embargoing - and it belongs to the lib-purgedcv skill. This script is only about weights.

Run:  python sample_weights.py     (numpy; no optional libraries; seed 0; ~1 s)
"""
from __future__ import annotations

import math
import time

import numpy as np

SEED = 0
N_BARS = 2000
SPAN = 20                # label horizon in bars
STEP = 1                 # one event per bar: the maximum-overlap case
MC_PATHS = 400           # Monte Carlo paths for the standard-error experiment
FEATURE_LAG = 5          # the null feature: the prior 5-bar return, known at t0
DECAY = 0.5              # time-decay parameter c in Snippet 4.11


# ------------------------------------------------------------------- events and overlap -----
def regular_events(n_bars: int = N_BARS, span: int = SPAN, step: int = STEP):
    """Events every `step` bars, each resolving `span` bars later. Returns (t0, t1)."""
    t0 = np.arange(0, n_bars - span, step, dtype=int)
    return t0, t0 + span


def num_concurrent(n_bars: int, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Labels spanning each bar, endpoints INCLUSIVE (AFML Snippet 4.1, page 60).

    mlfinpy 0.1.2 `num_concurrent_events` does `count.loc[t_in:t_out] += 1`, and pandas' label
    slicing includes both ends - so a label occupies span + 1 bars, not span.
    """
    diff = np.zeros(n_bars + 1, dtype=float)
    np.add.at(diff, np.asarray(t0, dtype=int), 1.0)
    np.add.at(diff, np.asarray(t1, dtype=int) + 1, -1.0)
    return np.cumsum(diff)[:n_bars]


def average_uniqueness(t0: np.ndarray, t1: np.ndarray, conc: np.ndarray) -> np.ndarray:
    """Mean of 1 / concurrency over each label's own span (AFML Snippet 4.2, page 62)."""
    inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1e-12), 0.0)
    cum = np.concatenate([[0.0], np.cumsum(inv)])
    t0 = np.asarray(t0, dtype=int)
    t1 = np.minimum(np.asarray(t1, dtype=int), conc.shape[0] - 1)
    return (cum[t1 + 1] - cum[t0]) / (t1 - t0 + 1)


def weights_by_return(t0: np.ndarray, t1: np.ndarray, log_ret: np.ndarray, conc: np.ndarray,
                      include_t0_bar: bool = True) -> np.ndarray:
    """|sum over the span of log_ret / concurrency|, normalised to sum to n (Snippet 4.10, p.69).

    mlfinpy 0.1.2 `_apply_weight_by_return` is
    `(ret.loc[t_in:t_out] / num_conc_events.loc[t_in:t_out]).sum()` with
    `ret = np.log(close_series).diff()`, then `.abs()`, and `get_weights_by_return` rescales with
    `weights *= weights.shape[0] / weights.sum()`. Because `ret[t0]` is the return INTO bar t0,
    that slice includes one bar the label did not earn; `include_t0_bar=False` drops it, and the
    demo measures the difference.
    """
    attr = np.where(conc > 0, log_ret / np.maximum(conc, 1e-12), 0.0)
    cum = np.concatenate([[0.0], np.cumsum(attr)])
    t0 = np.asarray(t0, dtype=int)
    t1 = np.minimum(np.asarray(t1, dtype=int), conc.shape[0] - 1)
    lo = t0 if include_t0_bar else t0 + 1
    w = np.abs(cum[t1 + 1] - cum[lo])
    s = w.sum()
    return w * (w.shape[0] / s) if s > 0 else w


def time_decay_weights(av_uniq: np.ndarray, decay: float = DECAY) -> np.ndarray:
    """Piecewise-linear decay over CUMULATIVE UNIQUENESS, not calendar time (Snippet 4.11, p.70).

    mlfinpy 0.1.2 `get_weights_by_time_decay`: `decay_w = tW.sort_index().cumsum()`;
    `slope = (1 - decay) / decay_w[-1]` for decay >= 0 else `1 / ((decay + 1) * decay_w[-1])`;
    `const = 1 - slope * decay_w[-1]`; `decay_w = const + slope * decay_w`; negatives clipped to 0.
    decay = 1 is no decay; decay = 0 sends the oldest observation to weight 0; decay < 0 erases
    the oldest fraction entirely.
    """
    cum = np.cumsum(np.asarray(av_uniq, dtype=float))
    last = cum[-1]
    slope = (1.0 - decay) / last if decay >= 0 else 1.0 / ((decay + 1.0) * last)
    const = 1.0 - slope * last
    w = const + slope * cum
    return np.maximum(w, 0.0)


def kish_ess(w: np.ndarray) -> float:
    """Kish's effective sample size, (sum w)^2 / sum(w^2). Equals n for uniform weights."""
    w = np.asarray(w, dtype=float)
    s2 = float(np.sum(w * w))
    return float(np.sum(w)) ** 2 / s2 if s2 > 0 else 0.0


# ------------------------------------------------------------------- bootstrap ---------------
def indicator_matrix(n_bars: int, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """bars x labels 0/1 matrix, endpoints inclusive (mlfinpy `get_ind_matrix`)."""
    m = np.zeros((n_bars, len(t0)), dtype=float)
    for j, (a, b) in enumerate(zip(np.asarray(t0, int), np.asarray(t1, int))):
        m[a:min(b, n_bars - 1) + 1, j] = 1.0
    return m


def ind_mat_average_uniqueness(ind_mat: np.ndarray) -> float:
    """mlfinpy `get_ind_mat_average_uniqueness`: mean of the non-zero 1/concurrency entries."""
    conc = ind_mat.sum(axis=1)
    u = np.divide(ind_mat.T, conc, out=np.zeros_like(ind_mat.T), where=conc != 0)
    return float(u[u > 0].mean())


def sequential_bootstrap(ind_mat: np.ndarray, size: int, rng) -> np.ndarray:
    """Draw `size` labels with probability proportional to uniqueness GIVEN what is drawn.

    AFML Snippets 4.5 and 4.6 (page 65), as cited by mlfinpy's `seq_bootstrap`. At each step the
    average uniqueness of every candidate is recomputed against the concurrency the already-drawn
    sample implies, and the draw is proportional to it.
    """
    n_bars, n_lab = ind_mat.shape
    prev = np.zeros(n_bars)
    out = np.empty(size, dtype=int)
    for k in range(size):
        denom = ind_mat + prev[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(ind_mat > 0, ind_mat / np.maximum(denom, 1e-12), 0.0)
        counts = ind_mat.sum(axis=0)
        avg = u.sum(axis=0) / np.maximum(counts, 1.0)
        p = avg / avg.sum()
        j = int(rng.choice(n_lab, p=p))
        out[k] = j
        prev = prev + ind_mat[:, j]
    return out


def drawn_uniqueness(ind_mat: np.ndarray, draw: np.ndarray) -> float:
    """Average uniqueness of a drawn (with-replacement) sample, mlfinpy's comparison metric."""
    sub = ind_mat[:, np.asarray(draw, dtype=int)]
    return ind_mat_average_uniqueness(sub)


# ------------------------------------------------------------------- the experiment ----------
def one_path(seed: int, n_bars: int = N_BARS, span: int = SPAN, step: int = STEP,
             lag: int = FEATURE_LAG):
    """A driftless log-price path, its overlapping span labels, and a causal null feature.

    Returns (log_ret, y, x, t0, t1). `x[i]` is the prior `lag`-bar return, known at t0 and
    independent of the label by construction, so any t-statistic it produces is a false positive.
    """
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, 0.01, n_bars)
    p = np.cumsum(log_ret)
    t0, t1 = regular_events(n_bars, span, step)
    keep = t0 >= lag
    t0, t1 = t0[keep], t1[keep]
    y = p[t1] - p[t0]
    x = p[t0] - p[t0 - lag]
    return log_ret, y, x, t0, t1


def ols_t(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float]:
    """Slope and its textbook (independent-observations) t-statistic, optionally weighted."""
    n = x.shape[0]
    w = np.ones(n) if w is None else np.asarray(w, dtype=float) * n / np.sum(w)
    X = np.column_stack([np.ones(n), x])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    resid = y - X @ coef
    s2 = float(np.sum(w * resid * resid)) / (n - 2)
    xtx_inv = np.linalg.pinv((X * w[:, None]).T @ X)
    return float(coef[1]), float(coef[1] / math.sqrt(s2 * xtx_inv[1, 1]))


def t_test_size(n_paths: int = MC_PATHS, n_bars: int = N_BARS, span: int = SPAN,
                step: int = STEP, seed0: int = SEED) -> dict:
    """Monte Carlo size of a nominal 5 % t-test on overlapping labels, weighted and not.

    The feature is independent of the label by construction, so a correctly sized test rejects
    5 % of the time. Anything above that is the overlap.
    """
    t_un = np.empty(n_paths)
    t_w = np.empty(n_paths)
    betas = np.empty(n_paths)
    for i in range(n_paths):
        _, y, x, t0, t1 = one_path(seed0 + i, n_bars, span, step)
        conc = num_concurrent(n_bars, t0, t1)
        u = average_uniqueness(t0, t1, conc)
        betas[i], t_un[i] = ols_t(x, y)
        _, t_w[i] = ols_t(x, y, u)
    return {"t_unweighted": t_un, "t_weighted": t_w, "beta": betas,
            "size_unweighted": float(np.mean(np.abs(t_un) > 1.96)),
            "size_weighted": float(np.mean(np.abs(t_w) > 1.96)),
            "sd_t_unweighted": float(t_un.std(ddof=1)),
            "sd_t_weighted": float(t_w.std(ddof=1))}


# ------------------------------------------------------------------- demo --------------------
def _main() -> None:
    t_all = time.time()
    rng = np.random.default_rng(SEED)

    print("=== 1. Concurrency and average uniqueness (AFML Snippets 4.1 p.60, 4.2 p.62) ===")
    print(f"  {N_BARS} bars, labels spanning {SPAN} bars, one event every `step` bars")
    print("    step   events   mean concurrency   mean uniqueness   analytic min(1, step/span)")
    for step in (1, 2, 5, 10, 20, 40):
        t0, t1 = regular_events(N_BARS, SPAN, step)
        conc = num_concurrent(N_BARS, t0, t1)
        u = average_uniqueness(t0, t1, conc)
        inner = slice(SPAN, len(t0) - SPAN)      # away from the ends, where concurrency ramps
        print(f"    {step:4d}   {len(t0):6d}   {conc[SPAN:-SPAN].mean():16.3f}   "
              f"{u[inner].mean():15.4f}   {min(1.0, step / (SPAN + 1)):26.4f}")
    print(f"  the analytic value uses span + 1 bars because both endpoints are inclusive - that")
    print(f"  is what `count.loc[t_in:t_out] += 1` does, and it is why step={SPAN} is not unique.")

    t0, t1 = regular_events()
    conc = num_concurrent(N_BARS, t0, t1)
    u = average_uniqueness(t0, t1, conc)
    n = len(t0)
    print(f"\n=== 2. Effective sample size, step={STEP} (every bar), span={SPAN} ===")
    print(f"  nominal sample size n = {n}")
    print(f"  sum of average uniqueness  = {u.sum():.1f}  ({u.sum() / n:.1%} of n)")
    print(f"  mean average uniqueness    = {u.mean():.4f}")
    print(f"  Kish effective size of uniform weights = {kish_ess(np.ones(n)):.1f} (= n, by "
          f"construction: uniform weights KNOW nothing about overlap)")
    print(f"  Kish effective size of the uniqueness weights = {kish_ess(u):.1f}")

    print(f"\n=== 3. What that does to a t-test ({MC_PATHS} paths, a feature independent of the "
          f"label) ===")
    mc = t_test_size()
    print(f"  the feature is the prior {FEATURE_LAG}-bar return and the label is the NEXT "
          f"{SPAN} bars, so the true slope is 0")
    print(f"  unweighted OLS: sd of the reported t-statistic {mc['sd_t_unweighted']:.2f} "
          f"(a correct one is 1.00)")
    print(f"  a nominal 5 % two-sided test rejects on {mc['size_unweighted']:.1%} of null paths")
    print(f"  uniqueness-weighted: sd of t {mc['sd_t_weighted']:.2f}, size "
          f"{mc['size_weighted']:.1%}")
    print(f"  implied correction factor on the t-statistic: divide by "
          f"{mc['sd_t_unweighted']:.2f}; a reported t of 3.0 is really "
          f"{3.0 / mc['sd_t_unweighted']:.2f}")
    n_eff_mc = n / mc["sd_t_unweighted"] ** 2
    print(f"  that inflation implies an effective sample size of {n_eff_mc:.0f} out of {n} "
          f"({n_eff_mc / n:.1%}), against the {u.sum():.0f} ({u.sum() / n:.1%}) that summed")
    print(f"    uniqueness predicts - uniqueness gives the right direction and is "
          f"{n_eff_mc / u.sum():.1f}x too pessimistic here, because the FEATURE is")
    print("    autocorrelated too and the two effects do not simply multiply.")
    print("  weighting does NOT fix this: weights change what each observation counts for, not")
    print("  the correlation between overlapping residuals. The fix is purged CV and an")
    print("  overlap-aware standard error.")

    print(f"\n=== 4. Return-attribution and time-decay weights (Snippets 4.10 p.69, 4.11 p.70) ===")
    log_ret = np.random.default_rng(SEED).normal(0.0, 0.01, N_BARS)
    wr = weights_by_return(t0, t1, log_ret, conc)
    print(f"  weight by |return attribution|: min {wr.min():.4f}, median {np.median(wr):.4f}, "
          f"max {wr.max():.4f}, sum {wr.sum():.1f} (normalised to n = {n})")
    order = np.argsort(wr)[::-1]
    print(f"  the top 10 % of events carry {wr[order[:n // 10]].sum() / wr.sum():.1%} of the "
          f"total weight; the bottom half carry "
          f"{wr[order[n // 2:]].sum() / wr.sum():.1%}")
    print(f"  Kish effective size of these weights: {kish_ess(wr):.1f} of {n} "
          f"({kish_ess(wr) / n:.1%})")
    for dec in (1.0, 0.75, 0.5, 0.0, -0.5):
        td = time_decay_weights(u, dec)
        zero = float(np.mean(td == 0.0))
        print(f"  decay c={dec:+.2f}: oldest weight {td[0]:.4f}, newest {td[-1]:.4f}, "
              f"{zero:.1%} of events at exactly zero, Kish {kish_ess(td):.1f}")
    print("  the decay is linear in CUMULATIVE UNIQUENESS, so a crowded stretch of history ages")
    print("  more slowly than a sparse one - it is not a calendar half-life.")

    print(f"\n=== 5. Sequential vs standard bootstrap (Snippets 4.5-4.6, page 65) ===")
    # Irregular events over a wide range, so a smarter draw HAS room to pick unique labels.
    g0 = np.random.default_rng(SEED)
    t0s = np.sort(g0.integers(0, 380, 80))
    t1s = t0s + 10
    im = indicator_matrix(int(t1s.max()) + 1, t0s, t1s)
    print(f"  {im.shape[1]} labels of span 10 at random starts over {im.shape[0]} bars; "
          f"whole-set average uniqueness {ind_mat_average_uniqueness(im):.4f}")
    std_u, seq_u = [], []
    for r in range(5):
        g = np.random.default_rng(SEED + r)
        std_u.append(drawn_uniqueness(im, g.integers(0, im.shape[1], im.shape[1])))
        seq_u.append(drawn_uniqueness(im, sequential_bootstrap(im, im.shape[1], g)))
    print(f"  standard bootstrap, drawn uniqueness:   mean {np.mean(std_u):.4f} "
          f"(5 draws, sd {np.std(std_u, ddof=1):.4f})")
    print(f"  sequential bootstrap, drawn uniqueness: mean {np.mean(seq_u):.4f} "
          f"(5 draws, sd {np.std(seq_u, ddof=1):.4f})")
    print(f"  ratio {np.mean(seq_u) / np.mean(std_u):.2f}x")

    print(f"\n=== 6. The bar-t0 return the reference implementation includes ===")
    wr_no = weights_by_return(t0, t1, log_ret, conc, include_t0_bar=False)
    rel = np.abs(wr - wr_no) / np.maximum(np.abs(wr_no), 1e-12)
    in_mean_units = np.abs(wr - wr_no) / wr_no.mean()
    rank = np.corrcoef(np.argsort(np.argsort(wr)), np.argsort(np.argsort(wr_no)))[0, 1]
    print(f"  `ret.loc[t_in:t_out]` starts at the return INTO bar t0, which the label did not "
          f"earn.")
    print(f"  dropping it moves each weight by a median {np.median(rel):.1%} of its own value "
          f"and {np.median(in_mean_units):.1%} of the mean weight;")
    print(f"  {np.mean(rel > 0.10):.1%} of events move by more than 10 % of their own value, the "
          f"largest move is {in_mean_units.max():.2f} mean weights, and the rank correlation")
    print(f"  between the two weight vectors is {rank:.4f} - the ordering survives, the "
          f"individual weights do not.")

    print("\nRule: count your effective sample size from label concurrency, not from len(y) -"
          " weight by uniqueness, and divide every t-statistic by the overlap factor.")
    print(f"total runtime {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    _main()
