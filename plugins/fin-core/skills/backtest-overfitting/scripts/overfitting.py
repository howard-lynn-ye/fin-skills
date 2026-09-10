#!/usr/bin/env python3
"""Is the surviving edge distinguishable from the best of N tries?

The seven guards elsewhere in this repo ask whether a backtest cheated mechanically -
peeked, leaked, priced the impossible. None of them asks the question this file exists
for: the backtest is clean, and it is still the MAXIMUM OF A SEARCH.

Two measurements answer it, and neither needs the strategies to be independent:

  PBO  - the Probability of Backtest Overfitting via Combinatorially Symmetric
         Cross-Validation. Split the sample into S blocks, form every C(S, S/2) way of
         calling half of them in-sample, pick the configuration that wins IS, and look
         up where that same configuration lands out-of-sample. PBO is the share of
         splits where the IS winner falls in the bottom half OS. Under no skill it is
         0.5: the winner is a coin flip.

  MinBTL - the Minimum Backtest Length. E[max Sharpe] over N null trials is a known
         function of N, so a claimed Sharpe and a trial count imply a sample length
         below which the claim cannot be supported by any amount of care.

Both are Bailey, Borwein, Lopez de Prado & Zhu. This file implements them from the
definitions, verifies what it can against a primary source, and MEASURES the properties
it cannot verify by hand:

  * PBO on a grid of pure-noise strategies converges to 0.5 (measured, many seeds);
  * planting one genuinely predictive strategy pulls it down (measured, by edge size);
  * the IS winner's OS rank decays toward a coin flip as N grows (measured);
  * the expected-max-Sharpe approximation is checked against a 200k-draw Monte Carlo.

Run:  python overfitting.py         (numpy + scipy only, seeded, ~20 s)

Definitions, and how far each one is verified - both against PDFs actually read:

  * MinBTL and E[max SR] - Bailey, Borwein, Lopez de Prado & Zhu, "Pseudo-Mathematics and
    Financial Charlatanism", Notices of the AMS 61(5), May 2014, 458-471 (open access,
    ams.org/notices/201405/rnoti-p458.pdf). Proposition 1 and Theorem 2. `__main__`
    section H reproduces the paper's own three worked anchors.
  * CSCV / PBO - the same authors' "The Probability of Backtest Overfitting" (Journal of
    Computational Finance 20(4), 2016), Algorithm 2.3 and Definition 2.2, read from the
    authors' hosted preprint at davidhbailey.com/dhbpapers/backtest-prob.pdf. The relative
    rank is defined THERE as rank/(N+1); see RANK_NOTE. One arithmetic slip in that
    preprint is flagged in PBO_SOURCE and reproduced in `__main__`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from math import comb, ceil

import numpy as np
from scipy.stats import norm

EULER_GAMMA = float(np.euler_gamma)      # 0.5772156649015329

MINBTL_SOURCE = (
    "Bailey, Borwein, Lopez de Prado & Zhu (2014), Notices of the AMS 61(5), 458-471,\n"
    "  'Pseudo-Mathematics and Financial Charlatanism', Proposition 1 and Theorem 2.\n"
    "  E[max SR] over N independent null trials, and MinBTL = (E[max SR] / SR*)^2 years,\n"
    "  bounded above by 2*ln(N) / SR*^2."
)

PBO_SOURCE = (
    "Bailey, Borwein, Lopez de Prado & Zhu, 'The Probability of Backtest Overfitting',\n"
    "  Journal of Computational Finance 20(4), 2016. Algorithm 2.3 defines CSCV;\n"
    "  Definition 2.2 defines PBO as the probability the IS-optimal configuration lands\n"
    "  below the OS median. S = 16 is the value the authors recommend, because on four\n"
    "  years of daily data it makes each block a quarter and so preserves the serial\n"
    "  correlation inside a block.\n"
    "  ! The hosted preprint prints C(16,8) = 12,780 twice. The correct value is 12,870,\n"
    "    which is what this file computes; C(24,12) and C(12,6) are printed correctly\n"
    "    there, so it is a transposition, not a different definition."
)

RANK_NOTE = (
    "w = rank / (N + 1), rank 1 = worst OS, rank N = best - the paper's own definition,\n"
    "  verbatim, and the (N+1) is why the logit stays finite when the IS winner also wins\n"
    "  OS. Section G measures what the alternatives would have changed."
)


# ======================================================================================
# Expected maximum Sharpe ratio, and the Minimum Backtest Length that follows from it
# ======================================================================================
def expected_max_sharpe(n_trials: int, sr_std: float = 1.0) -> float:
    """E[max SR] over `n_trials` independent trials whose SR estimates have sd `sr_std`.

    sr_std * [ (1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e)) ],  g = Euler-Mascheroni.

    This is the same expression `../../backtest-validation/scripts/trial_ledger.py`
    computes as `sr0` inside `deflated_sharpe`; DSR compares the observed Sharpe to it,
    MinBTL inverts it for the sample length. It is an approximation to the true expected
    maximum of N standard normals - `__main__` section E measures the error.
    """
    n = int(n_trials)
    if n < 2:
        raise ValueError("n_trials must be at least 2; the maximum of one trial is that "
                         "trial, and there is nothing to deflate")
    if not np.isfinite(sr_std) or sr_std < 0:
        raise ValueError("sr_std must be finite and non-negative")
    return float(sr_std * ((1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n)
                           + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n * np.e))))


def min_backtest_length(sharpe_annual: float, n_trials: int) -> float:
    """Years of data below which an annualised IS Sharpe of `sharpe_annual` found in
    `n_trials` trials is not evidence of anything.

    The annualised Sharpe estimated over y years of a zero-skill strategy has sd ~ 1/sqrt(y),
    so E[max SR] over N trials is expected_max_sharpe(N) / sqrt(y). Setting that equal to
    the claimed Sharpe and solving gives MinBTL = (expected_max_sharpe(N) / SR*)^2.

    Independent trials. Correlated configurations - a lookback grid on one series - have a
    smaller effective N, so this is an upper bound on the requirement and a lower bound on
    how badly an honest N is needed. PBO handles the correlated case directly.
    """
    sr = float(sharpe_annual)
    if not np.isfinite(sr) or sr <= 0:
        raise ValueError("sharpe_annual must be a positive finite number")
    return float((expected_max_sharpe(n_trials) / sr) ** 2)


def min_backtest_length_bound(sharpe_annual: float, n_trials: int) -> float:
    """The paper's own upper bound, 2*ln(N) / SR*^2, from sqrt(2 ln N) >= E[max SR]."""
    return float(2.0 * np.log(int(n_trials)) / float(sharpe_annual) ** 2)


def min_backtest_length_table(sharpe_annual: float,
                              trials: tuple[int, ...] = (2, 5, 10, 25, 50, 100, 500, 1000),
                              ) -> list[tuple[int, float, float]]:
    """[(n_trials, E[max SR] at unit sd, MinBTL years)] for one claimed Sharpe."""
    return [(n, expected_max_sharpe(n), min_backtest_length(sharpe_annual, n))
            for n in trials]


def credible(sharpe_annual: float, n_trials: int, n_obs: int,
             periods_per_year: int = 252) -> dict:
    """Does the sample actually reach the length this claim needs? One dict, no prose."""
    years = float(n_obs) / float(periods_per_year)
    need = min_backtest_length(sharpe_annual, n_trials)
    return {"sharpe_annual": float(sharpe_annual), "n_trials": int(n_trials),
            "sample_years": years, "min_backtest_length_years": need,
            "shortfall_years": need - years, "credible": bool(years >= need),
            "expected_max_sharpe_at_this_length": expected_max_sharpe(
                n_trials, 1.0 / np.sqrt(years)) if years > 0 else float("inf")}


# ======================================================================================
# CSCV: combinatorially symmetric cross-validation, and the PBO that comes out of it
# ======================================================================================
def relative_rank(rank: np.ndarray, n_configs: int, denom: str = "n_plus_1") -> np.ndarray:
    """Integer rank (1 = worst) -> relative rank w in (0, 1).

    denom='n_plus_1' : w = rank / (N + 1)      keeps the logit finite at both ends
    denom='n'        : w = rank / N            w = 1 for the OS winner -> infinite logit
    denom='midrank'  : w = (rank - 0.5) / N    the continuity-corrected variant
    """
    r = np.asarray(rank, dtype=float)
    n = int(n_configs)
    if denom == "n_plus_1":
        return r / (n + 1.0)
    if denom == "n":
        return r / n
    if denom == "midrank":
        return (r - 0.5) / n
    raise ValueError("denom must be 'n_plus_1', 'n' or 'midrank'")


def logit(w: np.ndarray) -> np.ndarray:
    """log(w / (1 - w)). Negative exactly when w < 0.5, i.e. bottom half out of sample."""
    w = np.asarray(w, dtype=float)
    with np.errstate(divide="ignore"):
        return np.log(w / (1.0 - w))


@dataclass
class CSCVResult:
    """Everything one CSCV run produced. `pbo` is the headline; the rest is the evidence."""

    pbo: float
    n_configs: int
    n_obs: int
    n_blocks: int
    n_combinations: int
    lam: np.ndarray            # (C,) logit of the IS winner's OS relative rank
    w: np.ndarray              # (C,) that relative rank
    is_best: np.ndarray        # (C,) which configuration won in sample
    is_perf: np.ndarray        # (C,) its IS performance
    os_perf: np.ndarray        # (C,) its OS performance
    os_rank: np.ndarray        # (C,) its OS rank, 1 = worst of N
    prob_loss: float           # share of splits where the IS winner LOSES out of sample
    slope: float               # OLS slope of OS on IS performance across splits
    n_distinct_winners: int
    notes: list[str] = field(default_factory=list)

    @property
    def mean_relative_rank(self) -> float:
        return float(self.w.mean())

    @property
    def null_pbo(self) -> float:
        """What PBO would be if the IS winner's OS rank were uniform. 0.5 for even N."""
        return null_pbo(self.n_configs)

    def report(self) -> str:
        w = 74
        lines = [
            "PBO via CSCV - Bailey, Borwein, Lopez de Prado & Zhu", "=" * w,
            f"  configurations N      : {self.n_configs}",
            f"  observations T        : {self.n_obs}   blocks S = {self.n_blocks}   "
            f"C(S, S/2) = {self.n_combinations}",
            f"  distinct IS winners   : {self.n_distinct_winners} of {self.n_configs}",
            "-" * w,
            f"  PBO                   : {self.pbo:.3f}   "
            f"(share of splits with logit <= 0)",
            f"  no-skill line         : {self.null_pbo:.3f}   "
            f"(uniform OS rank; not 0.5 when N is odd)",
            f"  mean OS relative rank : {self.mean_relative_rank:.3f}   "
            f"(0.5 = a coin flip)",
            f"  P(IS winner loses OS) : {self.prob_loss:.3f}",
            f"  OS-on-IS slope        : {self.slope:+.3f}   "
            f"(<= 0 means IS performance predicts OS performance backwards)",
            "-" * w,
        ]
        if self.pbo >= self.null_pbo:
            lines.append("  VERDICT: the in-sample winner is not better than a coin flip out of")
            lines.append("           sample. This is what a search over noise looks like.")
        elif self.pbo >= 0.2:
            lines.append("  VERDICT: the in-sample winner beats the median out of sample more")
            lines.append("           often than not, but not reliably.")
        else:
            lines.append("  VERDICT: the in-sample winner survives the split in most of the")
            lines.append("           C(S, S/2) ways of cutting the sample.")
        lines += [f"    ! {n}" for n in self.notes]
        return "\n".join(lines)


@lru_cache(maxsize=8)
def _combination_mask(s: int) -> np.ndarray:
    """(C, S) indicator of which blocks are IS, one row per C(S, S/2) split. Cached: the
    row set depends only on S, and rebuilding 12,870 rows per call dominated the runtime."""
    combos = np.array(list(combinations(range(s), s // 2)), dtype=int)
    mask = np.zeros((combos.shape[0], s), dtype=float)
    np.put_along_axis(mask, combos, 1.0, axis=1)
    mask.setflags(write=False)
    return mask


def null_pbo(n_configs: int) -> float:
    """The value PBO takes when the IS winner's OS rank is uniform - the "coin flip" line.

    Not 0.5 for odd N: the logit is <= 0 for rank <= (N+1)/2, and with N odd that is
    (N+1)/2 of the N ranks. N=5 gives 0.6, N=25 gives 0.52, every even N gives exactly 0.5.
    Compare a measured PBO against THIS, not against 0.5.
    """
    n = int(n_configs)
    return ceil(n / 2) / n


def _block_moments(perf: np.ndarray, n_blocks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-block count, sum and sum of squares, so every combination is a matrix product."""
    t, _ = perf.shape
    edges = np.linspace(0, t, n_blocks + 1).astype(int)
    counts = np.diff(edges).astype(float)
    sums = np.stack([perf[a:b].sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    sq = np.stack([(perf[a:b] ** 2).sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    return counts, sums, sq


def _sharpe(count: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Sharpe from the first two moments. ddof=1, and a flat series scores 0, not inf."""
    mean = s1 / count
    var = (s2 - count * mean ** 2) / (count - 1.0)
    sd = np.sqrt(np.maximum(var, 0.0))
    return np.where(sd > 0.0, mean / np.where(sd > 0.0, sd, 1.0), 0.0)


def cscv(returns, n_blocks: int = 16, denom: str = "n_plus_1",
         metric=None) -> CSCVResult:
    """Combinatorially symmetric cross-validation over a (T, N) panel of RETURNS.

    returns  : (T, N) per-period returns, one column per configuration you tried. Every
               configuration, including the ones you abandoned - that is the point.
    n_blocks : S, even. C(S, S/2) splits are formed; S=16 gives 12,870.
    denom    : how an integer rank becomes a relative rank; see `relative_rank`.
    metric   : None uses the Sharpe ratio through a fast moment path. A callable
               (rows, N) -> (N,) is applied to every IS and OS slice instead, which costs
               2 * C evaluations - use it on small S.

    The procedure: split T into S disjoint contiguous blocks; for each of the C(S, S/2)
    ways of choosing S/2 blocks as in-sample, the complement is out-of-sample. Both halves
    are the same size, which is what makes it SYMMETRIC and why IS and OS performance are
    directly comparable. n* = argmax IS performance; its OS rank among the N gives the
    relative rank w and the logit log(w/(1-w)). PBO = share of splits with logit <= 0.
    """
    perf = np.asarray(returns, dtype=float)
    if perf.ndim != 2:
        raise ValueError(f"returns must be a (T, N) panel; got shape {perf.shape}")
    t, n = perf.shape
    if n < 2:
        raise ValueError("CSCV needs at least 2 configurations; with one there is no "
                         "search to measure")
    if not np.isfinite(perf).all():
        raise ValueError("returns contains NaN/inf; decide what a missing period means "
                         "before splitting on it")
    s = int(n_blocks)
    if s < 2 or s % 2 != 0:
        raise ValueError("n_blocks must be an even integer >= 2")
    if s > t:
        raise ValueError(f"n_blocks={s} exceeds T={t}")

    mask = _combination_mask(s)
    c = mask.shape[0]

    if metric is None:
        counts, sums, sq = _block_moments(perf, s)
        n_is = mask @ counts
        is_perf = _sharpe(n_is[:, None], mask @ sums, mask @ sq)
        n_os = (1.0 - mask) @ counts
        os_perf = _sharpe(n_os[:, None], (1.0 - mask) @ sums, (1.0 - mask) @ sq)
    else:
        edges = np.linspace(0, t, s + 1).astype(int)
        blocks = [perf[a:b] for a, b in zip(edges[:-1], edges[1:])]
        is_perf = np.empty((c, n))
        os_perf = np.empty((c, n))
        for i, row in enumerate(mask):
            take = [blocks[j] for j in range(s) if row[j] > 0]
            drop = [blocks[j] for j in range(s) if row[j] == 0]
            is_perf[i] = np.asarray(metric(np.concatenate(take)), dtype=float)
            os_perf[i] = np.asarray(metric(np.concatenate(drop)), dtype=float)

    best = np.argmax(is_perf, axis=1)
    rows = np.arange(c)
    os_best = os_perf[rows, best]
    # rank 1 = worst OS. Ties count toward the top of the tie group, which is the
    # conservative direction: a tie never rescues the IS winner into the upper half.
    rank = (os_perf <= os_best[:, None]).sum(axis=1).astype(float)
    w = relative_rank(rank, n, denom)
    lam = logit(w)
    pbo = float((lam <= 0.0).mean())

    is_best_perf = is_perf[rows, best]
    slope = float(np.polyfit(is_best_perf, os_best, 1)[0]) if np.ptp(is_best_perf) > 0 else 0.0
    return CSCVResult(
        pbo=pbo, n_configs=n, n_obs=t, n_blocks=s, n_combinations=c, lam=lam, w=w,
        is_best=best, is_perf=is_best_perf, os_perf=os_best, os_rank=rank,
        prob_loss=float((os_best <= 0.0).mean()), slope=slope,
        n_distinct_winners=int(np.unique(best).size),
        notes=[f"rank rule: {denom}"])


def pbo(returns, n_blocks: int = 16) -> float:
    """Just the number, for a caller that wants one float."""
    return cscv(returns, n_blocks=n_blocks).pbo


# ======================================================================================
# Seeded data generators for the measurements below
# ======================================================================================
DAILY_VOL = 0.01
PPY = 252


def noise_panel(rng: np.random.Generator, n_obs: int, n_configs: int,
                edges_annual=()) -> np.ndarray:
    """(T, N) of independent daily returns. `edges_annual[i]` is column i's TRUE Sharpe."""
    x = rng.normal(0.0, DAILY_VOL, (n_obs, n_configs))
    for i, sr in enumerate(edges_annual):
        x[:, i] += sr * DAILY_VOL / np.sqrt(PPY)
    return x


def crossover_grid(rng: np.random.Generator, n_obs: int,
                   fasts=(5, 10, 15, 20, 30, 40, 50, 60),
                   slows=(80, 100, 120, 150, 180, 200)) -> np.ndarray:
    """A moving-average crossover grid on ONE random walk: correlated configurations.

    Every column trades the same price series, so the columns share most of their return
    and the effective number of independent trials is far below len(fasts) * len(slows).
    """
    ret = rng.normal(0.0, DAILY_VOL, n_obs + max(slows) + 1)
    price = np.exp(np.cumsum(ret))
    cols = []
    for f in fasts:
        for s in slows:
            fa = np.convolve(price, np.ones(f) / f, mode="valid")
            sa = np.convolve(price, np.ones(s) / s, mode="valid")
            m = min(len(fa), len(sa))
            pos = np.sign(fa[-m:] - sa[-m:])
            fwd = ret[-m + 1:]                     # position at t earns t -> t+1
            cols.append((pos[:-1] * fwd)[-n_obs:])
    return np.column_stack(cols)


# ======================================================================================
def _mean_se(vals) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size))


def _emax_montecarlo(n_trials: int, draws: int, rng: np.random.Generator) -> float:
    """E[max of n_trials standard normals], in chunks so the array never gets large."""
    total, done, chunk = 0.0, 0, 20_000
    while done < draws:
        k = min(chunk, draws - done)
        total += float(rng.standard_normal((k, n_trials)).max(axis=1).sum())
        done += k
    return total / draws


def main() -> None:
    bar = "=" * 78
    print("PROBABILITY OF BACKTEST OVERFITTING - measured, not asserted")
    print(bar)
    print("Every number below is produced by this file, from fixed seeds.")
    print("Rank rule: " + RANK_NOTE)

    T, S, R = 1008, 16, 24            # 4.0 years daily, C(16,8) = 12870, 24 panels per row
    print(f"\nPanel T = {T} ({T / PPY:.1f} years daily), blocks S = {S}, "
          f"C({S},{S // 2}) = {comb(S, S // 2)} splits per run, {R} independent panels "
          f"per table row.")

    # ---------------------------------------------------------------- A
    print("\n" + bar)
    print("A. PURE NOISE. Every configuration has a TRUE Sharpe of exactly zero.")
    print(bar)
    print("   PBO must sit on the no-skill line: the IS winner is a coin flip OS.")
    print("   That line is 0.5 for even N and (N+1)/2N for odd N, because the logit is")
    print("   <= 0 at rank (N+1)/2 and a rank is an integer.\n")
    print(f"  {'N':>5}{'panels':>8}{'no-skill line':>15}{'mean PBO':>10}{'std err':>9}"
          f"{'panel sd':>10}{'mean w':>9}{'P(OS loss)':>12}")
    print("  " + "-" * 78)
    for n_cfg in (5, 10, 20, 50, 100):
        reps = 60 if n_cfg <= 20 else R          # small N is cheap; buy precision there
        runs = [cscv(noise_panel(np.random.default_rng(1000 + s), T, n_cfg), S)
                for s in range(reps)]
        p = np.array([r.pbo for r in runs])
        m, se = _mean_se(p)
        print(f"  {n_cfg:>5}{reps:>8}{null_pbo(n_cfg):>15.3f}{m:>10.3f}{se:>9.3f}"
              f"{p.std(ddof=1):>10.3f}"
              f"{np.mean([r.mean_relative_rank for r in runs]):>9.3f}"
              f"{np.mean([r.prob_loss for r in runs]):>12.3f}")
    print("  " + "-" * 78)
    print("  Every row sits on its own no-skill line inside about two standard errors.")
    print("  Searching noise buys nothing out of sample - the property PBO exists to see.")
    print("  Read the 'panel sd' column too: ONE panel's PBO is a noisy estimate even")
    print(f"  though it averages {comb(S, S // 2)} splits, because all "
          f"{comb(S, S // 2)} reuse the same {T} rows.")

    demo = cscv(noise_panel(np.random.default_rng(1000), T, 50), S)
    print("\n" + demo.report())

    # ---------------------------------------------------------------- B
    print("\n" + bar)
    print("B. PLANT ONE REAL STRATEGY among 49 noise ones, and vary its edge.")
    print(bar)
    rb = R
    base = [noise_panel(np.random.default_rng(1000 + s), T, 50) for s in range(rb)]
    drift = DAILY_VOL / np.sqrt(PPY)             # one unit of annualised Sharpe per day
    print(f"  Paired: the SAME {rb} noise panels in every row - section A's N=50 panels -")
    print("  with the edge added to column 0. The 0.0 row IS that row of section A.\n")
    print(f"  {'true SR of the planted one':>28}{'PBO':>9}{'std err':>9}{'mean w':>9}"
          f"{'P(it wins IS)':>15}{'P(OS loss)':>12}")
    print("  " + "-" * 82)
    for true_sr in (0.0, 0.5, 1.0, 1.5, 2.0):
        runs = []
        for x in base:
            y = x.copy()
            y[:, 0] += true_sr * drift
            runs.append(cscv(y, S))
        m, se = _mean_se([r.pbo for r in runs])
        print(f"  {true_sr:>28.1f}{m:>9.3f}{se:>9.3f}"
              f"{np.mean([r.mean_relative_rank for r in runs]):>9.3f}"
              f"{np.mean([float((r.is_best == 0).mean()) for r in runs]):>15.3f}"
              f"{np.mean([r.prob_loss for r in runs]):>12.3f}")
    print("  " + "-" * 82)
    print("  PBO falls as the planted edge grows. Same 4 years, same 50 configurations:")
    print("  PBO measures the SEARCH, not the length of the sample.")

    # ---------------------------------------------------------------- C
    print("\n" + bar)
    print("C. ONE real strategy (true SR 1.0) plus N-1 noise ones. Grow N.")
    print(bar)
    print("   The in-sample-optimal configuration's OUT-OF-SAMPLE standing decays as the")
    print("   grid grows, because the winner is increasingly a lucky noise column.")
    print("   Nested and paired: one 200-column panel per seed, the real strategy in")
    print("   column 0, and each row keeps only the first N columns. Growing the grid is")
    print("   the only thing that changes - exactly what adding variants to a search does.\n")
    print(f"  {'N':>6}{'PBO':>9}{'mean OS rank w':>16}{'P(real one wins IS)':>21}"
          f"{'median OS SR (ann)':>20}")
    print("  " + "-" * 72)
    rc = 12
    wide = [noise_panel(np.random.default_rng(3000 + s), T, 200, edges_annual=(1.0,))
            for s in range(rc)]
    for n_cfg in (2, 5, 10, 25, 50, 100, 200):
        runs = [cscv(x[:, :n_cfg], S) for x in wide]
        print(f"  {n_cfg:>6}{np.mean([r.pbo for r in runs]):>9.3f}"
              f"{np.mean([r.mean_relative_rank for r in runs]):>16.3f}"
              f"{np.mean([float((r.is_best == 0).mean()) for r in runs]):>21.3f}"
              f"{np.mean([float(np.median(r.os_perf)) * np.sqrt(PPY) for r in runs]):>20.2f}")
    print("  " + "-" * 72)
    print("  Same real strategy, same 4 years, same true edge in every row. Only the")
    print("  number of things tried changed - and that alone decides whether the winner")
    print("  is the real one and whether it earns anything out of sample.")

    # ---------------------------------------------------------------- D
    print("\n" + bar)
    print("D. A CORRELATED GRID: 48 moving-average crossovers on ONE random walk.")
    print(bar)
    rd = 12
    runs = [cscv(crossover_grid(np.random.default_rng(4000 + s), T), S) for s in range(rd)]
    p = np.array([r.pbo for r in runs])
    m, se = _mean_se(p)
    indep = np.array([cscv(noise_panel(np.random.default_rng(1000 + s), T, 48), S).pbo
                      for s in range(rd)])
    g0 = crossover_grid(np.random.default_rng(4000), T)
    cc = np.corrcoef(g0, rowvar=False)
    off = cc[~np.eye(cc.shape[0], dtype=bool)]
    print(f"  columns                     : {g0.shape[1]} (8 fast x 6 slow lookbacks)")
    print(f"  mean pairwise correlation   : {off.mean():.3f}  "
          f"(independent columns would be ~0)")
    print(f"  PBO over {rd} random walks     : {m:.3f} +/- {se:.3f}   "
          f"(min {p.min():.3f}, max {p.max():.3f})")
    print(f"  panel sd of PBO             : {p.std(ddof=1):.3f}   vs "
          f"{indep.std(ddof=1):.3f} for 48 INDEPENDENT columns")
    print("  A random walk has no edge, so the no-skill line is still the honest answer,")
    print("  but one draw tells you much less here: correlated configurations are FEWER")
    print("  EFFECTIVE TRIALS. Report PBO with N, S and this correlation, never bare.")

    # ---------------------------------------------------------------- E
    print("\n" + bar)
    print("E. EXPECTED MAXIMUM SHARPE, and the MINIMUM BACKTEST LENGTH it implies.")
    print(bar)
    print("  Source: " + MINBTL_SOURCE)
    mc_rng = np.random.default_rng(9)
    draws = 200_000
    print(f"\n  The formula is an approximation. Against {draws:,} Monte-Carlo draws of")
    print("  max(N standard normals):\n")
    print(f"  {'N trials':>10}{'formula':>11}{'Monte Carlo':>14}{'error':>10}")
    print("  " + "-" * 45)
    for n_tr in (5, 10, 50, 100, 1000):
        emp = _emax_montecarlo(n_tr, draws, mc_rng)
        f = expected_max_sharpe(n_tr)
        print(f"  {n_tr:>10}{f:>11.4f}{emp:>14.4f}{f - emp:>+10.4f}")
    print("  " + "-" * 45)
    print("  It overstates the expected maximum slightly and always in the same direction,")
    print("  so MinBTL below is mildly conservative. That is the safe side to err on.")

    for claimed in (1.0, 2.0):
        print(f"\n  MinBTL in YEARS for a claimed annualised Sharpe of {claimed:.1f}:\n")
        print(f"  {'N trials':>10}{'E[max SR] over 1 year':>24}{'MinBTL (years)':>17}")
        print("  " + "-" * 52)
        for n_tr, emax, yrs in min_backtest_length_table(claimed):
            print(f"  {n_tr:>10}{emax:>24.3f}{yrs:>17.2f}")
        print("  " + "-" * 52)

    # ---------------------------------------------------------------- F
    print("\n" + bar)
    print("F. THE CONTRAST THAT MATTERS: your sample against what your claim needs.")
    print(bar)
    print(f"  The panel used above is {T} daily observations = {T / PPY:.1f} years.\n")
    print(f"  {'claimed SR':>12}{'N trials':>10}{'needs (yr)':>13}{'have (yr)':>12}"
          f"{'verdict':>28}")
    print("  " + "-" * 75)
    for claimed, n_tr in ((1.0, 10), (1.0, 50), (1.0, 500), (2.0, 50), (3.0, 500)):
        c = credible(claimed, n_tr, T, PPY)
        verdict = ("sample is long enough" if c["credible"]
                   else f"SHORT BY {c['shortfall_years']:.1f} YEARS")
        print(f"  {claimed:>12.1f}{n_tr:>10}{c['min_backtest_length_years']:>13.2f}"
              f"{c['sample_years']:>12.1f}{verdict:>28}")
    print("  " + "-" * 75)
    c = credible(1.0, 50, T, PPY)
    print(f"  Read the second row directly: 50 honest trials on {c['sample_years']:.1f} "
          f"years of daily")
    print(f"  data have an expected maximum Sharpe of "
          f"{c['expected_max_sharpe_at_this_length']:.2f} from noise alone, so a reported")
    print("  1.0 is BELOW what the search was expected to produce by luck.")

    # ---------------------------------------------------------------- G
    print("\n" + bar)
    print("G. HOW MUCH THE RANK CONVENTION MATTERS (the detail not pinned to a source)")
    print(bar)
    x = noise_panel(np.random.default_rng(1000), T, 51)
    print(f"  {'denominator':>14}{'PBO':>9}{'mean w':>9}{'mean logit':>13}"
          f"    (N = 51, so N+1 and N differ)")
    print("  " + "-" * 72)
    for d in ("n_plus_1", "n", "midrank"):
        r = cscv(x, S, denom=d)
        lam = r.lam[np.isfinite(r.lam)]
        print(f"  {d:>14}{r.pbo:>9.4f}{r.mean_relative_rank:>9.4f}{lam.mean():>13.4f}")
    print("  " + "-" * 72)
    print("  PBO asks only whether w <= 0.5 and the three conventions disagree about at")
    print("  most one rank, so the headline is stable. The mean LOGIT is not - denom='n'")
    print("  puts an infinite value on every split whose IS winner also wins OS.")

    # ---------------------------------------------------------------- H
    print("\n" + bar)
    print("H. REPRODUCING THE PUBLISHED NUMBERS (this is what 'verified' means here)")
    print(bar)
    print("  " + PBO_SOURCE)
    print(f"\n  C(16,8) computed here                : {comb(16, 8)}")
    print(f"  C(24,12) and C(12,6), for comparison : {comb(24, 12)}, {comb(12, 6)}")
    print("\n  Notices of the AMS 2014 gives three anchors in prose. All three reproduce:\n")
    print(f"  {'the paper says':<58}{'this file':>12}")
    print("  " + "-" * 72)
    print(f"  {'N = 10 configurations -> expected max Sharpe 1.57':<58}"
          f"{expected_max_sharpe(10):>12.4f}")
    print(f"  {'5 years of data supports at most 45 configurations':<58}"
          f"{min_backtest_length(1.0, 45):>12.3f}")
    print(f"  {'7 configurations already need a 2-year backtest':<58}"
          f"{min_backtest_length(1.0, 7):>12.3f}")
    print("  " + "-" * 72)
    print("  (The last two are MinBTL in years at an expected max Sharpe of 1.0.)")
    print(f"\n  {'N':>8}{'MinBTL (years)':>17}{'2*ln(N)/SR^2 bound':>22}")
    print("  " + "-" * 48)
    for n_tr in (10, 100, 1000):
        print(f"  {n_tr:>8}{min_backtest_length(1.0, n_tr):>17.2f}"
              f"{min_backtest_length_bound(1.0, n_tr):>22.2f}")
    print("  " + "-" * 48)
    print("  The bound holds in every row, as Theorem 2 requires.")
    print("\n  The PBO paper's own two worked examples, for calibration: an overfit")
    print("  seasonal rule on a random walk scored PBO = 55%, a genuine planted monthly")
    print("  effect scored 13%, and it recommends rejecting a model whose PBO exceeds 5%.")
    print("  That 5% threshold is far stricter than 'below the 0.5 coin-flip line'.")

    print("\n" + bar)
    print("RULE: PBO means nothing without its own no-skill line (0.5 for even N), the "
          "trial count N,")
    print("the block count S and the configuration correlation beside it - and a Sharpe "
          "whose MinBTL")
    print("at the HONEST N exceeds the sample you actually have is not evidence at all.")
    print(bar)


if __name__ == "__main__":
    main()
