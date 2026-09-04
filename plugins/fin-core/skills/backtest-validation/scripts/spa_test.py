#!/usr/bin/env python3
"""Guarded wrappers for `arch.bootstrap`'s data-snooping tests -- SPA, StepM, MCS.

🚨 THE TRAP THIS FILE EXISTS TO CLOSE:
    `arch.bootstrap`'s SPA, RealityCheck, StepM and MCS all take **LOSSES**, where LOWER
    IS BETTER. Verified against `arch/bootstrap/multiple_comparison.py`.

    Passing RETURNS does not raise. It does not warn. It INVERTS the test: your worst
    strategy is identified as the best, `better_models` names the ones you should throw
    away, and the p-value you quote answers the opposite hypothesis. `__main__` measures
    this below -- fed returns instead of losses, the procedure picks the single worst of
    ten strategies and reports it as the winner.

    losses = -returns          # for a return series, that is the entire conversion

So every function here takes RETURNS in the natural orientation -- higher is better -- and
negates internally, exactly once, at a single line you can read. Anything already
loss-shaped (squared forecast error, negative log-likelihood, absolute error) must NOT go
through these wrappers; hand it to `arch` directly.

Two smaller traps, also closed here:
  * `SPA.pvalues` returns THREE p-values -- `lower`, `consistent`, `upper`. `lower` is
    liberal and `upper` conservative; **`consistent` is the one to report**. Indexing
    `.pvalues[0]` or reading whichever is smallest is silent p-hacking.
  * `block_size` defaults to `int(sqrt(T))`. Preserving serial dependence is the entire
    reason for a block bootstrap, so it is set from `optimal_block_length` here whenever
    the caller does not specify one.

`import arch` is LAZY -- inside the functions -- so this module imports, and its demo
runs, on a machine without arch. If arch is missing you get a message naming the package
rather than an ImportError from three frames down.

Usage:
    from spa_test import spa_test, stepm_test, mcs_test

    res = spa_test(bench_returns, model_returns_df)   # RETURNS, not losses
    print(res.report())
    res.pvalue                       # the `consistent` p-value

    mcs = mcs_test(model_returns_df, size=0.10)
    mcs.included                     # models indistinguishable from the best

WHICH TEST ANSWERS WHICH QUESTION (see ../references/significance-tests.md):
    is the BEST of N better than a benchmark, correcting for the search?  -> SPA
    WHICH ones beat the benchmark, under family-wise error control?       -> StepM
    which models are indistinguishable from the best? (no benchmark)      -> MCS
All three need the per-period series of every candidate you tried, including the ones you
abandoned. If you kept only the winner's equity curve, none of them are available to you
and DSR is the fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ARCH_HINT = (
    "`arch` is not installed. SPA / StepM / MCS live in `arch.bootstrap`:\n"
    "    pip install arch\n"
    "arch 8.0.0, by Kevin Sheppard (Oxford). Licence NCSA -- permissive and BSD-like, but\n"
    "not one of the usual three, so flag it in a licence audit (GitHub reports\n"
    "NOASSERTION). Nothing else on PyPI implements Hansen's SPA correctly."
)


class ArchMissing(ImportError):
    """Raised when a wrapper needs `arch` and it is not installed."""


def _require_arch():
    """Lazy import, so this module is usable and testable without arch present."""
    try:
        import arch.bootstrap as ab
    except ImportError as exc:                       # pragma: no cover - env dependent
        raise ArchMissing(ARCH_HINT) from exc
    return ab


# --------------------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------------------
def _as_matrix(x, name: str) -> tuple[np.ndarray, list[str]]:
    """Coerce model returns to (T, k) float with column labels."""
    if isinstance(x, pd.DataFrame):
        arr, cols = x.to_numpy(dtype=float), [str(c) for c in x.columns]
    elif isinstance(x, pd.Series):
        arr, cols = x.to_numpy(dtype=float)[:, None], [str(x.name or f"{name}_0")]
    else:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        cols = [f"{name}_{i}" for i in range(arr.shape[1])]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D; got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/inf. Decide what a missing period means "
                         f"before bootstrapping over it -- dropping rows breaks the "
                         f"serial dependence the block bootstrap exists to preserve.")
    return arr, cols


def _as_vector(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a single series; got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/inf")
    return arr


def choose_block_size(losses: np.ndarray, override: int | None = None) -> tuple[int, str]:
    """Stationary-bootstrap block length, from `optimal_block_length` where available.

    Returns (block_size, provenance). Falls back to T**(1/3) -- the Politis-Romano rate --
    rather than arch's `sqrt(T)` default if `optimal_block_length` cannot be reached, and
    says so in the provenance string so the choice is never invisible.
    """
    t = int(losses.shape[0])
    if override is not None:
        return max(1, int(override)), "caller-specified"
    try:
        ab = _require_arch()
        obl = ab.optimal_block_length(losses)
        bs = int(np.ceil(float(np.asarray(obl["stationary"]).max())))
        return int(np.clip(bs, 1, max(1, t // 2))), "arch.optimal_block_length (stationary)"
    except Exception:
        bs = int(np.clip(int(np.ceil(t ** (1.0 / 3.0))), 1, max(1, t // 2)))
        return bs, "fallback T**(1/3) (optimal_block_length unavailable)"


# --------------------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------------------
@dataclass
class SPAResult:
    pvalue: float                       # the `consistent` p-value -- the one to report
    pvalues: dict[str, float]
    best_model: str
    better_models: list[str]
    mean_excess: pd.Series              # per model, mean return minus benchmark return
    block_size: int
    block_source: str
    reps: int
    n_obs: int
    model_names: list[str]
    source: str
    notes: list[str] = field(default_factory=list)

    def report(self, alpha: float = 0.05) -> str:
        w = 76
        lines = ["SPA - HANSEN (2005) SUPERIOR PREDICTIVE ABILITY", "=" * w,
                 f"  candidates tested   : {len(self.model_names)}   "
                 f"observations: {self.n_obs}",
                 f"  bootstrap           : {self.reps} reps, stationary, "
                 f"block {self.block_size} ({self.block_source})",
                 f"  implementation      : {self.source}",
                 f"  best by mean return : {self.best_model} "
                 f"({self.mean_excess.max():+.5f} per period vs benchmark)",
                 "-" * w]
        for k, v in self.pvalues.items():
            tag = "  <- REPORT THIS" if k == "consistent" else ""
            lines.append(f"  p({k:<10}) = {v:.4f}{tag}")
        lines.append("-" * w)
        verdict = ("REJECT H0: at least one candidate genuinely beats the benchmark"
                   if self.pvalue < alpha else
                   "CANNOT REJECT H0: the best of these is consistent with luck")
        lines.append(f"  VERDICT @ {alpha:.0%}        : {verdict}")
        if self.better_models:
            lines.append(f"  survivors           : {', '.join(self.better_models)}")
        lines += [f"    ! {n}" for n in self.notes]
        return "\n".join(lines)


@dataclass
class MCSResult:
    included: list[str]
    excluded: list[str]
    pvalues: pd.Series
    size: float
    block_size: int
    block_source: str
    reps: int
    n_obs: int
    source: str

    def report(self) -> str:
        w = 76
        return "\n".join([
            "MCS - HANSEN-LUNDE-NASON (2011) MODEL CONFIDENCE SET", "=" * w,
            f"  candidates          : {len(self.included) + len(self.excluded)}   "
            f"observations: {self.n_obs}",
            f"  bootstrap           : {self.reps} reps, block {self.block_size} "
            f"({self.block_source})",
            f"  size                : {self.size:.0%}",
            "-" * w,
            f"  INCLUDED ({len(self.included):>2})        : "
            f"{', '.join(self.included) if self.included else '-'}",
            f"  excluded ({len(self.excluded):>2})        : "
            f"{', '.join(self.excluded) if self.excluded else '-'}",
            "-" * w,
            "  'Included' means NOT DISTINGUISHABLE from the best at this size -- it is",
            "  not a ranking, and a large included set means low power, not many winners.",
        ])


# --------------------------------------------------------------------------------------
# the wrappers -- RETURNS in, negation internal
# --------------------------------------------------------------------------------------
def spa_test(benchmark_returns,
             model_returns,
             reps: int = 1000,
             block_size: int | None = None,
             seed: int | None = None,
             studentize: bool = True,
             nested: bool = False) -> SPAResult:
    """Hansen's SPA on RETURNS (higher is better). Negates to losses internally.

    benchmark_returns : (T,) per-period returns of the benchmark / incumbent strategy.
    model_returns     : (T, k) per-period returns of every candidate you tried -- including
                        the ones you abandoned. Omitting them is what SPA corrects for, so
                        omitting them defeats the test.
    reps              : bootstrap replications. Cost is O(reps x T x k).
    block_size        : stationary-bootstrap block length; from `optimal_block_length`
                        when None.

    H0: no candidate is better than the benchmark. A small `consistent` p-value rejects it.
    """
    bench = _as_vector(benchmark_returns, "benchmark_returns")
    models, names = _as_matrix(model_returns, "model")
    if models.shape[0] != bench.shape[0]:
        raise ValueError(f"benchmark has {bench.shape[0]} periods, models have "
                         f"{models.shape[0]}; they must be aligned on the same calendar")

    loss_bench = -bench                       # 🚨 THE NEGATION. Once, here, visible.
    loss_models = -models
    bs, src = choose_block_size(loss_models, block_size)

    ab = _require_arch()
    spa = ab.SPA(loss_bench, loss_models, block_size=bs, reps=reps,
                 bootstrap="stationary", studentize=studentize, nested=nested, seed=seed)
    spa.compute()
    pv = {str(k): float(v) for k, v in dict(spa.pvalues).items()}

    excess = pd.Series(models.mean(axis=0) - bench.mean(), index=names)
    try:
        better = [names[i] for i in np.atleast_1d(np.asarray(spa.better_models(0.05)))
                  if isinstance(i, (int, np.integer)) and 0 <= int(i) < len(names)]
    except Exception:
        better = []

    return SPAResult(
        pvalue=pv.get("consistent", float("nan")), pvalues=pv,
        best_model=str(excess.idxmax()), better_models=better, mean_excess=excess,
        block_size=bs, block_source=src, reps=reps, n_obs=int(models.shape[0]),
        model_names=names, source="arch.bootstrap.SPA",
        notes=["`lower` and `upper` bracket the influence of dominated models on the null; "
               "they are diagnostics, not alternatives to `consistent`."])


def stepm_test(benchmark_returns,
               model_returns,
               size: float = 0.05,
               reps: int = 1000,
               block_size: int | None = None,
               seed: int | None = None) -> list[str]:
    """Romano-Wolf StepM on RETURNS. Returns the names that beat the benchmark under FWER.

    SPA says whether ANY candidate is genuinely superior; StepM says WHICH, controlling
    the family-wise error rate across the whole set you tried.
    """
    bench = _as_vector(benchmark_returns, "benchmark_returns")
    models, names = _as_matrix(model_returns, "model")
    if models.shape[0] != bench.shape[0]:
        raise ValueError("benchmark and models must cover the same periods")

    loss_bench, loss_models = -bench, -models        # 🚨 THE NEGATION.
    bs, _ = choose_block_size(loss_models, block_size)

    ab = _require_arch()
    frame = pd.DataFrame(loss_models, columns=names)
    stepm = ab.StepM(pd.Series(loss_bench), frame, size=size, block_size=bs, reps=reps,
                     seed=seed)
    stepm.compute()
    return [str(m) for m in stepm.superior_models]


def mcs_test(model_returns,
             size: float = 0.10,
             reps: int = 1000,
             block_size: int | None = None,
             seed: int | None = None,
             method: str = "R") -> MCSResult:
    """Hansen-Lunde-Nason Model Confidence Set on RETURNS. NO benchmark argument.

    Returns the set of candidates not distinguishable from the best at `size`. Needs at
    least 2 candidates -- arch raises below that.
    """
    models, names = _as_matrix(model_returns, "model")
    if models.shape[1] < 2:
        raise ValueError("MCS needs at least 2 candidates; with one there is nothing to "
                         "compare it against (use PSR/DSR instead)")

    losses = -models                                  # 🚨 THE NEGATION.
    bs, src = choose_block_size(losses, block_size)

    ab = _require_arch()
    frame = pd.DataFrame(losses, columns=names)
    mcs = ab.MCS(frame, size=size, reps=reps, block_size=bs, method=method, seed=seed)
    mcs.compute()
    included = [str(m) for m in np.atleast_1d(np.asarray(mcs.included)).tolist()]
    excluded = [str(m) for m in np.atleast_1d(np.asarray(mcs.excluded)).tolist()]
    return MCSResult(included=included, excluded=excluded,
                     pvalues=pd.Series(mcs.pvalues.squeeze()), size=size,
                     block_size=bs, block_source=src, reps=reps,
                     n_obs=int(models.shape[0]), source="arch.bootstrap.MCS")


# --------------------------------------------------------------------------------------
# self-contained fallback -- so the demo runs, and only for that
# --------------------------------------------------------------------------------------
def _stationary_bootstrap_indices(n: int, block_size: int, reps: int,
                                  rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices, (reps, n).

    Each step either continues the current block (wrapping at the end) or jumps to a fresh
    uniform draw, with jump probability 1/block_size. Geometric block lengths are what make
    the resampled series stationary, which the fixed-length moving-block bootstrap is not.
    """
    p = 1.0 / max(block_size, 1)
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, reps)
    for t in range(1, n):
        jump = rng.random(reps) < p
        idx[:, t] = np.where(jump, rng.integers(0, n, reps), (idx[:, t - 1] + 1) % n)
    return idx


def spa_test_selfcontained(benchmark_returns,
                           model_returns,
                           reps: int = 1000,
                           block_size: int | None = None,
                           seed: int | None = None,
                           already_losses: bool = False) -> SPAResult:
    """Hansen's SPA_consistent, implemented here so `__main__` runs without arch.

    ⚠️ `arch.bootstrap.SPA` is the reference implementation -- use it in real work. This
    exists so the demonstration below is runnable on a bare numpy/pandas install, and it
    covers only the consistent p-value: no StepM, no MCS, no nested-model handling.

    `already_losses=True` skips the negation, and is used in the demo to show what the
    procedure does when it is fed returns as though they were losses.
    """
    bench = _as_vector(benchmark_returns, "benchmark_returns")
    models, names = _as_matrix(model_returns, "model")
    n, k = models.shape

    lb = bench if already_losses else -bench
    lm = models if already_losses else -models
    d = lb[:, None] - lm                       # (T, k); positive = model has lower loss
    dbar = d.mean(axis=0)

    bs = int(block_size) if block_size else int(np.clip(int(np.ceil(n ** (1 / 3))), 1, n // 2))
    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_indices(n, bs, reps, rng)
    dboot = d[idx].mean(axis=1)                # (reps, k)

    omega = dboot.std(axis=0, ddof=1) * np.sqrt(n)
    omega = np.where(omega > 0, omega, np.inf)     # a constant differential never rejects
    z = np.sqrt(n) * dbar / omega

    # Hansen's CONSISTENT recentring: models far enough below zero are excluded from the
    # null distribution. `lower` keeps only dbar >= 0, `upper` keeps everything.
    thresh = -np.sqrt(2.0 * np.log(max(np.log(n), 1.0000001)))
    g_cons = np.where(z >= thresh, dbar, 0.0)
    g_low = np.where(dbar >= 0.0, dbar, 0.0)
    g_up = dbar

    t_obs = max(0.0, float(np.max(z)))
    pv: dict[str, float] = {}
    tb_cons = np.zeros(reps)
    for label, g in (("lower", g_low), ("consistent", g_cons), ("upper", g_up)):
        zb = np.sqrt(n) * (dboot - g[None, :]) / omega[None, :]
        tb = np.maximum(0.0, zb.max(axis=1))
        pv[label] = float((tb >= t_obs).mean())
        if label == "consistent":
            tb_cons = tb

    # single-step Romano-Wolf: candidates clearing the 95th percentile of the max
    # distribution. A proper StepM iterates after dropping these; use arch for that.
    crit = float(np.quantile(tb_cons, 0.95))
    excess = pd.Series(models.mean(axis=0) - bench.mean(), index=names)
    return SPAResult(
        pvalue=pv["consistent"], pvalues=pv, best_model=str(excess.idxmax()),
        better_models=[names[i] for i in range(k) if z[i] > crit],
        mean_excess=excess, block_size=bs, block_source="T**(1/3)", reps=reps, n_obs=n,
        model_names=names, source="self-contained stationary bootstrap (arch absent)",
        notes=["teaching implementation -- use arch.bootstrap.SPA in real work",
               "`better_models` here is single-step only; arch's StepM is the real thing"])


def naive_best_of_n_pvalue(benchmark_returns, model_returns) -> tuple[str, float]:
    """What people actually do: t-test the winner, as if it had been the only candidate.

    Returns (best model name, one-sided p-value). This is the number SPA corrects.
    """
    from scipy import stats
    bench = _as_vector(benchmark_returns, "benchmark_returns")
    models, names = _as_matrix(model_returns, "model")
    d = models - bench[:, None]
    n = d.shape[0]
    t = d.mean(axis=0) / (d.std(axis=0, ddof=1) / np.sqrt(n))
    best = int(np.argmax(t))
    return names[best], float(stats.t.sf(t[best], n - 1))


# --------------------------------------------------------------------------------------
# offline demo -- runs with or without arch installed
# --------------------------------------------------------------------------------------
def _panel(rng: np.random.Generator, n: int = 750, k: int = 10,
           edge: float = 0.0, edge_on: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Benchmark plus k candidates. Only candidate `edge_on` has a real edge, size `edge`.

    All candidates share a common market factor, so the loss differentials are correlated
    across candidates -- which is the realistic case and the one a Bonferroni correction
    handles badly and the bootstrap handles properly.
    """
    common = rng.normal(0.0002, 0.008, n)
    bench = common + rng.normal(0.0, 0.003, n)
    models = common[:, None] + rng.normal(0.0, 0.006, (n, k))
    models[:, edge_on] += edge
    return bench, models, [f"strat_{i:02d}" for i in range(k)]


if __name__ == "__main__":
    BAR = "!" * 78
    print(BAR)
    print("!!  arch.bootstrap's SPA, RealityCheck, StepM and MCS all take LOSSES.")
    print("!!  LOWER IS BETTER. Passing returns does not raise, does not warn, and")
    print("!!  INVERTS the test.        losses = -returns")
    print("!!  Every wrapper in this file takes RETURNS and negates internally.")
    print(BAR)

    try:
        _require_arch()
        HAVE_ARCH = True
    except ArchMissing as exc:
        HAVE_ARCH = False
        print(f"\n`arch` is NOT installed in this environment, so the demo below runs the")
        print(f"self-contained stationary bootstrap instead. The wrappers themselves are")
        print(f"unexercised here; with arch present they would run identically.\n")
        print("  " + str(exc).replace("\n", "\n  "))

    runner = spa_test if HAVE_ARCH else spa_test_selfcontained
    REPS, SEED = 500, 12

    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("=== A. TEN CANDIDATES, NONE OF THEM ANY GOOD (true edge = 0 for all ten) ===")
    print("=" * 78)
    rng = np.random.default_rng(3)
    bench, models, names = _panel(rng, edge=0.0)
    mdf = pd.DataFrame(models, columns=names)

    win, p_naive = naive_best_of_n_pvalue(bench, mdf)
    res_a = runner(bench, mdf, reps=REPS, seed=SEED)
    print(f"  naive t-test on the winner ({win}) : p = {p_naive:.4f}")
    print(f"  SPA, correcting for all 10 trials    : p = {res_a.pvalue:.4f}")
    print()
    print(res_a.report())

    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("=== B. SAME SETUP, ONE CANDIDATE GENUINELY SUPERIOR (strat_04, +12bp/day) ===")
    print("=" * 78)
    rng = np.random.default_rng(3)
    bench_b, models_b, names_b = _panel(rng, edge=0.0012, edge_on=4)
    mdf_b = pd.DataFrame(models_b, columns=names_b)
    win_b, p_naive_b = naive_best_of_n_pvalue(bench_b, mdf_b)
    res_b = runner(bench_b, mdf_b, reps=REPS, seed=SEED)
    print(f"  naive t-test on the winner ({win_b}) : p = {p_naive_b:.4f}")
    print(f"  SPA, correcting for all 10 trials    : p = {res_b.pvalue:.4f}")
    print()
    print(res_b.report())
    print(f"\n  SPA still finds it. The correction costs power, it does not destroy it --")
    print(f"  a real edge survives the search adjustment, which is the point.")

    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("=== C. HOW OFTEN EACH APPROACH CRIES WOLF (null panels, repeated) ===")
    print("=" * 78)
    MC = 150
    nv, sp = np.empty(MC), np.empty(MC)
    for s in range(MC):
        r = np.random.default_rng(50_000 + s)
        b, m, nm = _panel(r, edge=0.0)
        f = pd.DataFrame(m, columns=nm)
        nv[s] = naive_best_of_n_pvalue(b, f)[1]
        sp[s] = spa_test_selfcontained(b, f, reps=250, seed=s).pvalue
    print(f"  {MC} independent null panels, 10 candidates each, nominal 5%:\n")
    print(f"  {'procedure':<44}{'false positives':>16}")
    print("  " + "-" * 60)
    print(f"  {'naive t-test on the best of 10':<44}{(nv < 0.05).mean():>15.1%}")
    print(f"  {'SPA (consistent p-value)':<44}{(sp < 0.05).mean():>15.1%}")
    print("  " + "-" * 60)
    print(f"  Testing the winner as though you had not searched turns a 5% test into a")
    print(f"  {(nv < 0.05).mean():.0%} one. That is the multiple-comparison correction, "
          f"measured.")

    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("=== D. THE TRAP ITSELF: the same data, fed as RETURNS instead of LOSSES ===")
    print("=" * 78)
    ranked = res_b.mean_excess.sort_values(ascending=False)
    truly_best, truly_worst = str(ranked.index[0]), str(ranked.index[-1])
    wrong = spa_test_selfcontained(bench_b, mdf_b, reps=REPS, seed=SEED,
                                   already_losses=True)
    print(f"  true best candidate by mean return   : {truly_best} "
          f"({ranked.iloc[0]:+.5f}/period)")
    print(f"  true worst candidate                 : {truly_worst} "
          f"({ranked.iloc[-1]:+.5f}/period)")
    print()
    d_correct = (-bench_b)[:, None] - (-models_b)
    d_wrong = bench_b[:, None] - models_b
    print(f"  correctly negated, the procedure ranks first : "
          f"{names_b[int(np.argmax(d_correct.mean(axis=0)))]}")
    print(f"  fed returns as losses, it ranks first        : "
          f"{names_b[int(np.argmax(d_wrong.mean(axis=0)))]}")
    print(f"  and reports p = {wrong.pvalue:.4f} for that inverted hypothesis "
          f"(correct run: p = {res_b.pvalue:.4f})")
    print()
    print("  It picked the WORST strategy of the ten and called it the best. No error, no")
    print("  warning, a publishable-looking p-value. That is why the negation in this file")
    print("  happens once, in one place, on a line you can read.")
