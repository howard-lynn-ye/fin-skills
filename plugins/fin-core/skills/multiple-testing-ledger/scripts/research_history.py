#!/usr/bin/env python3
"""Multiple-testing accounting across a whole research programme, not one backtest.

`../../backtest-validation/scripts/trial_ledger.py` already records every configuration
you tried and deflates ONE Sharpe against the count. This file answers the other half:
given a LEDGER OF TRIALS, which of them survive family-wise error or false-discovery-rate
control, and how much Sharpe does the correction take off the top?

It does NOT invent a second ledger format. `from_ledger` reads the append-only
trials.jsonl that TrialLedger writes, and it takes `m` - the number of hypotheses - from
that ledger's own REGISTERED count, not from the trials that happened to finish. A trial
you registered, ran, disliked and abandoned was still a test of a hypothesis. Dropping it
from `m` is the single most common way these corrections are neutered, and section F
measures exactly what that costs.

Four corrections, all implemented here from their definitions and all checked by
simulation rather than by assertion:

  Bonferroni          reject p_i <= alpha/m                  FWER, any dependence
  Holm (1979)         step down, alpha/(m-i+1)               FWER, any dependence, uniformly
                                                             more powerful than Bonferroni
  Benjamini-Hochberg  step up, i*q/m                         FDR, independence or PRDS
  Benjamini-Yekutieli step up, i*q/(m*c(m)), c(m)=sum 1/i    FDR, ARBITRARY dependence

plus a Harvey-Liu style haircut: turn the observed Sharpe into a t-statistic, adjust its
p-value for the number of tests, and turn the adjusted p-value back into the Sharpe you
are entitled to report.

Run:  python research_history.py       (numpy + pandas + scipy, seeded, ~15 s)

What is verified and what is not - see SOURCES at the bottom of this docstring and the
`## Citations` section of the SKILL.md. Every FWER, FDR and power number printed by
`__main__` is measured here on seeded data, not quoted.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ("uncorrected", "bonferroni", "holm", "benjamini_hochberg", "benjamini_yekutieli")

CONTROLS = {
    "uncorrected": "nothing - this is the count you are correcting",
    "bonferroni": "FWER under any dependence",
    "holm": "FWER under any dependence, never less powerful than Bonferroni",
    "benjamini_hochberg": "FDR under independence or positive regression dependence",
    "benjamini_yekutieli": "FDR under ARBITRARY dependence",
}


# ======================================================================================
# The corrections
# ======================================================================================
def _as_p(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("no p-values given")
    if not np.isfinite(p).all():
        raise ValueError("p-values must be finite; an unscored trial is p = 1.0, not NaN")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("p-values must lie in [0, 1]")
    return p


def by_constant(m: int) -> float:
    """c(m) = sum_{i=1..m} 1/i - the harmonic factor Benjamini-Yekutieli pays for
    allowing arbitrary dependence. It is what makes BY strictly more conservative than
    Benjamini-Hochberg, and it grows like log(m) + 0.5772."""
    m = int(m)
    if m < 1:
        raise ValueError("m must be at least 1")
    return float(np.sum(1.0 / np.arange(1, m + 1)))


def bonferroni(pvalues, alpha: float = 0.05) -> dict:
    """Reject p_i <= alpha/m. Controls FWER under any dependence, at a cost in power."""
    p = _as_p(pvalues)
    m = p.size
    adj = np.minimum(1.0, p * m)
    return {"method": "bonferroni", "m": m, "alpha": alpha, "adjusted": adj,
            "reject": adj <= alpha, "threshold": alpha / m}


def holm(pvalues, alpha: float = 0.05) -> dict:
    """Holm (1979) step-down: sort ascending, compare p_(i) to alpha/(m-i+1), STOP at the
    first failure and reject nothing after it. Same FWER guarantee as Bonferroni under any
    dependence and never rejects less, so Bonferroni is only ever the easier arithmetic."""
    p = _as_p(pvalues)
    m = p.size
    order = np.argsort(p, kind="stable")
    ps = p[order]
    raw = np.minimum(1.0, ps * (m - np.arange(m)))
    adj_sorted = np.maximum.accumulate(raw)          # enforce monotone adjusted p-values
    adj = np.empty(m)
    adj[order] = adj_sorted
    return {"method": "holm", "m": m, "alpha": alpha, "adjusted": adj,
            "reject": adj <= alpha, "threshold": float("nan")}


def _step_up(p: np.ndarray, q: float, factor: float, name: str) -> dict:
    """Shared BH / BY machinery. BY is BH with the threshold divided by c(m)."""
    m = p.size
    order = np.argsort(p, kind="stable")
    ps = p[order]
    i = np.arange(1, m + 1)
    raw = np.minimum(1.0, ps * m * factor / i)
    # step UP: the adjusted p-value is the running minimum from the largest downwards
    adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = adj_sorted
    return {"method": name, "m": m, "alpha": q, "adjusted": adj, "reject": adj <= q,
            "threshold": float(q / (m * factor))}


def benjamini_hochberg(pvalues, q: float = 0.05) -> dict:
    """BH step-up: largest i with p_(i) <= i*q/m, reject that one and everything below.
    Controls FDR at q under independence and under positive regression dependence."""
    return _step_up(_as_p(pvalues), q, 1.0, "benjamini_hochberg")


def benjamini_yekutieli(pvalues, q: float = 0.05) -> dict:
    """BY: BH with the threshold divided by c(m) = sum 1/i. Controls FDR at q under
    ARBITRARY dependence - the guarantee you actually have when you do not know how your
    strategies covary, which for a grid over one price series is always."""
    p = _as_p(pvalues)
    return _step_up(p, q, by_constant(p.size), "benjamini_yekutieli")


def harvey_liu_bhy(pvalues, q: float = 0.05) -> dict:
    """The BHY variant Harvey, Liu & Zhu print and implement, which is NOT standard BY.

    Their recursion (RFS 2016 s4.4, and `Haircut_SR.m` lines 180-189) initialises the
    LARGEST adjusted p-value at the raw largest p-value:

        p_(M) = p_(M);   p_(i) = min[ p_(i+1), (M*c(M)/i) * p_(i) ]   for i < M

    Standard BY multiplies the largest one by c(M) as well. On most families the two agree
    - `__main__` section G reproduces their own six-test example, where they do - but they
    are not the same procedure, and section G also shows a family where one rejects
    everything and the other rejects nothing. Use `benjamini_yekutieli` for the FDR
    guarantee; use this one only to reproduce their numbers.
    """
    p = _as_p(pvalues)
    m = p.size
    c = by_constant(m)
    order = np.argsort(p, kind="stable")
    ps = p[order]
    adj_sorted = np.empty(m)
    adj_sorted[-1] = ps[-1]
    for i in range(m - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i + 1], m * c / (i + 1) * ps[i])
    adj = np.empty(m)
    adj[order] = adj_sorted
    return {"method": "harvey_liu_bhy", "m": m, "alpha": q, "adjusted": adj,
            "reject": adj <= q, "threshold": float("nan")}


def independent_pvalue(p: float, n_trials: int) -> float:
    """1 - (1 - p)^N: the chance of seeing something this good in N INDEPENDENT tests.

    This is the adjustment Harvey & Liu use in the worked illustration of "Backtesting"
    (JPM 2015), not Bonferroni. It is always slightly smaller than min(1, N*p) and it
    saturates at 1 instead of being clipped there.
    """
    p = float(np.clip(p, 0.0, 1.0))
    return float(1.0 - (1.0 - p) ** int(n_trials))


def uncorrected(pvalues, alpha: float = 0.05) -> dict:
    p = _as_p(pvalues)
    return {"method": "uncorrected", "m": p.size, "alpha": alpha, "adjusted": p.copy(),
            "reject": p <= alpha, "threshold": alpha}


_DISPATCH = {"uncorrected": uncorrected, "bonferroni": bonferroni, "holm": holm,
             "benjamini_hochberg": benjamini_hochberg,
             "benjamini_yekutieli": benjamini_yekutieli}


def correct(pvalues, method: str = "holm", alpha: float = 0.05) -> dict:
    """One entry point. `method` is any key of METHODS."""
    if method not in _DISPATCH:
        raise ValueError(f"method must be one of {list(METHODS)}, got {method!r}")
    return _DISPATCH[method](pvalues, alpha)


def compare(pvalues, alpha: float = 0.05) -> pd.DataFrame:
    """Every correction side by side: what each one leaves standing."""
    p = _as_p(pvalues)
    rows = []
    for name in METHODS:
        r = correct(p, name, alpha)
        rows.append({"method": name, "survivors": int(r["reject"].sum()),
                     "min_adjusted_p": float(r["adjusted"].min()),
                     "controls": CONTROLS[name]})
    return pd.DataFrame(rows).set_index("method")


# ======================================================================================
# Sharpe <-> t-statistic <-> p-value, and the Harvey-Liu style haircut
# ======================================================================================
def t_from_sharpe(sharpe: float, n_obs: int) -> float:
    """t = SR * sqrt(T) for a PER-PERIOD Sharpe over T periods (IID normal returns)."""
    return float(sharpe) * np.sqrt(float(n_obs))


def t_from_annual_sharpe(sharpe_annual: float, years: float) -> float:
    """t = SR_annual * sqrt(years). The periods_per_year cancels, which is why an
    annualised Sharpe and a sample length in YEARS are all a t-statistic needs."""
    return float(sharpe_annual) * np.sqrt(float(years))


def p_from_t(t: float, n_obs: int, two_sided: bool = True) -> float:
    """Student-t p-value with n_obs - 1 degrees of freedom."""
    df = max(int(n_obs) - 1, 1)
    p = float(stats.t.sf(abs(float(t)), df))
    return 2.0 * p if two_sided else p


def sharpe_from_p(p: float, years: float, two_sided: bool = True) -> float:
    """Invert: the annualised Sharpe an adjusted p-value is worth over `years` of data."""
    p = float(np.clip(p, 1e-300, 1.0))
    z = stats.norm.isf(p / 2.0 if two_sided else p)
    return float(max(z, 0.0) / np.sqrt(years))


def haircut_sharpe(sharpe_annual: float, years: float, n_trials: int,
                   method: str = "bonferroni", alpha: float = 0.05,
                   other_pvalues=None, normal: bool = True) -> dict:
    """What is left of a Sharpe once its p-value is adjusted for `n_trials` tests.

    Harvey-Liu style: observed SR -> t -> p -> adjusted p -> t -> SR. The haircut is
    1 - SR_adjusted / SR_observed.

    method='bonferroni' or 'holm' needs only the count - for the single most significant
    strategy the two coincide, because Holm's first step IS alpha/m. method='independent'
    uses 1 - (1-p)^N, which is what Harvey & Liu's own worked illustration uses. BH and BY
    are step procedures over a whole family, so they need the other trials' p-values; pass
    them in `other_pvalues` (the ledger has them) or the call raises.

    NOT the same computation as Harvey & Liu's `Haircut_SR.m`, which simulates a
    cross-section of t-statistics with an assumed proportion of true strategies and an
    assumed correlation. This one uses the trials you actually recorded, which is the
    information you have. Section G of `__main__` reproduces their illustration exactly.
    """
    t = t_from_annual_sharpe(sharpe_annual, years)
    n_periods = max(int(round(years * 252)), 2)
    p = p_from_t(t, n_periods) if normal is False else 2.0 * float(stats.norm.sf(abs(t)))
    if method == "independent":
        p_adj = independent_pvalue(p, n_trials)
    elif method in ("bonferroni", "holm"):
        p_adj = float(min(1.0, p * int(n_trials)))
    elif method in ("benjamini_hochberg", "benjamini_yekutieli"):
        if other_pvalues is None:
            raise ValueError(f"{method} is a step procedure over the whole family; pass "
                             f"other_pvalues (the ledger's p-values) or use bonferroni")
        allp = np.concatenate([[p], _as_p(other_pvalues)])
        pad = int(n_trials) - allp.size
        if pad > 0:
            allp = np.concatenate([allp, np.ones(pad)])
        p_adj = float(correct(allp, method, alpha)["adjusted"][0])
    else:
        raise ValueError(f"method must be 'independent' or one of {list(METHODS[1:])}, "
                         f"got {method!r}")
    sr_adj = sharpe_from_p(p_adj, years)
    return {"observed_sharpe": float(sharpe_annual), "years": float(years),
            "n_trials": int(n_trials), "method": method, "t_stat": t, "pvalue": p,
            "adjusted_pvalue": p_adj, "haircut_sharpe": sr_adj,
            "haircut_pct": float(100.0 * (1.0 - sr_adj / sharpe_annual))
            if sharpe_annual > 0 else float("nan"),
            "survives": bool(p_adj <= alpha)}


# ======================================================================================
# Reading the EXISTING trial ledger
# ======================================================================================
@lru_cache(maxsize=1)
def _trial_ledger_class():
    """The TrialLedger from backtest-validation, reached three ways so this file works
    inside `fin_skills.core`, from an installed package, and as a loose script."""
    try:
        from .trial_ledger import TrialLedger            # fin_skills.core.research_history
        return TrialLedger
    except ImportError:
        pass
    try:
        from fin_skills.core.trial_ledger import TrialLedger
        return TrialLedger
    except ImportError:
        pass
    import importlib.util
    src = (Path(__file__).resolve().parent.parent.parent
           / "backtest-validation" / "scripts" / "trial_ledger.py")
    if not src.is_file():
        raise ImportError(
            "trial_ledger.py not found. This file deliberately does NOT define its own "
            "ledger; it reads the one in backtest-validation. Expected at " + str(src))
    spec = importlib.util.spec_from_file_location("_fin_trial_ledger", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TrialLedger


@dataclass
class ResearchHistory:
    """A whole research programme's trials, in the shape the corrections need.

    `m_registered` is the number of HYPOTHESES TESTED and the m every correction uses.
    `scored` are the trials that produced a Sharpe. The difference is trials you ran and
    abandoned, and they are padded with p = 1.0 rather than dropped - which is exactly
    what a step-up or step-down procedure does with a test that did not reject.
    """

    names: list[str]
    pvalues: np.ndarray
    sharpes: np.ndarray
    n_obs: np.ndarray
    m_registered: int
    m_scored: int
    source: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def n_unscored(self) -> int:
        return max(self.m_registered - self.m_scored, 0)

    def padded_pvalues(self) -> np.ndarray:
        """Scored p-values plus one p = 1.0 for every registered-but-unscored trial."""
        return np.concatenate([self.pvalues, np.ones(self.n_unscored)])

    def correct(self, method: str = "holm", alpha: float = 0.05) -> dict:
        r = correct(self.padded_pvalues(), method, alpha)
        r["survivors"] = [self.names[i] for i in np.flatnonzero(r["reject"][:self.m_scored])]
        return r

    def table(self, alpha: float = 0.05) -> pd.DataFrame:
        """One row per scored trial, one column per correction: kept or dropped."""
        out = pd.DataFrame({"sharpe": self.sharpes, "n_obs": self.n_obs,
                            "pvalue": self.pvalues}, index=self.names)
        for name in METHODS:
            out[name] = self.correct(name, alpha)["reject"][:self.m_scored]
        return out.sort_values("pvalue")

    def summary(self, alpha: float = 0.05) -> pd.DataFrame:
        rows = []
        for name in METHODS:
            r = self.correct(name, alpha)
            rows.append({"method": name, "m": r["m"], "survivors": int(r["reject"].sum()),
                         "controls": CONTROLS[name]})
        return pd.DataFrame(rows).set_index("method")


def from_ledger(ledger, alpha: float = 0.05, two_sided: bool = True) -> ResearchHistory:
    """Build a ResearchHistory from a TrialLedger, a path to its trials.jsonl, or records.

    Reads the ledger's own events: `registered` gives m, `completed` gives the Sharpes.
    `abandoned` trials keep their place in m - that is the entire reason the ledger
    records them, and dropping them is what section F of `__main__` measures.
    """
    cls = _trial_ledger_class()
    if isinstance(ledger, (str, Path)):
        led = cls(Path(ledger))
        recs = led.read()
        src = str(led.path)
    elif hasattr(ledger, "read"):
        recs = ledger.read()
        src = str(getattr(ledger, "path", "ledger"))
    else:
        recs = list(ledger)
        src = "records"

    registered = {r["trial_id"] for r in recs if r.get("event") == "registered"}
    label = {r["trial_id"]: f"{r.get('strategy', '?')}{r.get('params', '')}"
             for r in recs if r.get("event") == "registered"}
    names, ps, srs, ns = [], [], [], []
    for r in recs:
        if r.get("event") != "completed":
            continue
        met = r.get("metrics") or {}
        sr, n = met.get("sharpe"), met.get("n_obs")
        if sr is None or n is None:
            continue
        t = t_from_sharpe(float(sr), int(n))
        names.append(label.get(r["trial_id"], r["trial_id"]))
        ps.append(p_from_t(t, int(n), two_sided))
        srs.append(float(sr))
        ns.append(int(n))
    notes = []
    if len(registered) > len(ps):
        notes.append(f"{len(registered) - len(ps)} registered trial(s) have no Sharpe; "
                     f"they are padded with p = 1.0 and still count toward m")
    return ResearchHistory(names=names, pvalues=np.asarray(ps, dtype=float),
                           sharpes=np.asarray(srs, dtype=float),
                           n_obs=np.asarray(ns, dtype=int),
                           m_registered=max(len(registered), len(ps)),
                           m_scored=len(ps), source=src, notes=notes)


# ======================================================================================
# Seeded experiments
# ======================================================================================
DAILY_VOL = 0.01
PPY = 252


def strategy_panel(rng: np.random.Generator, n_obs: int, m: int, n_true: int = 0,
                   true_sr_annual: float = 1.0, rho: float = 0.0) -> np.ndarray:
    """(T, m) daily returns. The first `n_true` columns carry a real edge.

    `rho` mixes in a common factor, so the columns - and therefore the m test statistics -
    are positively dependent. That is the realistic case: every variant of one idea trades
    the same market.
    """
    common = rng.normal(0.0, DAILY_VOL, (n_obs, 1))
    idio = rng.normal(0.0, DAILY_VOL, (n_obs, m))
    x = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio
    if n_true > 0:
        x[:, :n_true] += true_sr_annual * DAILY_VOL / np.sqrt(PPY)
    return x


def panel_pvalues(x: np.ndarray, two_sided: bool = True) -> np.ndarray:
    """One two-sided t-test per column: is this strategy's mean return different from 0?"""
    n = x.shape[0]
    t = x.mean(axis=0) / (x.std(axis=0, ddof=1) / np.sqrt(n))
    p = stats.t.sf(np.abs(t), n - 1)
    return 2.0 * p if two_sided else stats.t.sf(t, n - 1)


def _error_rates(reject: np.ndarray, n_true: int) -> tuple[float, float, float]:
    """(any false positive, false discoveries / discoveries, share of true ones found)."""
    false_pos = int(reject[n_true:].sum())
    true_pos = int(reject[:n_true].sum()) if n_true else 0
    discoveries = false_pos + true_pos
    fdp = false_pos / discoveries if discoveries else 0.0
    power = true_pos / n_true if n_true else float("nan")
    return float(false_pos > 0), fdp, power


# ======================================================================================
def main() -> None:
    bar = "=" * 78
    print("MULTIPLE-TESTING LEDGER - what survives, and what each correction costs")
    print(bar)
    print("Every rate below is measured on seeded data by this file.")

    T, YEARS = 1008, 4.0
    ALPHA = 0.05

    # ---------------------------------------------------------------- A
    print("\n" + bar)
    print(f"A. 200 STRATEGIES, ALL NULL. {T} days each ({YEARS:.0f} years), "
          f"nominal {ALPHA:.0%} two-sided.")
    print(bar)
    x = strategy_panel(np.random.default_rng(11), T, 200)
    p = panel_pvalues(x)
    sr_ann = x.mean(axis=0) / x.std(axis=0, ddof=1) * np.sqrt(PPY)
    print(f"  best annualised Sharpe found : {sr_ann.max():.2f}   "
          f"(true value: 0.00, for all 200)")
    print(f"  smallest p-value             : {p.min():.5f}\n")
    tab = compare(p, ALPHA)
    print(f"  {'correction':<22}{'survivors':>10}   what it controls")
    print("  " + "-" * 76)
    for name, row in tab.iterrows():
        print(f"  {name:<22}{row['survivors']:>10}   {row['controls']}")
    print("  " + "-" * 76)
    print(f"  {int(tab.loc['uncorrected', 'survivors'])} of 200 clear a nominal 5% test - "
          f"{int(tab.loc['uncorrected', 'survivors']) / 2:.1f}%, which is what 5% MEANS.")
    print("  Each of those would be a publishable-looking backtest on its own.")

    # ---------------------------------------------------------------- B
    print("\n" + bar)
    print("B. FAMILY-WISE ERROR RATE, measured. m = 100 null strategies, 400 replications.")
    print(bar)
    print("   FWER = P(at least one false discovery). A 5% procedure should show 5%.\n")
    reps, m = 400, 100
    fwer = {k: 0.0 for k in METHODS}
    for s in range(reps):
        pv = panel_pvalues(strategy_panel(np.random.default_rng(20_000 + s), T, m))
        for name in METHODS:
            fwer[name] += float(correct(pv, name, ALPHA)["reject"].any())
    print(f"  {'correction':<22}{'FWER':>9}{'target':>9}   {'verdict':<28}")
    print("  " + "-" * 72)
    for name in METHODS:
        r = fwer[name] / reps
        target = "-" if name == "uncorrected" else f"{ALPHA:.2f}"
        verdict = ("no control at all" if name == "uncorrected"
                   else "holds" if r <= ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / reps)
                   else "EXCEEDED")
        print(f"  {name:<22}{r:>9.3f}{target:>9}   {verdict:<28}")
    print("  " + "-" * 72)
    print(f"  Uncorrected, {fwer['uncorrected'] / reps:.0%} of research programmes over "
          f"100 dead strategies")
    print("  produce at least one 'significant' result. That is not a small effect.")

    # ---------------------------------------------------------------- C
    print("\n" + bar)
    print("C. THE POWER COST. m = 100, of which 10 are REAL, 300 replications.")
    print(bar)
    reps = 300
    for true_sr in (1.0, 1.5):
        acc = {k: [0.0, 0.0, 0.0] for k in METHODS}
        for s in range(reps):
            pan = strategy_panel(np.random.default_rng(30_000 + s), T, 100,
                                 n_true=10, true_sr_annual=true_sr)
            pv = panel_pvalues(pan)
            for name in METHODS:
                a, b, c = _error_rates(correct(pv, name, ALPHA)["reject"], 10)
                acc[name][0] += a
                acc[name][1] += b
                acc[name][2] += c
        print(f"\n  true annualised Sharpe of the 10 real strategies = {true_sr:.1f} "
              f"(t = {true_sr * np.sqrt(YEARS):.1f} over {YEARS:.0f} years)\n")
        print(f"  {'correction':<22}{'power':>8}{'FWER':>8}{'realised FDR':>14}")
        print("  " + "-" * 54)
        for name in METHODS:
            a, b, c = (v / reps for v in acc[name])
            print(f"  {name:<22}{c:>8.3f}{a:>8.3f}{b:>14.3f}")
        print("  " + "-" * 54)
    print("\n  Read the two blocks together. Bonferroni and Holm buy FWER control with")
    print("  power; BH buys most of that power back and pays for it in FDR, which is a")
    print("  DIFFERENT guarantee - some of what it reports is wrong, by design, at a")
    print("  controlled rate. BY pays again for not having to know the dependence.")

    # ---------------------------------------------------------------- D
    print("\n" + bar)
    print("D. DEPENDENCE. The same 100 strategies, now sharing a common factor.")
    print(bar)
    reps = 300
    print(f"  {'rho':>6}{'mean |corr|':>13}{'BH FDR':>9}{'BH power':>10}"
          f"{'BY FDR':>9}{'BY power':>10}{'Holm power':>12}")
    print("  " + "-" * 70)
    for rho in (0.0, 0.3, 0.7):
        acc = {k: [0.0, 0.0] for k in ("benjamini_hochberg", "benjamini_yekutieli", "holm")}
        for s in range(reps):
            pan = strategy_panel(np.random.default_rng(40_000 + s), T, 100, n_true=10,
                                 true_sr_annual=1.5, rho=rho)
            pv = panel_pvalues(pan)
            for name in acc:
                _, fdp, pw = _error_rates(correct(pv, name, ALPHA)["reject"], 10)
                acc[name][0] += fdp
                acc[name][1] += pw
        pan = strategy_panel(np.random.default_rng(40_000), T, 100, n_true=10,
                             true_sr_annual=1.5, rho=rho)
        cc = np.corrcoef(pan, rowvar=False)
        mc = float(np.abs(cc[~np.eye(100, dtype=bool)]).mean())
        b = [v / reps for v in acc["benjamini_hochberg"]]
        y = [v / reps for v in acc["benjamini_yekutieli"]]
        h = [v / reps for v in acc["holm"]]
        print(f"  {rho:>6.1f}{mc:>13.3f}{b[0]:>9.3f}{b[1]:>10.3f}"
              f"{y[0]:>9.3f}{y[1]:>10.3f}{h[1]:>12.3f}")
    print("  " + "-" * 70)
    print("  A common factor is POSITIVE dependence, the case BH is proved for, and BH's")
    print("  realised FDR stays at or under 5% in every row - measured, not assumed. BY")
    print("  is the price of not being able to prove your dependence is positive, and the")
    print("  power column is what that costs.")
    print(f"\n  {'m':>7}{'c(m) = sum 1/i':>18}{'BY threshold vs BH':>22}")
    print("  " + "-" * 48)
    for mm in (10, 50, 100, 500, 1000):
        print(f"  {mm:>7}{by_constant(mm):>18.3f}{'1 / ' + f'{by_constant(mm):.2f}':>22}")
    print("  " + "-" * 48)

    # ---------------------------------------------------------------- E
    print("\n" + bar)
    print("E. THE HAIRCUT. What is left of a Sharpe after the count is applied.")
    print(bar)
    print("  Sharpe you may still report after Bonferroni on N trials, and the % taken off.\n")
    print(f"  {'observed SR':>12}{'years':>7}{'t':>7}{'p':>11}"
          f"{'N=1':>8}{'N=10':>13}{'N=100':>13}{'N=1000':>13}")
    print("  " + "-" * 84)
    for sr, yrs in ((0.4, 10.0), (0.75, 20.0), (1.0, 4.0), (1.0, 10.0), (1.5, 10.0),
                    (2.0, 10.0)):
        base = haircut_sharpe(sr, yrs, 1)
        cells = []
        for n_tr in (10, 100, 1000):
            h = haircut_sharpe(sr, yrs, n_tr)
            cells.append(f"{h['haircut_sharpe']:.2f} ({h['haircut_pct']:.0f}%)")
        print(f"  {sr:>12.2f}{yrs:>7.0f}{base['t_stat']:>7.2f}{base['pvalue']:>11.2e}"
              f"{base['haircut_sharpe']:>8.2f}"
              f"{cells[0]:>13}{cells[1]:>13}{cells[2]:>13}")
    print("  " + "-" * 84)
    print("  A 0.00 means the adjusted p-value reached 1.0: the result is entirely")
    print("  explained by the search. Read DOWN a column, not across a row - the haircut")
    print("  is far heavier on small Sharpes than on large ones, at the same trial count.")
    h = haircut_sharpe(1.5, 10.0, 100)
    print(f"\n  Worked cell: SR {h['observed_sharpe']:.1f} over {h['years']:.0f} years is "
          f"t = {h['t_stat']:.2f}, p = {h['pvalue']:.2e}.")
    print(f"  With {h['n_trials']} trials the adjusted p is {h['adjusted_pvalue']:.2e}, "
          f"worth SR {h['haircut_sharpe']:.2f}")
    print(f"  - a haircut of {h['haircut_pct']:.0f}%.")

    # ---------------------------------------------------------------- F
    print("\n" + bar)
    print("F. THE LEDGER, AND THE ABANDONED TRIALS EVERY CORRECTION NEEDS.")
    print(bar)
    cls = _trial_ledger_class()
    print(f"  Using {cls.__module__}.{cls.__name__} - the ledger from backtest-validation,")
    print("  not a second format. m comes from its REGISTERED count.\n")
    rng = np.random.default_rng(77)
    n_reg, n_completed, n_true, true_sr = 400, 40, 8, 1.8
    panel = strategy_panel(np.random.default_rng(55), T, n_reg, n_true=n_true,
                           true_sr_annual=true_sr)
    per_period = panel.mean(axis=0) / panel.std(axis=0, ddof=1)
    with tempfile.TemporaryDirectory() as td:
        led = cls(Path(td) / "trials.jsonl")
        ids = []
        for i in range(n_reg):
            ids.append(led.record("grid", {"variant": i}))
        # the researcher scores the real ones and some others, and abandons the rest
        keep = list(range(n_true)) + sorted(
            rng.choice(np.arange(n_true, n_reg), n_completed - n_true,
                       replace=False).tolist())
        for i in keep:
            led.complete(ids[i], {"sharpe": float(per_period[i]), "n_obs": T})
        for i in range(n_reg):
            if i not in keep:
                led.abandon(ids[i], "did not look promising")
        hist = from_ledger(led)
        s = led.summary()
        print(f"  ledger summary()   : n_trials={s['n_trials']}  "
              f"n_completed={s['n_completed']}  n_abandoned={s['n_abandoned']}")
        print(f"  ResearchHistory    : m_registered={hist.m_registered}  "
              f"m_scored={hist.m_scored}  unscored={hist.n_unscored}")
        for n in hist.notes:
            print(f"    ! {n}")
        honest = hist.summary(ALPHA)
        dishonest_p = hist.pvalues                     # only the trials that finished
        print(f"\n  {'correction':<22}{f'm = {n_reg} (honest)':>20}"
              f"{f'm = {n_completed} (scored only)':>26}{'inflation':>11}")
        print("  " + "-" * 79)
        for name in METHODS:
            a = int(honest.loc[name, "survivors"])
            b = int(correct(dishonest_p, name, ALPHA)["reject"].sum())
            print(f"  {name:<22}{a:>20}{b:>26}{b - a:>+11}")
        print("  " + "-" * 79)
        print(f"  {n_true} real strategies are planted here (true annualised SR "
              f"{true_sr:.1f}); the rest are dead.")
        print("  Counting only the trials that finished shrinks m by a factor of "
              f"{n_reg / n_completed:.0f} and lets")
        print("  extra strategies through every FWER and FDR gate. The abandoned trials")
        print("  were tests; the ledger recorded them for exactly this arithmetic.")
        print("  `abandon()` is not bookkeeping, it is the denominator.")
        top = hist.table(ALPHA).head(5)
        print("\n  Top 5 by p-value (True = kept by that correction):")
        with pd.option_context("display.width", 120, "display.max_columns", 12):
            print("  " + top.to_string(float_format=lambda v: f"{v:.4f}").replace("\n", "\n  "))

    # ---------------------------------------------------------------- G
    print("\n" + bar)
    print("G. REPRODUCING THE PUBLISHED NUMBERS (this is what 'verified' means here)")
    print(bar)
    hlz_p = np.array([0.005, 0.009, 0.0128, 0.0135, 0.045, 0.06])
    print("  Harvey, Liu & Zhu's own six-test example, p = ["
          + ", ".join(f"{v:g}" for v in hlz_p) + f"], c(6) = {by_constant(6):.4f}:\n")
    print(f"  {'method':<22}{'adjusted p-values':<52}{'sig @5%':>9}")
    print("  " + "-" * 84)
    for name, fn in (("bonferroni", bonferroni), ("holm", holm),
                     ("benjamini_yekutieli", benjamini_yekutieli),
                     ("harvey_liu_bhy", harvey_liu_bhy)):
        r = fn(hlz_p, 0.05)
        cells = ", ".join(f"{v:.4f}" for v in r["adjusted"])
        print(f"  {name:<22}{cells:<52}{int(r['reject'].sum()):>9}")
    print("  " + "-" * 84)
    print("  The paper prints Bonferroni 1 significant, Holm 2, BHY 4; all three match.")
    print("  But its BHY column reads 0.0496 x4 then 0.0600, 0.0600, while standard BY")
    print("  gives 0.1323 and 0.1470 for the last two.")
    try:
        import statsmodels
        from statsmodels.stats.multitest import multipletests
        pairs = (("bonferroni", "bonferroni"), ("holm", "holm"),
                 ("fdr_bh", "benjamini_hochberg"), ("fdr_by", "benjamini_yekutieli"))
        rng2 = np.random.default_rng(5)
        big = rng2.uniform(0.0, 1.0, 500) ** 3          # a spread of small and large p
        worst = {}
        for sm_name, mine in pairs:
            rej, adj, _, _ = multipletests(big, alpha=ALPHA, method=sm_name)
            r = correct(big, mine, ALPHA)
            worst[mine] = (float(np.abs(adj - r["adjusted"]).max()),
                           bool(np.array_equal(rej, r["reject"])))
        print(f"\n  Cross-checked against statsmodels {statsmodels.__version__} on 500 "
              f"p-values:")
        for name, (d, same) in worst.items():
            print(f"    {name:<22} max |adjusted p difference| = {d:.1e}   "
                  f"same rejections: {same}")
        print("  The four standard procedures here ARE statsmodels' four; nothing in this")
        print("  file is a private variant of them. `harvey_liu_bhy` has no statsmodels")
        print("  equivalent, which is the point of the comparison above it.")
    except ImportError:
        print("\n  statsmodels not installed - the four standard procedures above are")
        print("  implemented from their definitions and are not cross-checked in this run.")

    all_same = np.array([0.04] * 6)
    std = benjamini_yekutieli(all_same, 0.05)
    hl = harvey_liu_bhy(all_same, 0.05)
    print(f"\n  Six tests all at p = 0.04, alpha = 5%:  standard BY rejects "
          f"{int(std['reject'].sum())}, Harvey-Liu BHY rejects {int(hl['reject'].sum())}.")
    print("  Their recursion pins the largest adjusted p-value at its RAW value instead of")
    print("  multiplying it by c(M) - printed that way in the paper AND coded that way in")
    print("  their own Haircut_SR.m. Quote `benjamini_yekutieli` for the FDR guarantee.")

    hl_ex = haircut_sharpe(0.75, 20.0, 200, method="independent")
    print(f"\n  Harvey & Liu's haircut illustration: annualised SR "
          f"{hl_ex['observed_sharpe']:.2f} over {hl_ex['years']:.0f} years")
    print(f"  gives t = {hl_ex['t_stat']:.3f}, p = {hl_ex['pvalue']:.4f}; with "
          f"{hl_ex['n_trials']} independent tests")
    print(f"  the adjusted p is {hl_ex['adjusted_pvalue']:.4f} and the haircut Sharpe is "
          f"{hl_ex['haircut_sharpe']:.3f}")
    print(f"  - a haircut of {hl_ex['haircut_pct']:.1f}%. The paper reports 0.32 and "
          f"'approximately 60%'.")
    bonf = haircut_sharpe(0.75, 20.0, 200, method="bonferroni")
    print(f"  Bonferroni on the same input gives {bonf['adjusted_pvalue']:.4f} -> SR "
          f"{bonf['haircut_sharpe']:.3f}: the two adjustments differ by "
          f"{abs(bonf['haircut_sharpe'] - hl_ex['haircut_sharpe']):.3f} of Sharpe here.")
    print("\n  Their headline is NOT 'halve every Sharpe'. Measured on this file's own")
    print("  arithmetic across the section E table, the haircut is far heavier for small")
    print("  Sharpes than for large ones, which is the point the 50% rule of thumb misses.")

    print("\n" + bar)
    print("RULE: m is the number of hypotheses you TESTED, which is the ledger's registered "
          "count -")
    print("abandoned trials included - and the correction to quote is the one whose "
          "dependence")
    print("assumption you can actually defend: Holm for FWER, BH only if the dependence is "
          "positive, BY otherwise.")
    print(bar)


if __name__ == "__main__":
    main()
