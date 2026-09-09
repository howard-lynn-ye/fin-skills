"""Ratings transition matrices: cohort vs duration estimators, withdrawn ratings, matrix powers,
and the embedding problem.

WHY this exists: a published one-year transition matrix looks like a probability object you can
raise to a power, take a root of, or differentiate. Three things go wrong and none of them raises:

  1. A STRUCTURAL ZERO KILLS EMBEDDABILITY. The cohort estimator reports AAA -> D as exactly
     0.0000% because no AAA issuer defaulted inside one year, while P^2 puts a positive number
     there. No generator can produce that pattern, so `scipy.linalg.logm(P)` returns a matrix
     with NEGATIVE off-diagonal entries and the implied six-month matrix `expm(logm(P)/2)`
     contains a NEGATIVE PROBABILITY. It is a float, it prints, and nothing complains.

  2. THE DURATION ESTIMATOR DOES NOT HAVE THE PROBLEM. Estimating a generator directly from
     transition counts and time at risk (the Aalen-Johansen / Nelson-Aalen route) puts a small
     POSITIVE number in the AAA -> D cell, because the path AAA -> AA -> ... -> D was observed
     even though the single-step jump was not. Same data, embeddable answer.

  3. WITHDRAWN RATINGS ARE CENSORING, NOT AN OUTCOME. Treating NR as an absorbing state leaks
     probability mass that should have flowed on to default, and understates every long-horizon
     default probability.

Plus the arithmetic one: `5 x PD_1` is not the five-year PD, and it errs in BOTH directions --
low for investment grade (where default builds up through migration) and wildly high for CCC
(where the one-year PD is too large to multiply).

numpy + scipy only, seeded, no network, no files.

Usage:
    from migration import published_matrix, embedding_report, horizon_trap
    embedding_report()["min_root_entry"]     # a negative probability
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import expm, logm

SEED = 20260909
GRADES = ("AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D")
DEFAULT = len(GRADES) - 1

# A synthetic but realistically shaped annual matrix, given as OFF-DIAGONALS in per cent; each
# diagonal is set to 1 - sum(row) so the rows sum to 1 exactly. AAA -> D is a STRUCTURAL ZERO:
# that single cell is what the whole embedding section is about.
_OFF_DIAGONAL_PCT: dict[str, dict[str, float]] = {
    "AAA": {"AA": 9.00, "A": 0.50, "BBB": 0.10, "BB": 0.05, "B": 0.02, "CCC": 0.01, "D": 0.00},
    "AA":  {"AAA": 0.55, "A": 8.00, "BBB": 0.60, "BB": 0.15, "B": 0.08, "CCC": 0.03, "D": 0.02},
    "A":   {"AAA": 0.04, "AA": 1.90, "BBB": 5.60, "BB": 0.45, "B": 0.18, "CCC": 0.05, "D": 0.06},
    "BBB": {"AAA": 0.01, "AA": 0.15, "A": 3.80, "BB": 4.30, "B": 0.85, "CCC": 0.20, "D": 0.24},
    "BB":  {"AAA": 0.01, "AA": 0.05, "A": 0.25, "BBB": 5.60, "B": 7.50, "CCC": 0.80, "D": 0.90},
    "B":   {"AA": 0.04, "A": 0.15, "BBB": 0.35, "BB": 5.20, "CCC": 4.80, "D": 4.20},
    "CCC": {"A": 0.20, "BBB": 0.40, "BB": 1.20, "B": 11.00, "D": 19.80},
}


def published_matrix() -> np.ndarray:
    """The one-year matrix in the shape a rating agency publishes one: cohort estimates, an
    absorbing default state, exact row sums, and a structural zero at AAA -> D."""
    n = len(GRADES)
    p = np.zeros((n, n))
    idx = {g: i for i, g in enumerate(GRADES)}
    for row, cells in _OFF_DIAGONAL_PCT.items():
        i = idx[row]
        for col, v in cells.items():
            p[i, idx[col]] = v / 100.0
        p[i, i] = 1.0 - p[i].sum()
    p[DEFAULT, DEFAULT] = 1.0
    return p


def check_stochastic(p: np.ndarray, tol: float = 1e-12) -> dict[str, float]:
    """Everything a matrix must satisfy to be called a transition matrix."""
    return {"max_row_sum_error": float(np.max(np.abs(p.sum(axis=1) - 1.0))),
            "min_entry": float(p.min()),
            "is_stochastic": float(np.max(np.abs(p.sum(axis=1) - 1.0)) < tol and p.min() >= 0.0)}


# --------------------------------------------------------------------------- matrix powers
def matrix_power(p: np.ndarray, n: int) -> np.ndarray:
    if n < 0:
        raise ValueError("a negative matrix power is not a transition matrix")
    return np.linalg.matrix_power(p, n)


def cumulative_pd(p: np.ndarray, horizons: Sequence[int] = (1, 2, 3, 5, 10)) -> np.ndarray:
    """Cumulative default probability by grade and horizon, from P^n."""
    return np.array([[matrix_power(p, h)[i, DEFAULT] for h in horizons]
                     for i in range(len(GRADES) - 1)])


def horizon_trap(p: np.ndarray | None = None, horizon: int = 5) -> list[dict[str, float]]:
    """`horizon x PD_1` against the truth from P^horizon, by grade. It errs BOTH ways."""
    p = published_matrix() if p is None else p
    ph = matrix_power(p, horizon)
    out = []
    for i, g in enumerate(GRADES[:-1]):
        true_pd = float(ph[i, DEFAULT])
        naive = float(p[i, DEFAULT]) * horizon
        out.append({"grade": g, "pd1_pct": p[i, DEFAULT] * 100.0, "true_pct": true_pd * 100.0,
                    "naive_pct": naive * 100.0, "error_bp": (naive - true_pd) * 1e4})
    return out


# --------------------------------------------------------------------------- the embedding trap
@dataclass(frozen=True)
class EmbeddingReport:
    aaa_default_1y: float
    aaa_default_2y: float
    min_generator_offdiag: float
    generator_offdiag_cell: tuple[str, str]
    n_negative_generator_cells: int
    min_root_entry: float
    root_cell: tuple[str, str]
    n_negative_root_cells: int
    root_squared_error: float
    structural_zeros: tuple[tuple[str, str], ...]
    negative_cells: tuple[tuple[str, str], ...]
    zeros_explain_negatives: bool
    aaa_default_generator: float
    aaa_default_root: float


def structural_zeros(p: np.ndarray) -> list[tuple[str, str]]:
    """Off-diagonal cells a cohort estimator reported as exactly zero -- nobody was seen doing it."""
    return [(GRADES[i], GRADES[j]) for i in range(len(GRADES)) for j in range(len(GRADES))
            if i != j and p[i, j] == 0.0 and i != DEFAULT]


def generator_log(p: np.ndarray) -> np.ndarray:
    """The principal matrix logarithm. Real by construction here; the imaginary part is checked."""
    q = logm(p)
    if np.max(np.abs(np.imag(q))) > 1e-10:
        raise ValueError("the principal logarithm is complex - this matrix has no real log")
    return np.real(q)


def is_valid_generator(q: np.ndarray, tol: float = 1e-12) -> bool:
    """A generator has non-negative off-diagonals and rows summing to zero."""
    off = q - np.diag(np.diag(q))
    return bool(off.min() >= -tol and np.max(np.abs(q.sum(axis=1))) < 1e-8)


def embedding_report(p: np.ndarray | None = None) -> EmbeddingReport:
    """The structural zero, the negative generator entry it forces, and the negative probability
    that comes out of the implied six-month matrix."""
    p = published_matrix() if p is None else p
    q = generator_log(p)
    off = q.copy()
    np.fill_diagonal(off, np.inf)
    i, j = np.unravel_index(np.argmin(off), off.shape)
    root = expm(q / 2.0)
    ri, rj = np.unravel_index(np.argmin(root), root.shape)
    zeros = structural_zeros(p)
    negs = [(GRADES[a], GRADES[b]) for a, b in np.argwhere(off < 0)]
    return EmbeddingReport(
        aaa_default_1y=float(p[0, DEFAULT]),
        aaa_default_2y=float(matrix_power(p, 2)[0, DEFAULT]),
        min_generator_offdiag=float(off[i, j]),
        generator_offdiag_cell=(GRADES[i], GRADES[j]),
        n_negative_generator_cells=int((off < 0).sum()),
        min_root_entry=float(root[ri, rj]),
        root_cell=(GRADES[ri], GRADES[rj]),
        n_negative_root_cells=int((root < 0).sum()),
        root_squared_error=float(np.max(np.abs(root @ root - p))),
        structural_zeros=tuple(zeros),
        negative_cells=tuple(negs),
        zeros_explain_negatives=set(zeros) == set(negs),
        aaa_default_generator=float(q[0, DEFAULT]),
        aaa_default_root=float(root[0, DEFAULT]))


def fill_structural_zeros(p: np.ndarray, fill: float) -> np.ndarray:
    """Put `fill` in every off-diagonal zero and renormalise the rows. Default stays absorbing."""
    n = len(GRADES)
    q = np.where((p == 0.0) & ~np.eye(n, dtype=bool), fill, p.copy())
    q[DEFAULT] = 0.0
    q[DEFAULT, DEFAULT] = 1.0
    q[:DEFAULT] /= q[:DEFAULT].sum(axis=1, keepdims=True)
    return q


def fill_scan(fills: Sequence[float] = (1e-6, 1e-5, 3e-5, 5e-5, 1e-4, 5e-4),
              p: np.ndarray | None = None) -> list[dict[str, float]]:
    """How big does a structural zero have to be before the matrix becomes embeddable?

    "Not exactly zero" is not enough: the cell has to be consistent with the two-step paths
    that reach it, so a token epsilon leaves the generator negative.
    """
    p = published_matrix() if p is None else p
    out = []
    for f in fills:
        filled = fill_structural_zeros(p, f)
        q = generator_log(filled)
        off = q.copy()
        np.fill_diagonal(off, np.inf)
        out.append({"fill": f, "min_generator_offdiag": float(off.min()),
                    "valid_generator": float(is_valid_generator(q)),
                    "min_root_entry": float(expm(q / 2.0).min())})
    return out


def regularised_generator(p: np.ndarray | None = None) -> dict[str, object]:
    """The standard repair: clip the negative off-diagonals of logm(P) to zero and put the
    difference back on the diagonal so the rows still sum to zero (Israel-Rosenthal-Wei).

    It produces a VALID generator, and the price is that expm(Q) no longer reproduces P.
    """
    p = published_matrix() if p is None else p
    q = generator_log(p)
    off = np.clip(q - np.diag(np.diag(q)), 0.0, None)
    reg = off.copy()
    np.fill_diagonal(reg, -off.sum(axis=1))
    root = expm(reg / 2.0)
    return {"generator": reg, "valid": is_valid_generator(reg),
            "root": root, "min_root_entry": float(root.min()),
            "reproduction_error": float(np.max(np.abs(expm(reg) - p))),
            "reproduction_error_bp": float(np.max(np.abs(expm(reg) - p)) * 1e4)}


# --------------------------------------------------------------------------- the simulation
def true_generator() -> np.ndarray:
    """A VALID generator with a structural zero in the AAA -> D cell, used to simulate from.

    Because it is a real generator, `expm(Q)` puts a small POSITIVE number in AAA -> D: the
    two-step paths exist even though the one-step intensity is zero. That is the fact the
    cohort estimator cannot see and the duration estimator can.
    """
    p = published_matrix()
    q = generator_log(p)
    off = np.clip(q - np.diag(np.diag(q)), 0.0, None)
    off[0, DEFAULT] = 0.0                      # the structural zero, by construction
    reg = off.copy()
    np.fill_diagonal(reg, -off.sum(axis=1))
    reg[DEFAULT, :] = 0.0
    return reg


@dataclass(frozen=True)
class Panel:
    yearly: np.ndarray          # (n_firms, years + 1) rating each year, -1 once unobserved
    counts: np.ndarray          # observed jump counts N[i, j]
    exposure: np.ndarray        # time at risk Y[i], in years
    n_withdrawn: int


def simulate_panel(q: np.ndarray | None = None, n_firms: int = 4000, years: int = 20,
                   seed: int = SEED, withdrawal_intensity: Sequence[float] | None = None
                   ) -> Panel:
    """Simulate continuous-time rating histories, then record BOTH what a cohort study sees
    (the rating each 31 December) and what a duration study sees (every jump, and the time at
    risk that produced it).

    `withdrawal_intensity` is a per-grade hazard of the rating being withdrawn -- censoring,
    not an outcome. Higher for low grades, which is what makes the naive treatments biased.
    """
    q = true_generator() if q is None else q
    n = len(GRADES)
    rng = np.random.default_rng(seed)
    rates = -np.diag(q)
    jump_p = np.zeros((n, n))
    for i in range(n):
        if rates[i] > 0:
            row = q[i].copy()
            row[i] = 0.0
            jump_p[i] = row / rates[i]
    wd = np.zeros(n) if withdrawal_intensity is None else np.asarray(withdrawal_intensity,
                                                                     dtype=float)
    start = rng.choice(n - 1, size=n_firms, p=np.array([0.05, 0.12, 0.28, 0.30, 0.15, 0.08, 0.02]))
    yearly = np.full((n_firms, years + 1), -1, dtype=int)
    counts = np.zeros((n, n))
    exposure = np.zeros(n)
    n_withdrawn = 0
    for f in range(n_firms):
        state = int(start[f])
        t = 0.0
        t_wd = (rng.exponential(1.0 / wd[state]) if wd[state] > 0 else math.inf)
        yearly[f, 0] = state
        while t < years and state != DEFAULT:
            if t >= t_wd:
                n_withdrawn += 1
                break
            hold = rng.exponential(1.0 / rates[state]) if rates[state] > 0 else math.inf
            end = min(t + hold, years, t_wd)
            exposure[state] += end - t
            for k in range(int(math.ceil(t)), min(int(math.floor(end)), years) + 1):
                if k > 0:
                    yearly[f, k] = state
            if end >= years or end == t_wd:
                if end == t_wd and t_wd < years:
                    n_withdrawn += 1
                t = end
                break
            nxt = int(rng.choice(n, p=jump_p[state]))
            counts[state, nxt] += 1
            state, t = nxt, end
            t_wd = t + (rng.exponential(1.0 / wd[state]) if wd[state] > 0 else math.inf)
        if state == DEFAULT:
            for k in range(int(math.ceil(t)), years + 1):
                yearly[f, k] = DEFAULT
    return Panel(yearly=yearly, counts=counts, exposure=exposure, n_withdrawn=n_withdrawn)


def cohort_estimator(yearly: np.ndarray) -> np.ndarray:
    """The multinomial cohort estimator: N_ij / N_i over year-start / year-end pairs.

    Firm-years where the rating is not observed at BOTH ends are dropped -- that is the correct
    handling of a withdrawal, and it is not what "NR as a state" does.
    """
    n = len(GRADES)
    counts = np.zeros((n, n))
    a, b = yearly[:, :-1], yearly[:, 1:]
    ok = (a >= 0) & (b >= 0)
    np.add.at(counts, (a[ok], b[ok]), 1.0)
    tot = counts.sum(axis=1, keepdims=True)
    p = np.divide(counts, tot, out=np.zeros_like(counts), where=tot > 0)
    for i in range(n):
        if tot[i] == 0:
            p[i, i] = 1.0
    p[DEFAULT] = 0.0
    p[DEFAULT, DEFAULT] = 1.0
    return p


def duration_estimator(counts: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """The Aalen-Johansen / Nelson-Aalen generator: Q_ij = N_ij / Y_i, diagonal = -row sum."""
    n = counts.shape[0]
    q = np.zeros_like(counts)
    for i in range(n):
        if exposure[i] > 0:
            q[i] = counts[i] / exposure[i]
        q[i, i] = 0.0
        q[i, i] = -q[i].sum()
    q[DEFAULT, :] = 0.0
    return q


def estimator_comparison(n_firms: int = 4000, years: int = 20, seed: int = SEED
                         ) -> dict[str, object]:
    """Same simulated data, two estimators. Only one of them can see AAA -> D."""
    q_true = true_generator()
    panel = simulate_panel(q_true, n_firms, years, seed)
    p_true = expm(q_true)
    p_cohort = cohort_estimator(panel.yearly)
    q_dur = duration_estimator(panel.counts, panel.exposure)
    p_dur = expm(q_dur)
    rows = []
    for i, g in enumerate(GRADES[:-1]):
        rows.append({"grade": g, "true_pct": p_true[i, DEFAULT] * 100.0,
                     "cohort_pct": p_cohort[i, DEFAULT] * 100.0,
                     "duration_pct": p_dur[i, DEFAULT] * 100.0})
    return {"p_true": p_true, "p_cohort": p_cohort, "p_duration": p_dur, "q_duration": q_dur,
            "rows": rows,
            "cohort_aaa_default": float(p_cohort[0, DEFAULT]),
            "duration_aaa_default": float(p_dur[0, DEFAULT]),
            "true_aaa_default": float(p_true[0, DEFAULT]),
            "cohort_zero_cells": int((p_cohort == 0.0).sum() - (p_true == 0.0).sum()),
            "duration_valid_generator": is_valid_generator(q_dur),
            "duration_root_min": float(expm(q_dur / 2.0).min()),
            "cohort_root_min": float(expm(generator_log(p_cohort) / 2.0).min())
            if np.all(np.linalg.eigvals(p_cohort).real > 0) else float("nan")}


# --------------------------------------------------------------------------- withdrawals
def withdrawal_bias(n_firms: int = 4000, years: int = 20, seed: int = SEED,
                    horizon: int = 5) -> dict[str, object]:
    """Three ways to handle a withdrawn rating, and what each does to the 5-year PD.

    NR-as-a-state is the one that looks harmless: it keeps the rows summing to one, so every
    validity check passes while probability that should have reached default sits in NR.
    """
    q_true = true_generator()
    intensity = [0.02, 0.03, 0.04, 0.05, 0.08, 0.12, 0.20, 0.0]
    panel = simulate_panel(q_true, n_firms, years, seed, withdrawal_intensity=intensity)
    p_true = expm(q_true)
    censored = cohort_estimator(panel.yearly)                     # correct: drop the firm-year
    # NR as an extra absorbing state: a withdrawal is recorded as an outcome, not as censoring
    n = len(GRADES)
    counts = np.zeros((n + 1, n + 1))
    a, b = panel.yearly[:, :-1], panel.yearly[:, 1:]
    seen = a >= 0
    tgt = np.where(b >= 0, b, n)                                  # n == the NR state
    np.add.at(counts, (a[seen], tgt[seen]), 1.0)
    tot = counts.sum(axis=1, keepdims=True)
    nr = np.divide(counts, tot, out=np.zeros_like(counts), where=tot > 0)
    for i in range(n + 1):
        if tot[i] == 0:
            nr[i, i] = 1.0
    nr[DEFAULT] = 0.0
    nr[DEFAULT, DEFAULT] = 1.0
    nr[n] = 0.0
    nr[n, n] = 1.0
    rows = []
    for i, g in enumerate(GRADES[:-1]):
        t = float(matrix_power(p_true, horizon)[i, DEFAULT])
        c = float(matrix_power(censored, horizon)[i, DEFAULT])
        w = float(matrix_power(nr, horizon)[i, DEFAULT])
        rows.append({"grade": g, "true_pct": t * 100.0, "censored_pct": c * 100.0,
                     "nr_state_pct": w * 100.0, "nr_error_bp": (w - t) * 1e4,
                     "censored_error_bp": (c - t) * 1e4})
    return {"rows": rows, "n_withdrawn": panel.n_withdrawn,
            "nr_mass_5y": float(matrix_power(nr, horizon)[3, n]),
            "worst_nr_error_bp": max(abs(r["nr_error_bp"]) for r in rows),
            "worst_censored_error_bp": max(abs(r["censored_error_bp"]) for r in rows)}


# --------------------------------------------------------------------------- demo
if __name__ == "__main__":
    W = 96
    print("=" * W)
    print("RATINGS TRANSITIONS -- cohort vs duration, withdrawn ratings, and the embedding trap")
    print("=" * W)

    p = published_matrix()
    chk = check_stochastic(p)
    print("\n1. THE ONE-YEAR MATRIX (synthetic, shaped like a published one; per cent)")
    print(f"   {'from':>6}" + "".join(f"{g:>9}" for g in GRADES))
    for i, g in enumerate(GRADES):
        print(f"   {g:>6}" + "".join(f"{p[i, j] * 100:>9.4f}" for j in range(len(GRADES))))
    print(f"   row sums exact to {chk['max_row_sum_error']:.1e}, min entry "
          f"{chk['min_entry']:.4f}, stochastic {bool(chk['is_stochastic'])}")

    e = embedding_report()
    print("\n2. THE STRUCTURAL ZERO, AND WHY IT HAS NO GENERATOR")
    print(f"   AAA -> D at ONE year  {e.aaa_default_1y * 100:.4f}%   "
          f"(a cohort estimate: no AAA defaulted inside a year)")
    print(f"   AAA -> D at TWO years {e.aaa_default_2y * 100:.4f}%   (P^2 is positive)")
    print(f"   -> logm(P) has {e.n_negative_generator_cells} NEGATIVE off-diagonal cell(s); "
          f"the worst is")
    print(f"      {e.generator_offdiag_cell[0]} -> {e.generator_offdiag_cell[1]} at "
          f"{e.min_generator_offdiag:.3e}. A generator cannot have one.")
    print(f"   -> the implied SIX-MONTH matrix expm(logm(P)/2) has "
          f"{e.n_negative_root_cells} negative entries;")
    print(f"      the worst is {e.root_cell[0]} -> {e.root_cell[1]} = {e.min_root_entry:.3e} "
          f"-- A NEGATIVE PROBABILITY.")
    print(f"   -> and it still squares back to P to {e.root_squared_error:.1e}, so every check "
          f"you would run passes.")
    print(f"   The correspondence is EXACT: the negative cells are precisely the structural "
          f"zeros of P")
    print(f"   ({e.zeros_explain_negatives}) -- " +
          ", ".join(f"{a}->{b}" for a, b in e.structural_zeros))
    print(f"   AAA -> D itself: generator {e.aaa_default_generator:.4e}, six-month "
          f"{e.aaa_default_root:.4e}")

    print("\n3. 'NOT EXACTLY ZERO' IS NOT ENOUGH -- how big the cell has to be")
    print(f"   {'fill in every zero':>20}{'min generator off-diag':>26}{'valid':>8}"
          f"{'min six-month entry':>22}")
    for row in fill_scan():
        print(f"   {row['fill']:>20.1e}{row['min_generator_offdiag']:>26.3e}"
              f"{bool(row['valid_generator']):>8}{row['min_root_entry']:>22.3e}")
    print("   A token epsilon leaves the generator negative: the cell has to be consistent with")
    print("   the two-step paths that reach it. Note the six-month matrix turns non-negative")
    print("   BEFORE the generator does -- passing one test is not passing the other.")

    reg = regularised_generator()
    print("\n4. THE REPAIR, AND WHAT IT COSTS")
    print(f"   clip the negative off-diagonals to zero, put the difference on the diagonal:")
    print(f"   valid generator {reg['valid']}   min six-month entry "
          f"{reg['min_root_entry']:.3e} (non-negative)")
    print(f"   price: expm(Q_reg) no longer reproduces P -- worst cell off by "
          f"{reg['reproduction_error_bp']:.2f} bp")
    print("   You cannot have both. Decide which property you need before you take the root.")

    print("\n5. FIVE TIMES THE ONE-YEAR PD IS WRONG IN BOTH DIRECTIONS")
    print(f"   {'grade':>7}{'1y PD':>10}{'true 5y (P^5)':>16}{'5 x PD1':>11}{'error bp':>12}")
    for row in horizon_trap():
        print(f"   {row['grade']:>7}{row['pd1_pct']:>9.4f}%{row['true_pct']:>15.4f}%"
              f"{row['naive_pct']:>10.4f}%{row['error_bp']:>+12.1f}")
    print("   Investment grade defaults build up through MIGRATION, so 5 x PD1 is too low.")
    print("   CCC's one-year PD is too big to multiply, so 5 x PD1 runs off the end of [0, 1].")

    print("\n6. COHORT vs DURATION ON THE SAME SIMULATED DATA")
    cmp_ = estimator_comparison()
    print(f"   4,000 firms, 20 years, simulated from a VALID generator whose AAA -> D intensity")
    print(f"   is exactly zero. The truth still has a positive one-year AAA -> D: "
          f"{cmp_['true_aaa_default'] * 100:.6f}%")
    print(f"   {'grade':>7}{'true 1y PD':>14}{'cohort':>12}{'duration':>12}")
    for row in cmp_["rows"]:
        print(f"   {row['grade']:>7}{row['true_pct']:>13.6f}%{row['cohort_pct']:>11.6f}%"
              f"{row['duration_pct']:>11.6f}%")
    print(f"   cohort AAA -> D  = {cmp_['cohort_aaa_default'] * 100:.6f}%  <- EXACTLY zero, and "
          f"that is the trap")
    print(f"   duration AAA -> D = {cmp_['duration_aaa_default'] * 100:.6f}%  <- positive, "
          f"because AAA -> AA -> ... -> D was observed")
    print(f"   duration estimate is a valid generator: {cmp_['duration_valid_generator']}; "
          f"its six-month root's min entry")
    print(f"   is {cmp_['duration_root_min']:.3e} vs the cohort matrix's "
          f"{cmp_['cohort_root_min']:.3e}. Same data, one usable answer.")

    print("\n7. A WITHDRAWN RATING IS CENSORING, NOT AN OUTCOME")
    wb = withdrawal_bias()
    print(f"   {wb['n_withdrawn']} of 4,000 firms had a rating withdrawn, at a hazard that rises "
          f"as the grade falls.")
    print(f"   {'grade':>7}{'true 5y PD':>14}{'censored':>12}{'err bp':>9}"
          f"{'NR as a state':>16}{'err bp':>10}")
    for row in wb["rows"]:
        print(f"   {row['grade']:>7}{row['true_pct']:>13.4f}%{row['censored_pct']:>11.4f}%"
              f"{row['censored_error_bp']:>+9.0f}{row['nr_state_pct']:>15.4f}%"
              f"{row['nr_error_bp']:>+10.0f}")
    print(f"   Treating NR as an absorbing state parks {wb['nr_mass_5y'] * 100:.2f}% of the BBB "
          f"cohort in NR after five")
    print(f"   years and UNDERSTATES every default probability -- worst "
          f"{wb['worst_nr_error_bp']:.0f} bp against "
          f"{wb['worst_censored_error_bp']:.0f} bp for censoring.")

    print("\n" + "=" * W)
    print("THE RULE: a published transition matrix is a COHORT ESTIMATE with structural zeros, "
          "so it has")
    print("no generator. Take powers of it freely; never take a ROOT of it without checking "
          "the sign.")
    print("=" * W)
