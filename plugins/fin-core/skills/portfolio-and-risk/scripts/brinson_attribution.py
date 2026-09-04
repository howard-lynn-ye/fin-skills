#!/usr/bin/env python3
"""Brinson performance attribution -- Fachler and Hood-Beebower -- with Carino linking.

WHY this exists: there is no mature Brinson attribution library in Python. The formulas
are four lines each, so everyone re-derives them, and two things go wrong every time.

1. THE EFFECTS MUST SUM TO `portfolio_return - benchmark_return`. Exactly, to floating
   point. That identity is not a nicety, it is the only self-check attribution has, and it
   holds algebraically for any weights and returns that satisfy `sum(wp) == sum(wb) == 1`.
   So when it does NOT close, the arithmetic is fine and the DATA is wrong -- almost
   always weights and returns measured over different intervals (start-of-period weights
   against a return series that already includes a mid-period trade), or a missing cash
   line. Every function here raises with that diagnosis rather than reporting a residual
   row and letting you round it away.

2. SINGLE-PERIOD EFFECTS DO NOT SUM ACROSS PERIODS. Returns compound; effects are
   arithmetic. Add four quarterly allocation effects and the total will miss
   `Rp - Rb` by the cross-product terms -- and no Python library implements any linking
   algorithm to fix it. Carino (1999) does: scale each period's effects by `k_t / K`,
   where `k_t` is a logarithmic smoothing coefficient for that period and `K` the same
   coefficient for the whole horizon. The linked effects then sum to the compounded active
   return exactly, and each one stays interpretable as "what this sector contributed".

FACHLER vs HOOD-BEEBOWER -- one term apart, and it is the term people argue about:

    BHB      allocation_i = (wp_i - wb_i) *  rb_i
    Fachler  allocation_i = (wp_i - wb_i) * (rb_i - Rb)
    both     selection_i  =  wb_i * (rp_i - rb_i)
             interaction_i= (wp_i - wb_i) * (rp_i - rb_i)

Totals are identical -- the difference sums to `Rb * sum(wp_i - wb_i) == 0`. Only the
per-sector split changes, and Fachler's is the defensible one: it asks "did you overweight
a sector that beat the BENCHMARK", where BHB asks "did you overweight a sector with a
POSITIVE return". In a year when everything is up, BHB credits every overweight as a good
call, including the one that lagged the index by 800bps. Use Fachler; report BHB only when
a mandate demands it.

Usage:
    from brinson_attribution import brinson_single_period, multi_period_attribution, render

    panel = pd.DataFrame({"wp": ..., "wb": ..., "rp": ..., "rb": ...}, index=sectors)
    eff = brinson_single_period(panel, method="fachler")

    linked = multi_period_attribution([panel_q1, panel_q2, panel_q3, panel_q4])
    print(render(linked))
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

REQUIRED = ("wp", "wb", "rp", "rb")
EFFECTS = ("allocation", "selection", "interaction")
TOTAL = "TOTAL"


class ReconciliationError(AssertionError):
    """Raised when attribution effects do not sum to the active return.

    This is a DATA error, never an arithmetic one. See the message for the shortlist.
    """


# --------------------------------------------------------------------------------------
# single period
# --------------------------------------------------------------------------------------
def _validate(panel: pd.DataFrame, weight_tol: float) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing column(s) {missing}; "
                         f"expected {list(REQUIRED)}")
    out = panel[list(REQUIRED)].astype(float)
    if out.isna().any().any():
        bad = out.index[out.isna().any(axis=1)].tolist()
        raise ValueError(f"NaN in panel rows {bad}. A sector held at zero weight still "
                         f"needs a return (use the benchmark's); NaN is not zero.")
    sp, sb = float(out["wp"].sum()), float(out["wb"].sum())
    if abs(sp - 1.0) > weight_tol or abs(sb - 1.0) > weight_tol:
        raise ReconciliationError(
            f"weights do not sum to 1: sum(wp)={sp:.10f}, sum(wb)={sb:.10f} "
            f"(tolerance {weight_tol:g}).\n"
            f"  The Brinson identity requires sum(wp) == sum(wb) == 1. The usual causes:\n"
            f"    - cash / FX / residual sleeve omitted from the sector breakdown;\n"
            f"    - weights taken at a different timestamp from the returns;\n"
            f"    - a sector present in the portfolio but absent from the benchmark rows.\n"
            f"  Add the missing line at its real weight (return 0.0 for cash). Do NOT "
            f"renormalise -- that silently reallocates the gap across every sector.")
    return out


def brinson_single_period(panel: pd.DataFrame,
                          method: str = "fachler",
                          tol: float = 1e-10,
                          weight_tol: float = 1e-8) -> pd.DataFrame:
    """Single-period Brinson attribution. Returns per-sector effects plus a TOTAL row.

    panel  : DataFrame indexed by sector with columns
             wp (portfolio weight), wb (benchmark weight),
             rp (portfolio return in that sector), rb (benchmark return in that sector).
             Weights are start-of-period; returns cover exactly that period. If those two
             statements are not both true of your data, no attribution model can help you.
    method : "fachler" (default, recommended) or "bhb".
    tol    : reconciliation tolerance on sum(effects) - (Rp - Rb).

    The returned frame carries `.attrs` with rp, rb and active.
    """
    m = method.lower()
    if m not in ("fachler", "bhb"):
        raise ValueError(f"method must be 'fachler' or 'bhb'; got {method!r}")
    p = _validate(panel, weight_tol)
    wp, wb, rp, rb = (p["wp"].to_numpy(), p["wb"].to_numpy(),
                      p["rp"].to_numpy(), p["rb"].to_numpy())

    r_port = float(wp @ rp)
    r_bench = float(wb @ rb)
    active = r_port - r_bench

    dw = wp - wb
    alloc = dw * (rb - r_bench) if m == "fachler" else dw * rb
    sel = wb * (rp - rb)
    inter = dw * (rp - rb)

    eff = pd.DataFrame({"allocation": alloc, "selection": sel, "interaction": inter},
                       index=p.index)
    eff["total"] = eff.sum(axis=1)
    eff.loc[TOTAL] = eff.sum(axis=0)

    gap = float(eff.loc[TOTAL, "total"]) - active
    if abs(gap) > tol:
        raise ReconciliationError(
            f"{m} effects sum to {eff.loc[TOTAL, 'total']:.12f} but the active return is "
            f"{active:.12f} (gap {gap:.3e} > tol {tol:g}).\n"
            f"  The Brinson identity is exact, so this is a data problem, not a rounding "
            f"one. Check, in order:\n"
            f"    1. weights and returns measured over DIFFERENT intervals -- the single "
            f"most common cause;\n"
            f"    2. rp/rb that are already weighted contributions rather than sector "
            f"returns;\n"
            f"    3. a sector traded to zero mid-period, so its start weight does not "
            f"describe the holding.")

    eff.attrs.update({"rp": r_port, "rb": r_bench, "active": active, "method": m})
    return eff


def brinson_fachler(panel: pd.DataFrame, **kw) -> pd.DataFrame:
    """Brinson-Fachler: allocation measured against the TOTAL benchmark return."""
    return brinson_single_period(panel, method="fachler", **kw)


def brinson_hood_beebower(panel: pd.DataFrame, **kw) -> pd.DataFrame:
    """Brinson-Hood-Beebower (1986): allocation measured against zero. Legacy."""
    return brinson_single_period(panel, method="bhb", **kw)


# --------------------------------------------------------------------------------------
# multi-period linking -- Carino (1999)
# --------------------------------------------------------------------------------------
def carino_coefficient(r: float) -> float:
    """k = log(1+Rp) - log(1+Rb) all over (Rp - Rb). Here for the single-argument case.

    Not used directly; `carino_coefficients` handles the pair and the removable
    singularity at Rp == Rb. Kept because the scalar form log(1+r)/r is the coefficient
    that appears in the literature when the active return is zero.
    """
    return float(np.log1p(r) / r) if r != 0.0 else 1.0


def carino_coefficients(rp: Sequence[float],
                        rb: Sequence[float]) -> tuple[np.ndarray, float, float, float]:
    """Per-period smoothing coefficients k_t and the horizon coefficient K.

        k_t = [log(1+Rp_t) - log(1+Rb_t)] / (Rp_t - Rb_t)
        K   = [log(1+Rp)   - log(1+Rb)  ] / (Rp   - Rb)      with Rp, Rb COMPOUNDED

    At Rp_t == Rb_t the ratio is 0/0; the limit is 1/(1+Rp_t), which is what is used --
    a naive implementation divides by zero exactly when the two happen to tie, which is
    common in a flat month and produces a silent NaN in one period's effects.

    Returns (k_t, K, Rp_total, Rb_total).
    """
    rp = np.asarray(rp, dtype=float)
    rb = np.asarray(rb, dtype=float)
    if rp.shape != rb.shape or rp.ndim != 1:
        raise ValueError("rp and rb must be 1-D arrays of the same length")
    if (rp <= -1.0).any() or (rb <= -1.0).any():
        raise ValueError("a period return of -100% or worse makes log(1+r) undefined; "
                         "attribution cannot span a wipeout")

    d = rp - rb
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(d != 0.0, (np.log1p(rp) - np.log1p(rb)) / np.where(d != 0.0, d, 1.0),
                     1.0 / (1.0 + rp))

    rp_tot = float(np.prod(1.0 + rp) - 1.0)
    rb_tot = float(np.prod(1.0 + rb) - 1.0)
    dt = rp_tot - rb_tot
    big_k = (float((np.log1p(rp_tot) - np.log1p(rb_tot)) / dt) if dt != 0.0
             else float(1.0 / (1.0 + rp_tot)))
    return k, big_k, rp_tot, rb_tot


def carino_link(effects: Sequence[pd.DataFrame],
                rp: Sequence[float] | None = None,
                rb: Sequence[float] | None = None,
                tol: float = 1e-10) -> pd.DataFrame:
    """Link per-period Brinson effects into horizon effects that reconcile exactly.

    effects : one frame per period, as returned by `brinson_single_period` (TOTAL row
              included; it is recomputed, not trusted).
    rp, rb  : per-period portfolio and benchmark returns. Defaults to the `.attrs` each
              frame carries, so the normal call is `carino_link(list_of_effects)`.

    Each period's effects are scaled by k_t / K. Because the effects within a period
    already sum to (Rp_t - Rb_t), the scaled totals sum to
    (1/K) * sum_t [log(1+Rp_t) - log(1+Rb_t)] = (1/K) * [log(1+Rp) - log(1+Rb)]
    = Rp - Rb, the COMPOUNDED active return. That telescoping is the whole trick.
    """
    if not effects:
        raise ValueError("no periods supplied")
    if rp is None:
        rp = [float(e.attrs["rp"]) for e in effects]
    if rb is None:
        rb = [float(e.attrs["rb"]) for e in effects]
    if not (len(rp) == len(rb) == len(effects)):
        raise ValueError("rp, rb and effects must have the same length")

    k, big_k, rp_tot, rb_tot = carino_coefficients(rp, rb)
    sectors = effects[0].index.drop(TOTAL, errors="ignore")
    for i, e in enumerate(effects):
        if not sectors.equals(e.index.drop(TOTAL, errors="ignore")):
            raise ValueError(
                f"period {i} has a different sector list. Linking requires a stable "
                f"breakdown; a sector that enters mid-horizon must appear in every period "
                f"at zero weight, not be absent.")

    linked = pd.DataFrame(0.0, index=sectors, columns=list(EFFECTS))
    for kt, e in zip(k, effects):
        linked += e.loc[sectors, list(EFFECTS)] * (kt / big_k)
    linked["total"] = linked.sum(axis=1)
    linked.loc[TOTAL] = linked.sum(axis=0)

    active = rp_tot - rb_tot
    gap = float(linked.loc[TOTAL, "total"]) - active
    if abs(gap) > tol:
        raise ReconciliationError(
            f"Carino-linked effects sum to {linked.loc[TOTAL, 'total']:.12f} but the "
            f"compounded active return is {active:.12f} (gap {gap:.3e}).\n"
            f"  Linking is exact when each period already reconciles, so one of the "
            f"single-period frames does not -- rebuild them with brinson_single_period, "
            f"which refuses to return an unreconciled frame.")

    naive = sum(float(e.loc[TOTAL, "total"]) for e in effects)
    linked.attrs.update({"rp": rp_tot, "rb": rb_tot, "active": active,
                         "naive_sum": naive, "residual": naive - active,
                         "K": big_k, "k_t": k, "n_periods": len(effects),
                         "method": effects[0].attrs.get("method", "?")})
    return linked


def multi_period_attribution(panels: Sequence[pd.DataFrame],
                             method: str = "fachler",
                             tol: float = 1e-10) -> pd.DataFrame:
    """The whole pipeline: per-period panels -> per-period effects -> Carino-linked."""
    effects = [brinson_single_period(p, method=method, tol=tol) for p in panels]
    return carino_link(effects, tol=tol)


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------
def render(effects: pd.DataFrame, title: str = "BRINSON ATTRIBUTION") -> str:
    """Fixed-width text table in basis points. Effects are always quoted in bps."""
    a = effects.attrs
    w = 74
    lines = [title, "=" * w,
             f"{'sector':<22}{'alloc':>11}{'select':>11}{'interact':>11}{'total':>11}",
             "-" * w]
    for name, row in effects.iterrows():
        if name == TOTAL:
            lines.append("-" * w)
        # + 0.0 collapses negative zero, which shows up on any sector held at benchmark
        lines.append(f"{str(name):<22}{row['allocation'] * 1e4 + 0.0:>10.1f} "
                     f"{row['selection'] * 1e4 + 0.0:>10.1f} "
                     f"{row['interaction'] * 1e4 + 0.0:>10.1f} "
                     f"{row['total'] * 1e4 + 0.0:>10.1f}")
    lines.append("=" * w)
    if "rp" in a:
        lines.append(f"portfolio {a['rp']:+.4%}   benchmark {a['rb']:+.4%}   "
                     f"active {a['active'] * 1e4:+.1f} bps"
                     + (f"   over {a['n_periods']} periods" if "n_periods" in a else ""))
    if "naive_sum" in a:
        lines += [f"naive sum of single-period effects : {a['naive_sum'] * 1e4:+.1f} bps",
                  f"Carino-linked effects              : "
                  f"{float(effects.loc[TOTAL, 'total']) * 1e4:+.1f} bps  "
                  f"(= active, exactly)",
                  f"residual the naive sum leaves      : "
                  f"{a['residual'] * 1e4:+.1f} bps  <- unattributable without linking",
                  f"K = {a['K']:.6f}   k_t = "
                  f"[{', '.join(f'{v:.6f}' for v in a['k_t'])}]"]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# offline demo
# --------------------------------------------------------------------------------------
def _demo_panels() -> list[pd.DataFrame]:
    """3 sectors, 4 periods. Returns are large enough that compounding visibly bites."""
    sectors = ["Tech", "Energy", "Staples"]
    spec = [
        # (wp,               wb,               rp,                  rb)
        ([0.50, 0.30, 0.20], [0.40, 0.30, 0.30], [0.120, -0.040, 0.030], [0.100, -0.020, 0.020]),
        ([0.55, 0.25, 0.20], [0.40, 0.30, 0.30], [-0.080, 0.090, 0.010], [-0.060, 0.070, 0.005]),
        ([0.45, 0.35, 0.20], [0.40, 0.30, 0.30], [0.150, 0.020, -0.010], [0.130, 0.040, 0.000]),
        ([0.60, 0.20, 0.20], [0.40, 0.30, 0.30], [-0.050, -0.030, 0.070], [-0.070, -0.010, 0.025]),
    ]
    return [pd.DataFrame({"wp": wp, "wb": wb, "rp": rp, "rb": rb}, index=sectors)
            for wp, wb, rp, rb in spec]


if __name__ == "__main__":
    panels = _demo_panels()

    print("=== A. one period, Fachler vs Hood-Beebower ===")
    f1 = brinson_fachler(panels[0])
    b1 = brinson_hood_beebower(panels[0])
    print(render(f1, "PERIOD 1 - BRINSON-FACHLER"))
    print()
    print(render(b1, "PERIOD 1 - BRINSON-HOOD-BEEBOWER"))
    print(f"\nBoth reconcile to the same {f1.attrs['active'] * 1e4:+.1f} bps of active "
          f"return, and both give")
    print(f"selection and interaction identically. Only ALLOCATION differs -- Tech "
          f"{f1.loc['Tech', 'allocation'] * 1e4:+.1f} bps under")
    print(f"Fachler against {b1.loc['Tech', 'allocation'] * 1e4:+.1f} bps under BHB. The "
          f"benchmark rose {f1.attrs['rb']:+.2%} that period, so BHB")
    print("credits a Tech overweight for beating ZERO; Fachler asks whether it beat the "
          "index.")

    print("\n" + "=" * 74)
    print("=== B. four periods: why the effects cannot simply be added ===")
    per = [brinson_single_period(p) for p in panels]
    for i, e in enumerate(per, 1):
        print(f"  period {i}:  Rp {e.attrs['rp']:+7.3%}   Rb {e.attrs['rb']:+7.3%}   "
              f"active {e.attrs['active'] * 1e4:+7.1f} bps   "
              f"(effects sum {e.loc[TOTAL, 'total'] * 1e4:+7.1f} bps, reconciled)")

    naive_total = sum(float(e.loc[TOTAL, "total"]) for e in per)
    rp_c = float(np.prod([1 + e.attrs["rp"] for e in per]) - 1.0)
    rb_c = float(np.prod([1 + e.attrs["rb"] for e in per]) - 1.0)
    print(f"\n  naive sum of the four active returns : {naive_total * 1e4:+.1f} bps")
    print(f"  COMPOUNDED active return             : {(rp_c - rb_c) * 1e4:+.1f} bps"
          f"   ({rp_c:+.4%} - {rb_c:+.4%})")
    print(f"  unexplained residual                 : "
          f"{(naive_total - (rp_c - rb_c)) * 1e4:+.1f} bps")
    print("  Every single period reconciles perfectly and the sum still misses by that "
          "much.")
    print("  Nothing is wrong with the periods -- addition is simply the wrong operator "
          "for")
    print("  compounded returns, and the residual grows with horizon and volatility.")

    print("\n=== C. Carino linking closes it ===")
    linked = multi_period_attribution(panels)
    print(render(linked, "FOUR-PERIOD CARINO-LINKED ATTRIBUTION"))
    resid = float(linked.loc[TOTAL, "total"]) - linked.attrs["active"]
    print(f"\nLinked total minus compounded active return = {resid:.3e} -- floating-point "
          f"zero.")
    print("Each sector row is still readable as that sector's contribution to the "
          "four-period")
    print("active return, which a residual row bolted onto a naive sum never is.")

    print("\n" + "=" * 74)
    print("=== D. the guard: what a stale weight vector looks like ===")
    broken = panels[0].copy()
    broken.loc["Staples", "wp"] = 0.14      # a 6% cash sleeve nobody put in the table

    # what an unguarded implementation would have returned, computed the long way
    wp, wb = broken["wp"].to_numpy(), broken["wb"].to_numpy()
    rp_v, rb_v = broken["rp"].to_numpy(), broken["rb"].to_numpy()
    rb_tot = float(wb @ rb_v)
    unguarded = float(((wp - wb) * (rb_v - rb_tot)).sum()
                      + (wb * (rp_v - rb_v)).sum()
                      + ((wp - wb) * (rp_v - rb_v)).sum())
    true_active = float(wp @ rp_v) - rb_tot
    print(f"  effects an unguarded Fachler returns : {unguarded * 1e4:+.1f} bps")
    print(f"  actual active return of those inputs : {true_active * 1e4:+.1f} bps")
    print(f"  silently mis-stated by               : "
          f"{(unguarded - true_active) * 1e4:+.1f} bps\n")
    try:
        brinson_fachler(broken)
    except ReconciliationError as exc:
        print("brinson_fachler(panel_with_missing_cash_line) raised instead:\n")
        print(f"  ReconciliationError: {exc}")
    print(f"\nThat is the whole point of the assertion. A 6% cash sleeve missing from the "
          f"table\nmoves the answer by {abs(unguarded - true_active) * 1e4:.0f} bps with "
          f"nothing in the output to show it, and every sector row\nstays plausible enough "
          f"to defend in a client meeting.")
