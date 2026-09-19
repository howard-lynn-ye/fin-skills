#!/usr/bin/env python3
"""Experiment E3: Numerical Parity & Guard Runtime Scaling Benchmark.

Evaluates:
1. Numerical parity of `fin-skills` algorithms and statistical primitives against
   reference implementations (`scipy`, `statsmodels`, `sklearn`, and exact closed forms).
2. Wall-clock runtime and memory scaling of `fin-skills` executable guards (`Bundle.check()`)
   across dataset sizes N in {250, 1000, 5000, 25000} rows.

Outputs machine-readable results to `benchmarks/PARITY_AND_COST_RESULTS.json`.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from fin_skills.algorithms import default_registry
from fin_skills.api import Bundle

ROOT = Path(__file__).resolve().parent
DATA_SIZES = (250, 1000, 5000, 25000)


def _reference_min_variance_weights(returns: pd.DataFrame) -> np.ndarray:
    """Reference long-only minimum variance weights via scipy SLSQP."""
    cov = returns.cov().to_numpy()
    k = cov.shape[0]
    w0 = np.full(k, 1.0 / k)
    bounds = [(0.0, 1.0)] * k
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = optimize.minimize(
        lambda w: float(w @ cov @ w),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    return res.x


def _reference_fractional_diff_weights(d: float, threshold: float = 1e-4) -> np.ndarray:
    """Exact binomial recursion for fractional differentiation weights."""
    weights = [1.0]
    k = 1
    while True:
        w_next = -weights[-1] * (d - k + 1.0) / k
        if abs(w_next) < threshold:
            break
        weights.append(w_next)
        k += 1
    return np.array(weights[::-1], dtype=float)


def run_numerical_parity_suite(seed: int = 42) -> list[dict[str, Any]]:
    """Run numerical parity checks across core financial & statistical primitives."""
    rng = np.random.default_rng(seed)
    n_obs, n_assets = 600, 5
    factor = rng.normal(0.0004, 0.012, size=n_obs)
    idio = rng.normal(0.0, 0.008, size=(n_obs, n_assets))
    betas = np.array([0.7, 0.9, 1.0, 1.15, 1.3])
    asset_returns = pd.DataFrame(
        np.outer(factor, betas) + idio,
        columns=[f"ASSET_{i}" for i in range(n_assets)],
    )
    reg = default_registry()
    results: list[dict[str, Any]] = []

    # 1. Equal-Weight Portfolio Parity
    ew_out = np.asarray(reg.run("equal_weight", {"asset_returns": asset_returns}))
    ew_ref = np.full(n_assets, 1.0 / n_assets)
    err_ew = float(np.max(np.abs(ew_out - ew_ref)))
    results.append({
        "primitive": "equal_weight_portfolio",
        "reference_library": "exact_closed_form (1/N)",
        "max_abs_error": err_ew,
        "tolerance": 1e-12,
        "passed": err_ew <= 1e-12,
    })

    # 2. Inverse-Volatility Risk Parity Weights
    iv_out = np.asarray(reg.run("inverse_volatility", {"asset_returns": asset_returns}))
    vols = asset_returns.std(ddof=1).to_numpy()
    iv_ref = (1.0 / vols) / np.sum(1.0 / vols)
    err_iv = float(np.max(np.abs(iv_out - iv_ref)))
    results.append({
        "primitive": "inverse_volatility_weights",
        "reference_library": "numpy/pandas exact sample std^-1",
        "max_abs_error": err_iv,
        "tolerance": 1e-10,
        "passed": err_iv <= 1e-10,
    })

    # 3. Minimum-Variance Portfolio Optimization Parity (vs scipy SLSQP / PyPortfolioOpt formulation)
    mv_out = np.asarray(reg.run("min_variance", {"asset_returns": asset_returns}))
    mv_ref = _reference_min_variance_weights(asset_returns)
    cov = asset_returns.cov().to_numpy()
    var_out = float(mv_out @ cov @ mv_out)
    var_ref = float(mv_ref @ cov @ mv_ref)
    rel_obj_diff = abs(var_out - var_ref) / max(var_ref, 1e-12)
    results.append({
        "primitive": "min_variance_portfolio_qp",
        "reference_library": "scipy.optimize.minimize (SLSQP convex QP)",
        "max_abs_error": float(np.max(np.abs(mv_out - mv_ref))),
        "relative_objective_diff": float(rel_obj_diff),
        "tolerance": 1e-3,
        "passed": rel_obj_diff <= 1e-3,
    })

    # 4. Parametric Normal VaR & Expected Shortfall (95%)
    r_series = asset_returns["ASSET_0"]
    nvar_out = reg.run("normal_var_es", {"returns": r_series})
    mu = float(r_series.mean())
    sigma = float(r_series.std(ddof=1))
    z_alpha = float(stats.norm.ppf(0.05))
    var_ref_val = -(mu + sigma * z_alpha)
    es_ref_val = -(mu - sigma * float(stats.norm.pdf(z_alpha)) / 0.05)
    err_var = abs(float(nvar_out["var"]) - var_ref_val)
    err_es = abs(float(nvar_out["expected_shortfall"]) - es_ref_val)
    max_err_nvar = max(err_var, err_es)
    results.append({
        "primitive": "parametric_normal_var_es_95",
        "reference_library": "scipy.stats.norm analytical VaR/ES",
        "max_abs_error": max_err_nvar,
        "tolerance": 1e-6,
        "passed": max_err_nvar <= 1e-6,
    })

    # 5. Historical Simulation VaR & Expected Shortfall (95%)
    hvar_out = reg.run("historical_var_es", {"returns": r_series})
    q_05 = float(np.quantile(r_series.to_numpy(), 0.05))
    hvar_ref = -q_05
    tail = r_series.to_numpy()[r_series.to_numpy() <= q_05]
    hes_ref = -float(np.mean(tail))
    max_err_hvar = max(
        abs(float(hvar_out["var"]) - hvar_ref),
        abs(float(hvar_out["expected_shortfall"]) - hes_ref),
    )
    results.append({
        "primitive": "historical_var_es_95",
        "reference_library": "numpy exact empirical quantile & tail mean",
        "max_abs_error": max_err_hvar,
        "tolerance": 1e-6,
        "passed": max_err_hvar <= 1e-6,
    })

    # 6. Historical Per-Period Volatility Parity
    hvol_out = float(reg.run("historical_volatility", {"returns": r_series})["volatility"])
    hvol_ref = float(r_series.std(ddof=1))
    err_hvol = abs(hvol_out - hvol_ref)
    results.append({
        "primitive": "historical_period_volatility",
        "reference_library": "pandas.Series.std(ddof=1)",
        "max_abs_error": err_hvol,
        "tolerance": 1e-10,
        "passed": err_hvol <= 1e-10,
    })

    # 7. Fractional Differentiation Binomial Weight Recursion Parity (Lopez de Prado, 2018)
    from fin_skills.ml.frac_diff import adf_tstat, ffd_weights, statsmodels_adf, weights_closed_form
    ffd_out = np.asarray(ffd_weights(d=0.4, thresh=1e-4)).ravel()
    ffd_ref = np.asarray(weights_closed_form(d=0.4, size=len(ffd_out))).ravel()
    err_ffd = float(np.max(np.abs(ffd_out - ffd_ref)))
    results.append({
        "primitive": "fractional_differentiation_weights_ffd",
        "reference_library": "exact closed-form binomial series (d=0.4)",
        "max_abs_error": err_ffd,
        "tolerance": 1e-12,
        "passed": err_ffd <= 1e-12,
    })

    # 8. Augmented Dickey-Fuller (ADF) Unit Root Test Parity vs statsmodels
    price_path = 100.0 + np.cumsum(r_series.to_numpy())
    adf_internal = float(adf_tstat(price_path, lags=1))
    adf_sm_res = statsmodels_adf(price_path, lags=1)
    adf_sm = float(adf_sm_res[0] if isinstance(adf_sm_res, tuple) else adf_sm_res)
    err_adf = abs(adf_internal - adf_sm)
    results.append({
        "primitive": "augmented_dickey_fuller_tstat",
        "reference_library": "statsmodels.tsa.stattools.adfuller",
        "max_abs_error": err_adf,
        "tolerance": 1e-6,
        "passed": err_adf <= 1e-6,
    })

    return results


def _build_synthetic_bundle(n_rows: int, seed: int = 7) -> Bundle:
    """Generate realistic Bundle of length `n_rows` for guard scaling tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-03", periods=n_rows, freq="B")
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, size=n_rows))), index=dates)
    ret = close.pct_change().fillna(0.0)
    turnover = pd.Series(rng.uniform(0.01, 0.05, size=n_rows), index=dates)
    adv = pd.Series(rng.uniform(5e6, 2e7, size=n_rows), index=dates)
    prices = pd.DataFrame({
        "open": close * (1.0 - 0.002),
        "high": close * (1.0 + 0.006),
        "low": close * (1.0 - 0.006),
        "close": close,
        "volume": rng.integers(100_000, 5_000_000, size=n_rows),
    }, index=dates)
    return Bundle(
        returns=ret,
        turnover=turnover,
        adv=adv,
        prices=prices,
        close=close,
        book=1_000_000.0,
        rf=0.02,
        periods_per_year=252,
    )


def run_guard_scaling_benchmark(repetitions: int = 5) -> dict[str, Any]:
    """Measure wall-clock runtime (ms) and peak memory (KB) across dataset sizes."""
    scaling_rows: list[dict[str, Any]] = []
    for n_rows in DATA_SIZES:
        bundle = _build_synthetic_bundle(n_rows)

        # Warmup
        report = bundle.check()
        n_guards_run = len(report.ran)

        timings_ms: list[float] = []
        tracemalloc.start()
        for _ in range(repetitions):
            t0 = time.perf_counter()
            bundle.check()
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        scaling_rows.append({
            "n_rows": n_rows,
            "guards_evaluated": n_guards_run,
            "p50_ms": round(float(np.median(timings_ms)), 3),
            "p95_ms": round(float(np.percentile(timings_ms, 95)), 3),
            "mean_ms": round(float(np.mean(timings_ms)), 3),
            "peak_memory_kb": round(peak_bytes / 1024.0, 2),
        })

    log_n = np.log([r["n_rows"] for r in scaling_rows])
    log_t = np.log([max(r["p50_ms"], 1e-6) for r in scaling_rows])
    slope, _ = np.polyfit(log_n, log_t, 1)

    return {
        "scaling_curve": scaling_rows,
        "empirical_complexity_exponent_alpha": round(float(slope), 4),
        "complexity_class": "Sub-linear / Linear O(N)" if slope <= 1.05 else f"O(N^{slope:.2f})",
        "max_n_tested": max(DATA_SIZES),
        "p50_ms_at_25k_rows": scaling_rows[-1]["p50_ms"],
    }


def main() -> int:
    parity_results = run_numerical_parity_suite()
    scaling_results = run_guard_scaling_benchmark()

    all_parity_passed = all(item["passed"] for item in parity_results)
    summary = {
        "experiment_id": "E3_numerical_parity_and_guard_runtime_scaling",
        "all_parity_checks_passed": all_parity_passed,
        "parity_checks_count": len(parity_results),
        "parity_matrix": parity_results,
        "guard_runtime_scaling": scaling_results,
    }
    output_path = ROOT / "PARITY_AND_COST_RESULTS.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if all_parity_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
