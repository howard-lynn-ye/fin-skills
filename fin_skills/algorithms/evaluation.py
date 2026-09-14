"""Expanding-window forecast comparison, with explicit gaps and complete-fold scoring."""
from __future__ import annotations

import numpy as np

from .core import strings
from .runtime import integer, validate_data


def walk_forward(series, *, initial_train: int, horizon: int = 1, gap: int = 0,
                 candidates=("naive", "mean", "drift"), metric: str = "mae",
                 seasonal_period: int | None = None, registry=None) -> dict:
    """Select and refit a forecast method using chronological validation folds.

    Each fit receives only series[:train_end]. Predict gap+horizon steps, discard gap,
    and score the following horizon rows. Validation windows do not overlap. A candidate
    failing ANY fold is excluded from ranking, and the error remains in the report.
    The winner's score is a selection/validation score, not an untouched test estimate.
    The returned forecast is refit on all supplied history and starts AFTER its last row.
    """
    if registry is None:
        from . import _DEFAULT
        registry = _DEFAULT
    initial_train = integer(initial_train, "initial_train")
    horizon = integer(horizon, "horizon", maximum=1_000)
    gap = integer(gap, "gap", maximum=1_000, minimum=0)
    candidates = strings(candidates, "candidates")
    if not 1 <= len(candidates) <= 20:
        raise ValueError("candidates must contain 1..20 distinct algorithm IDs")
    if metric not in ("mae", "rmse"):
        raise ValueError("metric must be mae or rmse")
    raw = {"series": series}
    if seasonal_period is not None:
        raw["seasonal_period"] = seasonal_period
    clean, n = validate_data(registry, "forecast", raw)
    ends = list(range(initial_train, n - gap - horizon + 1, horizon))
    if not 2 <= len(ends) <= 200:
        raise ValueError("walk_forward requires 2..200 complete validation folds")
    for id in candidates:
        if registry.get(id).task != "forecast":
            raise ValueError(f"{id} is not a forecast algorithm")
    series = np.asarray(clean["series"])
    folds = [{"train_start": 0, "train_end": end, "validation_start": end + gap,
              "validation_end": end + gap + horizon} for end in ends]
    ranked, rejected = [], {}
    for id in candidates:
        errors, fold_scores = [], []
        for i, fold in enumerate(folds):
            data = dict(clean, series=series[:fold["train_end"]].copy())
            try:
                prediction = np.asarray(registry.run(id, data, horizon=gap + horizon), float)
                if prediction.shape != (gap + horizon,) or not np.isfinite(prediction).all():
                    raise ValueError("forecast must have requested horizon and finite values")
                actual = series[fold["validation_start"]:fold["validation_end"]]
                residual = prediction[gap:] - actual
                loss = np.abs(residual) if metric == "mae" else residual**2
                if not np.isfinite(loss).all():
                    raise ValueError("validation loss overflowed")
                errors.extend(loss.tolist())
                score = float(loss.mean())
                fold_scores.append(score if metric == "mae" else float(np.sqrt(score)))
            except Exception as exc:  # Preserve failed candidates; never rank partial scores.
                rejected[id] = {"fold": i, "error": f"{type(exc).__name__}: {exc}",
                                "completed_folds": len(fold_scores)}
                break
        if id not in rejected:
            score = float(np.mean(errors))
            ranked.append({"algorithm": id, "score": score if metric == "mae" else
                           float(np.sqrt(score)), "fold_scores": fold_scores,
                           "scored_observations": len(errors)})
    ranked.sort(key=lambda row: (row["score"], row["algorithm"]))
    selected = ranked[0]["algorithm"] if ranked else None
    report = {"selected": selected, "metric": metric, "ranking": ranked,
              "rejected": rejected, "folds": folds, "gap": gap,
              "basis": "expanding-window validation; lower error is better",
              "test_performance_estimated": False,
              "warning": "Selection uses these validation folds; reserve a later untouched test period.",
              "unused_validation_tail": n - folds[-1]["validation_end"],
              "forecast": None, "forecast_start_position": n}
    if selected:
        try:
            future = np.asarray(registry.run(selected, clean, horizon=horizon), float)
            if future.shape != (horizon,) or not np.isfinite(future).all():
                raise ValueError("refit forecast must have requested horizon and finite values")
            report["forecast"] = future.tolist()
        except Exception as exc:
            report["refit_error"] = f"{type(exc).__name__}: {exc}"
    return report
