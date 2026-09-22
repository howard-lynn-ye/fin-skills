"""Recompute prediction IC with point-in-time checks and date-block uncertainty.

CSV columns: variant,date,asset,prediction,target,prediction_time,feature_available_at,
fit_end,label_end. Optional credibility_updated_at is checked when present. fit_end
means latest LABEL AVAILABILITY used to fit, not just the date of a training row.
This checks declared timestamps; it cannot authenticate them or rerun model training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REQUIRED = {"variant", "date", "asset", "prediction", "target", "prediction_time",
            "feature_available_at", "fit_end", "label_end"}


def evaluate(path: Path, *, block_days=5, horizon_days=5, draws=2000, seed=20260921):
    if block_days < horizon_days or horizon_days < 1 or draws < 100:
        raise ValueError("block_days must cover the target horizon; draws >= 100")
    frame = pd.read_csv(path)
    if REQUIRED - set(frame):
        raise ValueError(f"missing columns: {sorted(REQUIRED - set(frame))}")
    if frame.empty or frame[list(REQUIRED)].isna().any().any():
        raise ValueError("empty data or missing required values")
    if frame.duplicated(["variant", "date", "asset"]).any():
        raise ValueError("duplicate variant/date/asset")
    if not np.isfinite(frame[["prediction", "target"]].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite predictions or outcomes")
    times = ["prediction_time", "feature_available_at", "fit_end", "label_end"]
    if "credibility_updated_at" in frame:
        times.append("credibility_updated_at")
    for key in times:
        frame[key] = pd.to_datetime(frame[key], utc=True, errors="raise")
        if frame[key].isna().any():
            raise ValueError(f"missing {key}")
    for key in ("feature_available_at", "fit_end", "credibility_updated_at"):
        if key in frame and (frame[key] > frame.prediction_time).any():
            raise ValueError(f"future information in {key}")
    if (frame.label_end <= frame.prediction_time).any():
        raise ValueError("outcome must end after prediction")
    variants, daily = {}, {}
    for variant, group in frame.groupby("variant"):
        rows = []
        undefined = []
        for day, cross in group.groupby("date"):
            if len(cross) < 3 or cross.prediction.nunique() < 2 or cross.target.nunique() < 2:
                undefined.append(str(day))
            else:
                rows.append((day, float(spearmanr(cross.prediction, cross.target).statistic)))
        series = pd.Series(dict(rows), dtype=float).sort_index()
        if len(series) < 2 * block_days:
            raise ValueError(f"{variant}: insufficient valid dates for block uncertainty")
        daily[str(variant)] = series
        pooled = float(spearmanr(group.prediction, group.target).statistic)
        variants[str(variant)] = {"rows": len(group), "valid_dates": len(series),
                                  "undefined_dates": undefined,
                                  "pooled_rank_ic": pooled if np.isfinite(pooled) else None,
                                  "mean_daily_rank_ic": float(series.mean()),
                                  "daily_mean_ci95": block_ci(series.to_numpy(), block_days, draws, seed)}
    comparisons = {}
    names = sorted(daily)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            l = frame[frame.variant.astype(str) == left].set_index(["date", "asset"])
            r = frame[frame.variant.astype(str) == right].set_index(["date", "asset"])
            if set(l.index) != set(r.index):
                raise ValueError("paired variants must contain the same date/asset rows")
            if not np.allclose(l.target, r.reindex(l.index).target, rtol=0, atol=0):
                raise ValueError("paired variants disagree on outcomes")
            paired = pd.concat([daily[left], daily[right]], axis=1).dropna()
            if len(paired) < 2 * block_days:
                raise ValueError("insufficient paired valid dates")
            delta = paired.iloc[:, 1].to_numpy() - paired.iloc[:, 0].to_numpy()
            comparisons[f"{right} minus {left}"] = {"paired_dates": len(delta),
                "mean_daily_ic_difference": float(delta.mean()),
                "ci95": block_ci(delta, block_days, draws, seed)}
    return {"status": "RECOMPUTED_FROM_PREDICTIONS", "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(frame), "variants": variants, "paired_comparisons": comparisons,
            "block_days": block_days, "horizon_days": horizon_days, "draws": draws, "seed": seed,
            "limits": "Timestamp assertions are supplied metadata, not authenticated provenance. "
                      "Prediction evaluation does not reproduce training or establish trading profitability."}


def block_ci(values, block, draws, seed):
    rng = np.random.default_rng(seed)
    n = len(values)
    starts = rng.integers(0, n, size=(draws, int(np.ceil(n / block))))
    indices = (starts[:, :, None] + np.arange(block)) % n
    means = values[indices.reshape(draws, -1)[:, :n]].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-days", type=int, default=5)
    parser.add_argument("--horizon-days", type=int, default=5)
    args = parser.parse_args()
    result = evaluate(args.predictions, block_days=args.block_days, horizon_days=args.horizon_days)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
