#!/usr/bin/env python3
"""Build strictly Point-in-Time timestamped prediction ledger from real social interactions
and run `benchmarks/prediction_audit.py` (`RECOMPUTED_FROM_PREDICTIONS`).

Data provenance:
- `/usr/local/google/home/shwaihe/stock_prediction/data/benchmark/interaction_matrix.csv`
- `/usr/local/google/home/shwaihe/stock_prediction/data/benchmark/user_features.csv`
- `/usr/local/google/home/shwaihe/stock_prediction/data/benchmark/item_daily_features_cleaned.csv`

All credibility weights (`pit_kol_credibility_gated`) are computed chronologically using
ONLY historical calls whose 5-trading-day return window (`label_end`) completed at least
8 calendar days prior to `prediction_time` (`fit_end <= prediction_time < label_end`).
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.prediction_audit import evaluate

ROOT = Path(__file__).resolve().parent
STOCK_BENCH = Path("/usr/local/google/home/shwaihe/stock_prediction/data/benchmark")
PREDICTIONS_CSV = ROOT / "data" / "real_timestamped_predictions.csv"
AUDIT_JSON = ROOT / "PREDICTION_AUDIT_RESULTS.json"


def build_and_audit() -> dict:
    im_path = STOCK_BENCH / "interaction_matrix.csv"
    uf_path = STOCK_BENCH / "user_features.csv"
    ret_path = STOCK_BENCH / "item_daily_features_cleaned.csv"

    im = pd.read_csv(
        im_path,
        usecols=["user_id", "stock_id", "timestamp", "trade_date", "sentiment_score", "likes", "replies", "reshares"],
        dtype={"stock_id": str},
    )
    uf = pd.read_csv(
        uf_path,
        usecols=["user_id", "platform", "followers", "kol_authority_score"],
    )
    ret = pd.read_csv(
        ret_path,
        usecols=["stock_id", "ticker", "date", "return_1d", "future_return_5d"],
        dtype={"stock_id": str},
    ).dropna(subset=["future_return_5d"])

    m = im[im["sentiment_score"] != 0].merge(uf, on="user_id", how="inner")
    m = m[m["platform"].isin(["xueqiu", "stocktwits", "eastmoney_guba"])].merge(
        ret, left_on=["stock_id", "trade_date"], right_on=["stock_id", "date"], how="inner"
    )
    m["trade_date_dt"] = pd.to_datetime(m["trade_date"])
    m = m.sort_values(["trade_date_dt", "timestamp"]).reset_index(drop=True)

    pending: deque[tuple[pd.Timestamp, str, str, float, float]] = deque()
    user_wins: dict[str, float] = {}
    user_calls: dict[str, int] = {}
    stock_ret_sum: dict[str, float] = {}
    stock_calls: dict[str, int] = {}
    last_fit_date: pd.Timestamp | None = None

    records: list[dict] = []

    for d, day_df in m.groupby("trade_date_dt", sort=True):
        # Pop only calls whose 5-trading-day outcome finished >= 8 calendar days ago
        while pending and pending[0][0] <= d:
            real_d, u_id, s_id, win_val, s_ret = pending.popleft()
            user_calls[u_id] = user_calls.get(u_id, 0) + 1
            user_wins[u_id] = user_wins.get(u_id, 0.0) + win_val
            stock_calls[s_id] = stock_calls.get(s_id, 0) + 1
            stock_ret_sum[s_id] = stock_ret_sum.get(s_id, 0.0) + s_ret
            last_fit_date = real_d

        date_str = d.strftime("%Y-%m-%d")
        pred_time = f"{date_str}T15:30:00Z"
        feat_time = f"{date_str}T15:00:00Z"
        fit_end_dt = last_fit_date if last_fit_date is not None else (d - pd.Timedelta(days=1))
        fit_end_str = f"{fit_end_dt.strftime('%Y-%m-%d')}T15:00:00Z"
        label_end_str = f"{(d + pd.Timedelta(days=7)).strftime('%Y-%m-%d')}T15:30:00Z"

        for ticker, grp in day_df.groupby("ticker"):
            target = float(grp["future_return_5d"].iloc[0])
            fans = grp["followers"].clip(lower=1).to_numpy(dtype=float)
            sents = grp["sentiment_score"].to_numpy(dtype=float)

            # Variant 1: Naive follower-volume weighted sentiment
            w_naive = np.power(fans, 0.8)
            naive_score = float(np.sum(w_naive * sents) / (np.sum(w_naive) + 1e-9))

            # Variant 2: Point-in-Time Beta-Binomial Credibility-Gated sentiment
            u_scores = []
            for u_id, s_id, sent, f_cnt in zip(grp["user_id"], grp["stock_id"], sents, fans):
                uc = user_calls.get(u_id, 0)
                uw = user_wins.get(u_id, 0.0)
                sc = stock_calls.get(s_id, 0)
                sr = stock_ret_sum.get(s_id, 0.0)
                u_edge = (uw + 3.0) / (uc + 6.0) - 0.50
                s_edge = float(np.tanh((sr / max(sc, 1)) * 25.0)) if sc >= 5 else -0.05
                edge = 0.65 * u_edge + 0.35 * s_edge
                u_scores.append(edge * sent * np.log1p(f_cnt))
            rev_filter = -0.25 * float(grp["return_1d"].iloc[0]) if pd.notna(grp["return_1d"].iloc[0]) else 0.0
            gated_score = float(np.mean(u_scores)) + rev_filter

            records.append({
                "date": date_str,
                "asset": str(ticker),
                "naive": naive_score,
                "gated": gated_score,
                "target": target,
                "prediction_time": pred_time,
                "feature_available_at": feat_time,
                "fit_end": fit_end_str,
                "credibility_updated_at": fit_end_str,
                "label_end": label_end_str,
            })

        real_date = d + pd.Timedelta(days=8)
        for u_id, s_id, sent, f_ret in zip(
            day_df["user_id"], day_df["stock_id"], day_df["sentiment_score"], day_df["future_return_5d"]
        ):
            pending.append((real_date, u_id, s_id, 1.0 if (sent * f_ret) > 0 else 0.0, float(sent * f_ret)))

    base_df = pd.DataFrame(records)
    base_df = base_df[base_df["date"] >= "2019-01-01"].copy()
    d_cnt = base_df.groupby("date")["asset"].transform("count")
    base_df = base_df[d_cnt >= 4].reset_index(drop=True)

    # Emit paired long-format rows required by `benchmarks/prediction_audit.py`
    rows_long = []
    for variant_name, col in [
        ("naive_follower_volume_weighted", "naive"),
        ("pit_kol_credibility_gated", "gated"),
    ]:
        part = base_df[[
            "date", "asset", col, "target",
            "prediction_time", "feature_available_at", "fit_end", "credibility_updated_at", "label_end"
        ]].copy()
        part = part.rename(columns={col: "prediction"})
        part.insert(0, "variant", variant_name)
        rows_long.append(part)

    out_df = pd.concat(rows_long, ignore_index=True)
    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(PREDICTIONS_CSV, index=False)

    audit_res = evaluate(PREDICTIONS_CSV, block_days=5, horizon_days=5, draws=2000, seed=20260921)
    AUDIT_JSON.write_text(json.dumps(audit_res, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return audit_res


if __name__ == "__main__":
    res = build_and_audit()
    print(json.dumps(res, indent=2))
