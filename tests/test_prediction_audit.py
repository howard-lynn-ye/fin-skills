import numpy as np
import pandas as pd
import pytest

from benchmarks.prediction_audit import evaluate


@pytest.fixture
def predictions(tmp_path):
    rng = np.random.default_rng(22)
    rows = []
    for day in pd.bdate_range("2025-01-01", periods=20):
        for asset in range(6):
            target = rng.normal()
            for variant, sign in [("baseline", -1), ("gated", 1)]:
                rows.append(dict(variant=variant, date=str(day.date()), asset=str(asset),
                    prediction=sign * target + rng.normal(0, .1), target=target,
                    prediction_time=day, feature_available_at=day - pd.Timedelta(days=1),
                    fit_end="2024-12-01", label_end=day + pd.Timedelta(days=7)))
    path = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_recompute_paired_scores_not_saved_summaries(predictions):
    result = evaluate(predictions, draws=100)
    assert result["variants"]["baseline"]["mean_daily_rank_ic"] < -.8
    assert result["variants"]["gated"]["mean_daily_rank_ic"] > .8
    assert result["paired_comparisons"]["gated minus baseline"]["ci95"][0] > 1
    assert result == evaluate(predictions, draws=100)


@pytest.mark.parametrize("field", ["fit_end", "feature_available_at", "credibility_updated_at"])
def test_future_metadata_rejected(predictions, field):
    frame = pd.read_csv(predictions)
    frame[field] = "2026-01-01"
    frame.to_csv(predictions, index=False)
    with pytest.raises(ValueError, match="future"):
        evaluate(predictions, draws=100)


def test_duplicate_and_unpaired_rows_rejected(predictions):
    frame = pd.read_csv(predictions)
    frame.iloc[1:].to_csv(predictions, index=False)
    with pytest.raises(ValueError, match="same date/asset"):
        evaluate(predictions, draws=100)
    pd.concat([frame, frame.iloc[:1]]).to_csv(predictions, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(predictions, draws=100)
