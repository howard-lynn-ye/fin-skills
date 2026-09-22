"""Offline market-state -> news-risk -> strategy example, using synthetic data only."""
import json

import numpy as np
import pandas as pd

from fin_skills.algorithms import recommend_strategy, research


def main():
    rng = np.random.default_rng(21)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(.003, .004, 120))),
                       index=pd.date_range("2026-01-01", periods=120, tz="UTC"))
    now = prices.index[-1].isoformat()
    news = [{"id": "synthetic-1", "source": "synthetic_fixture", "kind": "news",
             "title": "EXAMPLE: Company X trading halt", "url": "https://example.com/fixture",
             "published_at": now, "observed_at": now}]
    report = recommend_strategy(prices, as_of=now, news=news, news_keywords=["Company X"])
    print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))
    study = research("signal", {"prices": prices}, initial_train=70, horizon=10, holdout=20,
                     candidates=["donchian_breakout", "macd", "bollinger_reversion"], cost_bps=5)
    print("Temporal study:", study.status, "selected:", study.selected)
    print("TAKEAWAY")
    print("Synthetic example only: market-state rules propose candidates, not proven winners.")
    print("News risk is screened using only records observed by the decision timestamp.")
    print("Each strategy can be evaluated separately with temporal holdout and costs.")


if __name__ == "__main__":
    main()
