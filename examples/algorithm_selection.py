"""Offline examples: task routing, execution schedules and chronological model selection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fin_skills.algorithms import auto_run, catalog, walk_forward


def main():
    returns = pd.DataFrame(np.random.default_rng(7).normal(size=(250, 3)) * [0.01, 0.02, 0.03],
                           columns=["A", "B", "C"])
    allocation = auto_run("portfolio", {"asset_returns": returns}, objective="min_variance")
    print("Allocation:", allocation["selection"]["selected"])
    print(allocation["result"].to_string())

    schedule = auto_run("execution", {"shares": 1000, "volume_forecast": [1, 2, 1]})
    print("Execution schedule:", schedule["selection"]["selected"], schedule["result"].tolist())

    comparison = walk_forward(np.arange(80.), initial_train=40, horizon=5, gap=2)
    print("Forecast selected by validation MAE:", comparison["selected"])
    print("Next forecast:", comparison["forecast"])
    print("Algorithm/backend entries:", len(catalog()))
    print("TAKEAWAY")
    print("Choose an allocation method using an explicit objective and decimal returns.")
    print("Execution algorithms return schedules; no orders are placed.")
    print("Forecast validation selects a method; reserve later data for an untouched test.")


if __name__ == "__main__":
    main()
