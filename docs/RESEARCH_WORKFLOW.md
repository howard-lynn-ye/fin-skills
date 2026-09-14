# Algorithm research workflow

The Python entry point is `fin_skills.algorithms.research`. It connects input checks,
candidate selection, chronological evaluation, the existing trial ledger and audit guards.
The JSON/MCP equivalent is `research_algorithms`; it returns a record without writing files.

## Install and run

From a checkout:

```bash
python -m pip install -e ".[stats]"
python examples/research_workflow.py
```

Core forecasting, risk, signals and native allocation require only the base dependencies.
Use `.[algorithms]` for the additional forecasting, boosting, volatility, regime and pricing
backends, `.[qlib]` for Qlib, and `.[optimizers]` for optimizer bridges. The legacy `.[all]`
extra retains its previous meaning; it does not install every third-party algorithm library.
`fin-skills-doctor --task forecast` checks each relevant optional import in a fresh process.
An importable library can still fail on a particular input or solver problem.

## A complete forecast study

```python
import numpy as np
from fin_skills.algorithms import research, fit, load_model

series = np.arange(100.)  # substitute your chronological numeric observations
report = research(
    "forecast", {"series": series},
    initial_train=40, horizon=10, gap=1, holdout=20,
    candidates=("naive", "mean", "drift"),
    ledger_path="trials.jsonl",
)
print(report.render())
report.save("research.json")

# Fit for subsequent use separately, after assessing the study.
model = fit(report.selected, {"series": series})
model.save("model.zip")
restored = load_model("model.zip", trusted=True)
prediction = restored.predict(horizon=5)
```

Paths must not already exist when saving a report/model. The ledger is append-only.
Model files contain Python pickle: `trusted=True` is appropriate only for artifacts whose
origin you trust. A checksum detects corruption, not malicious code. Exact environment
matching is the default; reinstall recorded versions before loading an older artifact.
Research JSON is data-only, carries `schema_version=1`, and requires no pickle loader.

`fit` supports forecasting, regression and classification. Supervised fitting accepts `X`
and `y`, followed by `model.predict(X_new)`; transforms fit only on the training data.
Named DataFrame features must retain the same names and order at prediction time. Models
preserve fitted state, so repeated prediction does not refit or incorporate other test rows.

## Time, units and selection

- Rows are chronological. Pandas indexes must be unique and increasing, and training indexes
  must match exactly. Plain arrays assert positional chronology but cannot prove timestamps.
- Returns are simple decimal returns per row. The existing input policy rejects values below
  -1 and absolute per-period column means above 0.10. Clean missing values explicitly.
- Every reported interval end is exclusive. A validation fold fits `[0, train_end)` and
  evaluates `[train_end + gap, train_end + gap + horizon)`.
- The final `holdout` rows never enter candidate ranking. The selected candidate is refit
  through `development_end - gap`, then evaluated once on the holdout. It is not reselected
  after seeing the holdout. Reusing that period to tune parameters makes it development data.
- Set `gap` to cover label availability/overlap for your task. Ordered rows alone cannot prove
  point-in-time features, correct corporate actions or absence of survivorship bias.
- `parameters` maps algorithm IDs to dictionaries, e.g. `{"hrp": {"linkage": "single"}}`.
  A candidate that fails any validation fold is excluded, with the error retained.
- Without `initial_train`, the workflow uses suitability rules. A runtime failure is recorded
  as `failed`; it does not silently replace the selected algorithm.

| Task | Selection loss (lower is better) | Holdout output |
|---|---|---|
| forecast / regression | MAE or mean per-fold RMSE | Predictions and actual values |
| classification | Error rate | Predicted and actual integer classes |
| portfolio | Per-fold variance or negative mean net return | Weights, net returns, turnover |
| volatility | QLIKE using squared returns as a noisy variance proxy | Variance-derived forecast |
| risk | VaR quantile loss | VaR/ES and observed violation rate; ES is not separately scored |
| signal | Negative mean net return | Lagged positions and net returns |

Fold scores are averaged equally; folds have equal horizons. Forecasts span the gap as well
as the scored rows. Volatility and risk estimates stay fixed over each validation block.
Pricing, execution schedules, cointegration and retrospective regime detection support
execution with an audit record; they do not have a temporal selection loss in this workflow.

Portfolio evaluation rebalances to the fitted target each row and charges proportional
`cost_bps` on traded notional, including entry from cash and terminal liquidation. Signal
evaluation charges changes in unit exposure and exit. These are reference evaluations;
neither includes impact, borrow availability, financing, exchange calendars or order fills.
They must not be presented as an executable FX/derivatives trading simulation.

Signal adapters return positions already lagged by one bar. This workflow uses those
positions directly. The separate reference engine applies its own signal lag: do not feed
these positions into its decision interface without accounting for the extra delay.
HMM states and change points are retrospective outputs, never historical trading positions.

## Audit and provenance

Pass a `fin_skills.data.provenance.Provenance` as `provenance`. The result stores it alongside
an independent hash of the exact algorithm input, including array contents, index and feature
order. Source provenance may describe raw data before transformation; the input hash identifies
what this algorithm actually received. Provenance is recorded, not independently certified.

Pass `audit_bundle=Bundle(...)` for the existing staged `reality_check`, or specify `guards`
to run named guards. For portfolio/signal holdouts, evaluated net returns feed the audit
automatically. Missing inputs remain `partial`/`not_evaluated`; an empty audit never becomes
`passed`. An audit failure changes the research status to `audit_failed`.

The record contains candidate reasons, rejected candidates, fold boundaries, parameters,
library versions, trial IDs, test results, timing and audit findings. It does not fabricate
the universe, cost curve or factor benchmark required by a full strategy `ResultCard`.
That existing card remains the publication interface once those inputs are available.

## Failure examples

```python
from fin_skills.algorithms import auto_run
import numpy as np

# Variance-dependent methods are excluded; equal weight remains executable.
result = auto_run("portfolio", {"asset_returns": np.zeros((60, 3))})
assert result["selection"]["selected"] == "equal_weight"

# A hard objective can leave no valid algorithm. The reasons remain available.
result = auto_run("portfolio", {"asset_returns": np.zeros((60, 3))},
                  objective="min_variance")
assert not result["executed"]
print(result["selection"]["rejected"])
```

`recommend(Request(...))` without actual data is metadata-only advice. Supply `data` and
`parameters` for the shared preflight, or use `auto_run`/`research` to derive the input facts.
Validation errors and unknown fields raise `ValueError`/`TypeError`. `run` propagates backend
errors; `research` records candidate/execution errors in its result. Code exceptions, a failed
audit and missing audit evidence are distinct outcomes.
