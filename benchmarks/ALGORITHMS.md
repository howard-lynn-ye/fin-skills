# Algorithm selection benchmark

Run `python benchmarks/algorithm_bench.py`. It is offline and checks the fixture SHA-256
before scoring. `ALGORITHM_RESULTS.json` contains every case, development ranking, held-out
score, naive baseline, timing, Python allocation peak and exact input hash.

The fixed design uses ECB USD, JPY, GBP and CHF quotes per euro, split into the calendar
blocks 2005-2011, 2012-2018 and 2019-2025. Within each block, the first half starts training,
the final fifth is held out, and intermediate expanding folds select among naive, mean and
drift. A one-row gap separates fitting and scoring. Calendar blocks expose different periods;
they are not inferred bull/bear labels, and the shared EUR base makes the series dependent.

Seeded synthetic stationary, drifting and variance-break cases supplement real observations.
The result reports wins, ties AND losses versus the naive baseline. No passing criterion
requires the selector to beat naive, and these holdouts must not be used to tune routing.
The benchmark is a reproducible forecast baseline panel, not an all-algorithm leaderboard.
Additional real-FX panels cover portfolio variance, VaR quantile loss, volatility QLIKE,
signal net-return loss, regression MAE and classification error. These panels have their own
baselines and units. Regression/classification features use only lagged USD/EUR changes;
training-mean/majority predictors are explicit local benchmark baselines. The signal panel
has a single candidate and measures execution only. Portfolio/signal costs are illustrative
proportional charges; their outputs are not achievable FX profits. Install `.[stats]` to run
the supervised panels. Keep each panel's losses separate when interpreting the report.

## Data terms and provenance

Source: European Central Bank. The original information is available free of charge from
https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip . The fixture selects four currency
columns and dates, sorts rows and serializes CSV; it does not adjust the reference-rate values.
See `data/ecb_fx.provenance.json` for retrieval time, source/fixture hashes and transformations.
ECB copyright/usage conditions are at
https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html .
The data remains subject to those terms and is not relicensed under this repository's MIT license.

Reference rates are informational, not executable bid/ask quotes. This fixture is unsuitable
for claiming realizable FX strategy profits. It is also a present-day historical snapshot,
not a point-in-time vintage database. Run `python benchmarks/fetch_ecb.py` only for an explicit
reviewed refresh; a refresh changes the benchmark input and its recorded hash.

`tracemalloc` measures Python-tracked allocations, not total native memory. Runtime varies
with hardware and dependency versions. Comparisons across incompatible units are not averaged
into a single MAE; the per-case ratio to naive is available for inspection.
