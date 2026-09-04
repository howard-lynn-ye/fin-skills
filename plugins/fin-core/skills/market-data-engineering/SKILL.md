---
name: market-data-engineering
description: >-
  Store, join and parallelize market data without silently corrupting it. Covers pandas 3.x, polars,
  pyarrow, duckdb, dask, ray and ibis; time-series stores (ArcticDB, QuestDB, ClickHouse, TimescaleDB,
  kdb/pykx, Parquet layouts); as-of joins in pandas, polars, DuckDB, QuestDB and ClickHouse; float and
  timestamp precision; timezone round-tripping; partitioning; and reproducible immutable snapshots.
  TRIGGER — use when joining quotes to trades or signals to prices, when writing or reading Parquet,
  Feather, HDF5 or CSV of market data, when choosing a dataframe engine or time-series database, when
  a panel is too big for memory, when parallelizing a walk-forward backtest, when timestamps or
  timezones come back wrong, or when two runs of the same pipeline disagree.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# Market data engineering

The layer under every backtest, and the one where corruption is silent. Most of what follows was
**executed**, not read from docs.

## 1. The findings that will change your code

| # | Finding |
|---|---|
| 1 | 🚨 **`polars` is now an empty shim** — the wheel is `py3-none-any`, ~865 KB, **no compiled code**. It hard-depends on `polars-runtime-32==<same version>` (~50 MB, `cp310-abi3`). Split landed at 1.34.0b2 (2025-09-26). **A lockfile listing only `polars` does not pin the engine.** `polars-lts-cpu` froze at 1.33.1, the release before. **2.0.0rc1 shipped 2026-09-02.** |
| 2 | 🚨 **`polars.join_asof` does NOT check sortedness when you pass `by=`** — its own docstring: the in-memory engine *"cannot check the sortedness if 'by' groups are provided"*. Unsorted input → **silently wrong rows, no error**. That is the exact per-symbol quote join everyone writes |
| 3 | ✅ **`pandas.merge_asof` DOES raise** `ValueError: left keys must be sorted`. **The "pandas silently gives garbage" folklore is out of date** — polars is now the unsafe one |
| 4 | 🚨 **pandas and polars disagree on what "sorted" means.** pandas wants a **global** sort by `on`; polars wants sorted **within each `by` group**. Porting code between them silently changes results |
| 5 | 🚨 **pandas 3.0 (2026-01-21) defaults to `datetime64[us]`, not `[ns]`.** Any `.astype("int64")` on timestamps now yields integers **1000× smaller** |
| 6 | 🚨 **float64 cannot hold epoch nanoseconds.** At 2024 timestamps the float64 step is **256 ns** — two ticks 100 ns apart collide. Epoch *microseconds* are exact until year 2255 |
| 7 | 🚨 **float32 breaks integer exactness above 2²⁴ = 16,777,216** — *below* the daily share volume of any liquid large-cap. `16,777,217 → 16,777,216`, but 20,000,000 is fine, so it is **inconsistent and hard to detect** |
| 8 | 🚨 **Parquet does not store timezones — by design.** The spec: *"time zone information gets lost… we can only reconstruct the instant, but not the original representation."* Your pandas round-trip works only via Arrow's pandas-metadata blob; **any other engine reads UTC** |
| 9 | 🚨 **CSV converts `America/New_York` to a fixed `UTC-05:00`** — instants survive, **DST awareness is destroyed**, so session-local calculations are an hour wrong for half the year |
| 10 | 🚨 **ArcticDB's two primary sources contradict each other on production use** — see §5 |
| 11 | ⚠️ **`pykx` is `License :: Other/Proprietary`**, and its GitHub repo has **78 stars** — three orders of magnitude below every OSS alternative here |

## 2. 🚨 The as-of join — where look-ahead actually enters

This is the single most important operation in market-data engineering, and its default is wrong for
execution modelling.

✅ **Executed.** A signal fires at `09:30:01`; the quote table has a quote stamped at exactly `09:30:01`.

```python
pd.merge_asof(signal, quotes, on="time")
#   time                 sig    bid
#   2024-01-02 09:30:01    1  101.0    <-- the 09:30:01 quote. LOOK-AHEAD.

pd.merge_asof(signal, quotes, on="time", allow_exact_matches=False)
#   2024-01-02 09:30:01    1  100.0    <-- the strictly prior quote. CORRECT.
```

**Why the tie happens in practice, and why it matters more than one cent:**
- **Bar data.** Your signal comes from the `09:30` bar and you join it to that bar's close — you have
  traded at the close you used to decide. Compounded by `resample`'s intraday **`label="left"`**
  default (unchanged in pandas 3.0, verified in source), **a 5-minute bar labelled `09:30` contains
  data through `09:34:59`** — so the strategy sees up to five minutes of future.
- **Same-timestamp quote revisions.** Feeds routinely stamp the quote update *caused by your trade*
  with the same microsecond. Matching it means you filled at the post-impact price you created.
- **Daily fundamentals.** Joining an earnings release stamped `2024-02-01` to a signal on
  `2024-02-01` assumes you traded on news released that day, often before the open.

🚨 **The tell is that the equity curve looks good but not absurd** — Sharpe 2 to 4 rather than 40 —
**which is exactly the range that survives review.**

**Rule:** when the right-hand table is *information you are reacting to*, use
`allow_exact_matches=False`. Reserve `True` for reference data genuinely known beforehand (a static
sector map, a prior-day close).

**Per-engine semantics and the sortedness asymmetry:** `references/asof-joins.md`.

## 3. pandas 3.x — what broke

Beyond the `datetime64[us]` default (§1):
- **Copy-on-Write is mandatory**, and 🚨 **`SettingWithCopyWarning` was removed** — **chained
  assignment is now a silent no-op.** Code that "worked" by mutating a view now does nothing, quietly.
- **`zoneinfo` replaces `pytz`**, which is **no longer installed**. `import pytz` fails.
- **`offsets.Day` is now a calendar day**, not 24 hours — it matters across DST.
- **`M`, `Q`, `Y` frequency aliases were removed outright** (use `ME`, `QE`, `YE`).

## 4. Precision — what silently rounds

| Column | Safe type | Why |
|---|---|---|
| Price | `float64`, or **int cents / `Decimal`** for exactness | float64 is fine for prices; the danger is aggregation over millions of rows |
| **Volume** | 🚨 **`int64` — never `float32`** | float32 loses integer exactness above **16,777,216**, below any large-cap's daily volume |
| **Nanosecond timestamps** | 🚨 **`int64` — never float** | float64's step is **256 ns** at 2024 dates |
| Microsecond timestamps | float64 tolerable (exact to 2255) | pandas 3.0's `[us]` default accidentally makes this safe |

**Never let a nanosecond timestamp touch a float column.** If you must serialize timestamps as
numbers, use `int64` epoch nanos or epoch micros — and record which.

## 5. 🚨 ArcticDB's licence contradicts itself

- `LICENSE.txt`'s Additional Use Grant carves out **only** a multi-tenant *"Database Service"*.
- `README.md` says you may not use it *"for production use **or** for a Database Service"*.

**These are different claims and I could not resolve which controls.** Get it in writing from the
vendor before depending on it; this is a flag, not legal advice. Conversion to Apache-2.0 is **two
years per version**, and the published table stops at 6.21 while current is **6.24.0** — so waiting
for conversion pins you permanently two years behind.

🚨 **ArcticDB's `as_of` is version-as-of, not join-as-of** — verified in source. Two different things
share the name. **LanceDB and Delta Lake "time travel" are the same category error.** No single tool
does both; the honest pattern is **ArcticDB (data vintages) + DuckDB/polars (temporal joins)**.

## 6. Storage format

| Need | Use |
|---|---|
| Panel storage, cross-engine | **Parquet** — but see the timezone caveat |
| Fast local analytics over Parquet | **DuckDB** |
| Versioned vintages (point-in-time data) | **ArcticDB** (licence caveat) |
| Tick ingest at scale, native as-of | **QuestDB** — the cleanest as-of design of any engine here |
| Never | **pickle** for anything you keep — it round-trips perfectly *and* pins you to one pandas version, executes arbitrary code on load, and is unreadable by anything else |

**Timezone rule:** store UTC as the instant and **carry the exchange timezone as a separate column or
in your own metadata.** Do not rely on the format to preserve it.

Details, partitioning schemes and schema-evolution traps: `references/storage-formats.md`.

## 7. Parallelising a walk-forward without leaking

The five leak sources are ranked in `references/storage-formats.md`, but the key insight is one
sentence:

🔑 **"Different numbers when parallelised" is a pre-existing leak that the parallelism exposed, not a
parallelism bug.**

**The one test that catches all five: assert that serial and parallel results are identical.** If
they differ, you have shared mutable state — a scaler fitted outside the fold, a global cache, a
random seed, an accumulating feature store — and it was leaking before you parallelised.

## 8. Reproducibility

A backtest you cannot re-run bit-for-bit is an anecdote. The minimum:
- **Hash the input data**, not just its path. Content-addressed snapshots.
- **Record the retrieval timestamp** and the vendor's own version/vintage where it has one.
- **Pin the engine, not just the wrapper** — see the `polars` shim in §1.
- Cache at the **HTTP layer** (`requests-cache`), the **domain layer** (`yfinance-cache`, which
  understands market calendars and only re-fetches genuinely new bars), and the **artifact layer**
  (an immutable hashed snapshot). They solve different problems.

## ❓ Not verified

**No benchmarks were run** — polars, duckdb and pyarrow were not installed on the research machine,
so **every throughput and latency claim in the reference files is marked ⚠️ or ❓.** Treat the
performance guidance as directional and measure on your own data.
