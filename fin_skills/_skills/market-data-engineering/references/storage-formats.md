# Storage, precision, partitioning and parallelism

## 1. 🚨 Timezones do not survive serialization

**Parquet does not store timezones — by design.** The spec says: *"time zone information gets lost…
we can only reconstruct the instant, but not the original representation."*

Your pandas → Parquet → pandas round-trip appears to preserve `America/New_York` only because Arrow
writes a **pandas-metadata blob** alongside. **Any other engine — DuckDB, Spark, ClickHouse, a
different language — reads UTC.**

✅ **CSV is worse, executed:** `America/New_York` round-trips to a **fixed `UTC-05:00` offset**. The
instants survive, but **DST awareness is destroyed**, so every subsequent session-local calculation is
an hour wrong for half the year — and it is wrong in a way that looks like data, not like an error.

**The rule:** store **UTC as the instant**, and carry the exchange timezone as a **separate column or
in your own manifest**. Never rely on the format to preserve it.

## 2. 🚨 Precision — what silently rounds

| Column | Safe type | The limit |
|---|---|---|
| Price | `float64`, or int-cents / `Decimal` if exactness is contractual | fine for prices; the risk is aggregation over millions of rows |
| **Volume** | 🚨 **`int64` — never `float32`** | float32 loses integer exactness above **2²⁴ = 16,777,216** |
| **Nanosecond timestamps** | 🚨 **`int64` — never float** | float64's step is **256 ns** at 2024 dates |
| Microsecond timestamps | float64 tolerable | exact below 2⁵³, i.e. until year **2255** |

✅ **Executed:** `16,777,217` stored as float32 becomes `16,777,216`. **But 20,000,000 is exact** —
so the corruption is **inconsistent and therefore hard to detect**. Daily volume for any liquid
large-cap sits right in the broken range.

✅ **Executed:** at 2024 epoch nanoseconds the float64 representable step is **256 ns**, so two ticks
100 ns apart collide into one value. pandas 3.0's `[us]` default accidentally makes float
round-tripping safe — but do not rely on an accident.

## 3. Format comparison

| Format | Use it for | Against it |
|---|---|---|
| **Parquet** | The default for panels. Columnar, compressed, predicate pushdown, cross-engine | 🚨 no timezone; schema evolution is a live hazard (§5) |
| **Feather / Arrow IPC** | Fast local handoff between processes | Larger on disk; less universally read |
| **HDF5** | Legacy; hierarchical numeric data | Concurrency is painful; the ecosystem has moved on |
| **CSV** | Interchange with humans and spreadsheets | 🚨 destroys timezones; no types; slow |
| **pickle** | 🚨 **nothing you intend to keep** | Round-trips perfectly *and* pins you to one pandas version, **executes arbitrary code on load**, unreadable by anything else. "It round-trips perfectly" is not a reason to use it |

## 4. 🚨 Partitioning — the small-file problem kills projects

| Scheme | Good for | Bad for |
|---|---|---|
| **By date** (`dt=2024-01-02/`) | Cross-sectional queries, daily appends, backfills | Single-symbol full history — touches every partition |
| **By symbol** (`symbol=AAPL/`) | Per-symbol time series (the backtest loop) | Cross-sectional queries; **3000 tiny files per day** if you also append daily |
| **Both** (`dt=…/symbol=…`) | Neither, really | 🚨 **2520 days × 3000 symbols = 7.56 M files** |
| ✅ **By date, sorted by symbol within file** | **Both**, via row-group statistics | Requires you to actually sort before writing |

🚨 At even 4 KB of footer/schema overhead, 7.56 M files is **~30 GB of pure metadata**, and listing
the dataset on object storage takes minutes before a single byte of data is read. **Aim for
128 MB – 1 GB per file.**

✅ **The recommendation:** partition by **date** (or `year=`/`month=` for coarser granularity) and
**sort by `symbol, timestamp` within each file**. Parquet stores per-row-group min/max statistics, so
a sorted symbol column lets `WHERE symbol='AAPL'` skip almost every row group via predicate
pushdown — **symbol-partition performance without symbol partitions.**
⚠️ Row-group pruning is standard reader behaviour; the skip rate was not measured.

## 5. 🚨 Schema evolution — the failure that eats a weekend

You have `dt=2024-01-01/` … `dt=2024-06-30/`, all with `volume: int64`. On July 1 the vendor starts
sending nulls, your loader infers `float64`, and writes a partition with **a different type for the
same column**. Then:

- `pd.read_parquet("dataset/")` or `pyarrow.dataset` unifies across fragments and **either raises** a
  type-unification error **or silently promotes the whole column to `float64`** — reintroducing the
  float-volume bug from §2.
- An engine that reads only the *first* fragment's schema (some do) reads July's float bytes as int64
  and produces **garbage numbers with no error at all.**

Equally silent relatives:
- **Column added mid-history** → old partitions read back all-`NULL`, which looks like missing data
  rather than a schema change.
- **Column renamed** → two half-populated columns.
- **Partition-key type drifts** (`dt` as string in some directories, date in others) → Hive discovery
  infers per-directory and filters silently miss.

**Defences:**
1. **Write an explicit `pyarrow.Schema` on every write.** Never let the loader infer.
2. **Validate before writing** — assert incoming dtypes equal the stored schema and fail loudly.
3. **`int64` volume with an explicit sentinel** or a separate null mask, rather than letting nulls
   force a float promotion.
4. **Consider Delta Lake** (`deltalake` 1.6.3) if schema enforcement matters — it enforces schema on
   write and makes evolution an explicit versioned operation rather than an accident. **That is
   Delta's real value for market data, more than time travel.**

## 6. 🚨 Parallelising a walk-forward without leaking

Folds look independent, so the instinct is `Parallel(delayed(run_fold))(folds)`. **The leakage is not
in the parallelism — parallelism just makes pre-existing leakage non-deterministic and harder to spot.**

**The five leaks, in the order they actually happen:**

1. **Shared mutable state captured in the closure.** A scaler, encoder or feature cache created once
   outside the loop and `.fit()` inside each fold. Serially, fold *k* silently inherits fold *k−1*'s
   fitted parameters — **already a bug**. In parallel with a process pool each worker gets its own
   pickled copy, so results **change** versus serial.
   🔑 **The classic tell: *"my backtest gives different numbers when I parallelise it."* That is not a
   parallelism bug — it is a leak the parallelism exposed. Investigate it; do not "fix" it by going
   back to serial.**
2. **Fitting any transform on the full series.** Normalising with a mean/std over all history, then
   slicing folds, leaks the future into every training set. Every `fit` must happen **inside** the
   fold on training data only, then be applied unchanged to the test slice.
3. **A global cache keyed without the fold boundary.** `@lru_cache` on `get_features(symbol)` returns
   features computed with full-history knowledge. Key by `(symbol, fold_end_timestamp)` or compute
   inside the fold.
4. **Warm-started model state.** Reusing one model object so fold *k* starts from fold *k−1*'s
   weights. Construct a fresh model per fold.
5. 🚨 **Shared RNG — the subtle one.** A single global `np.random.seed(42)` gives *serial* folds a
   deterministic but **fold-order-dependent** stream. Run them in parallel and each worker inherits a
   copy of the same state, so **every fold draws identical random numbers** — a different bug.
   Neither is reproducible in the way you want. Pass an explicit
   `np.random.default_rng(seed + fold_index)` into each fold.

**The correct shape** is a pure function of `(fold_index, data_slice, config)` that constructs
everything it needs internally and returns only results.

**The one test that catches all five:**

```python
serial   = [run_fold(i, folds[i], cfg) for i in range(n)]
parallel = Parallel(n_jobs=-1)(delayed(run_fold)(i, folds[i], cfg) for i in range(n))
assert serial == parallel, "shared state between folds — a pre-existing leak"
```

## 7. Reproducible snapshots

A backtest you cannot re-run bit-for-bit is an anecdote.

- **Hash the content, not the path.** A content-addressed snapshot id makes "same data" checkable.
- **Record the retrieval timestamp** and the vendor's own vintage/version where one exists.
- 🚨 **Pin the engine, not just the wrapper.** `polars` is a shim over `polars-runtime-32`; a lockfile
  listing only `polars` does not pin the code that runs.
- **Three cache layers solve different problems:** HTTP (`requests-cache`) avoids refetching bytes;
  domain (`yfinance-cache` — it understands market calendars and only refetches genuinely new bars)
  avoids refetching *bars*; artifact (an immutable hashed snapshot) makes a *result* reproducible.
  Only the third one makes a backtest re-runnable.
