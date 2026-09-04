# Dataframe engines

Which engine to reach for, and the two packaging/semantics changes that landed in 2025–26 and break
existing code silently: **polars became an empty shim** and **pandas 3.0 changed what a timestamp is**.

All rows ✅ verified against the PyPI JSON API and GitHub API, probed 2026-09-04.

| pip | Version | Released | Stars | Licence | Python | Wheels |
|---|---|---|---|---|---|---|
| `polars` | 1.44.1 | 2026-08-26 | 39,646 | MIT | ≥3.10 | **1, `py3-none-any`** 🚨 |
| `polars` (next) | **2.0.0rc1** | **2026-09-02** | — | MIT | ≥3.10 | `py3-none-any` |
| `polars-runtime-32` | 1.44.1 | 2026-08-26 | — | MIT | ≥3.10 | 9, `cp310-abi3` |
| `pandas` | 3.0.5 | 2026-07-22 | 49,635 | BSD-3-Clause | **≥3.11** | 42 |
| `pyarrow` | 25.0.1 | 2026-08-10 | 17,080 | Apache-2.0 | ≥3.10 | 43 |
| `duckdb` | 1.5.5 | 2026-07-22 | 40,993 | MIT | ≥3.10 | 35 |
| `ibis-framework` | 12.0.0 | 2026-02-07 | 6,653 | Apache-2.0 | ≥3.10 | 1, pure |
| `dask` | 2026.8.0 | 2026-08-24 | 13,910 | BSD-3-Clause | ≥3.10 | 1, pure |
| `ray` | 2.58.0 | 2026-08-23 | 43,700 | Apache-2.0 | ≥3.10 | 20, **no musllinux** |

## 🚨 Trap 1 — `polars` is an empty 865 KB shim

✅ Verified from `pypi.org/pypi/polars/json`. `polars-1.44.1-py3-none-any.whl` is **865,208 bytes** and
contains **no compiled code**. Its `Requires-Dist` opens with an unconditional hard pin:

```
polars-runtime-32==1.44.1                       # mandatory, NOT an extra
polars-runtime-64==1.44.1 ; extra == "rt64"
polars-runtime-compat==1.44.1 ; extra == "rtcompat"
```

The engine lives in `polars-runtime-32` — ✅ 9 files at 1.44.1, MIT, `cp310-abi3` wheels for macOS
x86_64/arm64, manylinux + **musllinux** x86_64/aarch64, win_amd64/arm64, 47–54 MB each. `cp310-abi3`
means one wheel covers Python 3.10 through 3.14+. (`ray`, not polars, is the one with no musl wheel.)

✅ **The split landed at 1.34.0b2 (2025-09-26)** — a 5,966-byte wheel, where 1.34.0b1 (2025-09-23) still
shipped ~40 MB binaries. Everything from 1.34 on is split. ✅ Corroborating: `polars-lts-cpu` is frozen
at **1.33.1 (2025-09-09)**, the version immediately before the split — strong circumstantial evidence
that `rtcompat` replaced it ⚠️ (no changelog says so).

**Two real consequences:**

1. **Air-gapped and vendored installs break.** `pip download polars` fetches the 865 KB shim. You must
   also vendor `polars-runtime-32` **for your exact platform** at the identical version. The `==` pin
   at least makes skew a hard resolution failure, not a misbehaviour.
2. 🚨 **`py3-none-any` fools platform-audit tooling.** Any script that flags "packages with no wheel for
   our platform" now green-lights polars as pure Python while the real 50 MB binary goes unaudited. If
   you keep an install matrix, polars' row must point at `polars-runtime-32`.

✅ **`polars` 2.0.0rc1 shipped 2026-09-02.** ❓ Its breaking changes are unreviewed here. **Pin
`polars<2`** until you have read the migration guide.

## 🚨 Trap 2 — pandas 3.0 changed what a timestamp is

pandas **3.0.0 released 2026-01-21**, current **3.0.5**, requires **Python ≥3.11**. All items below ✅
verified against upstream `doc/source/whatsnew/v3.0.0.rst`.

### (a) `datetime64[us]` is the new default — integers come out 1000× smaller

| Input | 2.x | 3.0 |
|---|---|---|
| `pd.to_datetime(["2024-03-22 11:36"])` · `pd.Series([stdlib_datetime])` | `datetime64[ns]` | **`datetime64[us]`** |
| `pd.to_datetime([0], unit="s")` | `datetime64[ns]` | **`datetime64[s]`** |
| string with 9 decimal places | `datetime64[ns]` | `datetime64[ns]` (falls back) |

String parsing now defaults to microseconds, **explicitly including `read_csv` and `read_json`**.
Upstream's own warning: *"One big exception is converting to integers, which will give integers 1000x
smaller."*

🚨 **Blast radius:**
- `df["ts"].astype("int64")` as an epoch-nanos join key now yields **epoch micros**. Keys built in one
  process and consumed in another — or persisted to Parquet last quarter — mismatch by 1000×, and an
  as-of join on them returns nothing, or matches at absurd distances that `tolerance` then silently
  converts to NaN.
- CSV tick loaders that assumed `[ns]` now **truncate sub-microsecond exchange timestamps** (ITCH, some
  FIX feeds) lossily on read.
- **Remedy (upstream's own):** never `astype("int64")`; pin the unit first with `Series.dt.as_unit("ns")`.

### (b) `SettingWithCopyWarning` is removed — chained assignment is a silent no-op

Copy-on-Write is mandatory. Every indexing operation *"**always** behaves as a copy"*, chained
assignment *"will stop working"*, and **`SettingWithCopyWarning` is removed** (`mode.copy_on_write`
*"no longer has any impact"*). 🚨 **Worse than it sounds for a backtester.** In 2.x,
`df[df.symbol=="AAPL"]["signal"] = 1` **warned** and did nothing. In 3.0 it **does nothing, silently**.
A no-op in signal construction gives an all-zero signal column and a flat, entirely plausible equity
curve. ✅ Upstream's advice: **upgrade to 2.3 first** to collect the deprecation warnings before jumping.

### (c) pytz is gone

`Timestamp.tz_localize("US/Pacific").tz` now returns `zoneinfo.ZoneInfo`, not a pytz `DstTzInfo`, and
ambiguous/nonexistent times raise **`ValueError`**, not `pytz.AmbiguousTimeError`. Two breakages:
🚨 `except pytz.exceptions.AmbiguousTimeError` blocks around exchange-calendar DST handling **stop
catching**, and 🚨 **`pytz` is no longer installed with pandas** — `import pytz` in your own code
`ImportError`s on a fresh env. Use `pandas[timezone]` or pin `pytz` yourself.

### (d) `M` / `Q` / `Y` aliases removed — `df.resample("M")` raises

Removed, not deprecated: `M`/`Q`/`Y`/`BM`/`SM`/`CBM`/`BQ`/`BY` → `ME`/`QE`/`YE`/`BME`/`SME`/`CBME`/
`BQE`/`BYE`. Also removed: `kind` and `axis` on `resample`, `Resampler.fillna`, `include_groups`.
✅ **The label/closed defaults did NOT change** — month/quarter/year/week-end freqs are
`closed="right", label="right"`; **everything intraday is `closed="left", label="left"`**. That
asymmetry is the actual look-ahead trap and it predates 3.0; see `asof-joins.md`.

### (e) Three more silent numeric changes

- **String dtype by default (PDEP-14).** `pd.Series(["a","b"]).dtype` is `str`, not `object`.
  🚨 `select_dtypes(include="object")` **stops selecting your symbol column**; stuffing a sentinel
  `-1` or `None` into a ticker column now **raises**; the NA sentinel is `np.nan`, **not `pd.NA`**.
- **`offsets.Day` is a calendar day, not 24 h** (GH#61985). `t + Day(1)` on a tz-aware timestamp
  **may now raise** on DST boundaries where it previously returned a wrong-but-quiet answer — a daily
  rebalance loop throws twice a year.
- **NaN == NA in nullable dtypes.** `pnl / capital` with zero capital used to give a `NaN` that
  survived `.dropna()` and counted in `.count()`. It now vanishes. 🚨 **Sharpe denominators computed
  with `.std()` change value between 2.x and 3.0** — silent drift in reported performance, not an error.

## Positioning — which engine for what

⚠️ Architectural characterisation from documentation and design, **not benchmarked**.

| Engine | Memory model | Lazy | Out-of-core | Use it when |
|---|---|---|---|---|
| **pandas** 3.x | NumPy blocks (+ Arrow opt-in), CoW | eager only | ❌ | Default for anything that fits in RAM. Widest ecosystem. `pd.col()` in 3.0 is "initial support" for expressions, ✅ **not** a lazy engine |
| **polars** | Arrow columnar, own pool | ✅ `LazyFrame` + optimiser | ✅ streaming | Best single-node default for large panels. Multi-threaded Rust, **releases the GIL**. ❓ streaming coverage of *all* ops at 1.44 unverified |
| **duckdb** | vectorised ~2048-row chunks | ✅ SQL planner | ✅ **spills to disk** ⚠️ | When the access pattern is SQL-shaped, and for `ASOF JOIN` — see `asof-joins.md` |
| **pyarrow** · **ibis** | Arrow, immutable · none (a **compiler**) | pushdown · by construction | ✅ · inherits | The interchange layer, not compute · one expression API across backends |
| **dask** | partitioned pandas | ✅ task graph | ✅ | Only if you actually have a cluster; scheduler overhead is real at billions of rows |
| **ray** | `ray.data` Arrow blocks | ✅ | ✅ object-store spill | Distributed compute wholesale. ✅ **no musllinux wheels** — Alpine containers cannot `pip install ray` |

**Avoid:** ⚠️ `vaex` — dormant (4.19.0 on 2026-02-03, repo push 2026-04-01, `requires_python` **unset**);
its mmap niche was taken by polars streaming and DuckDB. ⚠️ `modin` — 0.37.1 on **2025-10-02**, and
running it on top of pandas 3.0's CoW/string-dtype semantics is ❓ an unverified combination.

## Sizing reality — the 10y × 3000-ticker minute panel

✅ Arithmetic: 390 min × 252 d × 10 y = **982,800 bars/ticker** → **≈2.95 B rows**. At
`{ts int64, sym int32, ohlc float64, vol int64}` = 52 B → **≈153 GB raw**; ~20–40 GB Parquet+ZSTD ⚠️.
**It does not fit in RAM in any engine.** Loop one symbol (~51 MB) or one day (~61 MB) at a time, or
`scan_parquet` with predicate pushdown. 🚨 **The `pivot`/`unstack` bomb:** a wide close-price matrix
alone is 982,800 × 3000 × 8 B = **23.6 GB for one field** ✅, all four OHLC ≈ 94 GB — and it is
**dense**, so every minute a ticker did not trade costs a full 8-byte NaN. If you need wide, do it
**per field, per year, in float32**.

## Minimal correct setup

```python
import pandas as pd, polars as pl

# pandas 3.0: parsing defaults to datetime64[us]. Pin the unit BEFORE any int cast,
# or your epoch key is 1000x off and every as-of join silently misses.
df = pd.read_csv("ticks.csv", parse_dates=["ts"])
df["ts"] = df["ts"].dt.as_unit("ns")          # never df["ts"].astype("int64") directly
df = df.resample("5min", on="ts", label="left", closed="left").last()  # "5T"/"M" now raise

# Chained assignment is a SILENT no-op in 3.0 (SettingWithCopyWarning was removed).
df.loc[df["symbol"] == "AAPL", "signal"] = 1  # .loc, always

# polars: the wheel is an 865 KB shim; polars-runtime-32 must be vendored too.
lf = pl.scan_parquet("panel/**/*.parquet").filter(pl.col("symbol") == "AAPL")
out = lf.collect()                             # pin polars<2 until 2.0 is reviewed
```

## Related

- `asof-joins.md` — the resample label asymmetry above becomes look-ahead the moment you join.
- `storage-formats.md` — float32 volume, float64 nanoseconds, and Parquet's dropped timezone.
- `timeseries-stores.md` — when a dataframe engine is no longer the right layer.
