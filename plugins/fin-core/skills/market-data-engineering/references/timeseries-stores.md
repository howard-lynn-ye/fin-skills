# Time-series stores

Where market data lives once it outgrows Parquet files. The decisive question is **which of the two
things called "as of" you need** — and no single system provides both.

All rows ✅ verified against PyPI JSON + GitHub API, probed 2026-09-04.

| pip | Version | Released | Stars (server repo) | Licence | Python | Wheels |
|---|---|---|---|---|---|---|
| `arcticdb` | 6.24.0 | 2026-09-01 | 2,506 | 🚨 **BSL 1.1** | **unset** ⚠️ | 18 (cp39–cp314; linux x86_64, **macOS arm64 only**, win_amd64). **No sdist** |
| `questdb` (client) | 5.0.0 | 2026-07-27 | 17,299 | Apache-2.0 | ≥3.10 | 49 incl. win32/win_amd64/musl |
| `clickhouse-connect` | 1.8.0 | 2026-09-03 | 49,645 | Apache-2.0 | **≥3.10,<3.15** 🚨 | 60 |
| `psycopg` (→ TimescaleDB) | 3.3.5 | 2026-08-31 | — | ⚠️ **LGPL-3.0-only** | ≥3.10 | 1 pure (binary in `psycopg[binary]`) |
| `pykx` (→ kdb+) | 4.1.0 | 2026-08-19 | 🚨 **78** | 🚨 **Other/Proprietary** | ≥3.9 | 30. **No sdist** |
| `deltalake` | 1.6.3 | 2026-08-21 | 3,293 | Apache-2.0 | ≥3.10 | 8 (`cp310-abi3`) |
| `influxdb-client` / `influxdb3-python` · `lancedb` | 1.50.0 / 0.21.0 · 0.38.0 | 2026-01-23 / 08-27 · 08-31 | 31,731 · 11,352 | MIT · Apache-2.0 | ≥3.7 / ≥3.9 · ≥3.10 | pure · 4 abi3 |

## 🚨 Trap 1 — ArcticDB's LICENSE.txt and README contradict each other on production use

I read **both** primary sources on `man-group/ArcticDB@master`. They do not say the same thing.
Reported as found, not resolved.

**`LICENSE.txt` — ✅ verbatim:**

> **Additional Use Grant:** You may make use of the Licensed Work under the terms of this License,
> provided that you may not use the Licensed Work for a Database Service. A "Database Service" is a
> commercial offering that allows third parties … to access the functionality of the Licensed Work by
> creating tables whose schemas are controlled by such third parties.

Under standard BSL 1.1 the base grant is *non-production only*, and the Additional Use Grant is what
**expands** it (*"…permitting limited production use."* — ✅ same file). Read literally, this permits
everything **except** a multi-tenant Database Service.

**`README.md` — ✅ verbatim, and it says something else:**

> …users may not use ArcticDB **for production use or** for a Database Service, without agreement with
> Man Group Operations Limited.

🚨 **The conflict:** LICENSE.txt carves out *only* Database Service; the README additionally carves out
*production use*. For an in-house fund running ArcticDB on its own research cluster, the two readings
give **opposite answers**. ❓ Which controls is unresolved and this is not legal advice — if you intend
production use, get written confirmation from `info@arcticdb.io` before building on it.

**Conversion to Apache-2.0 is per-version and two years deep** ✅ — LICENSE.txt: *"This License applies
separately for each version … and the Change Date may vary for each version"*, with a BSL backstop at
the fourth anniversary, whichever is earlier. The README's table runs 1.0 → 2025-03-16 through
**6.21 → 2028-08-04**. 🚨 **Current 6.24.0 is not in the table at all** ✅, and conversion never lets you
stay current: by the time 6.21 turns Apache-2.0 in Aug 2028, ArcticDB will be ~9.x and still BSL.
"Wait for it to open-source" pins you permanently two years behind.

## 🚨 Trap 2 — ArcticDB's `as_of` is version-as-of, not join-as-of

The most confused point in the domain. Two completely different features share the name:
**(A) version as-of / vintage** — *"what did this dataset look like on 2024-03-15, before the
restatement?"*, which defends against **restatement and revision bias**; and **(B) temporal as-of
join** — *"what was the prevailing quote at the instant of this trade?"*, which defends against
**look-ahead bias**.

✅ Verified from `python/arcticdb/version_store/library.py` on master, `Library.read` — `as_of` accepts
*"int: specific version number … str: snapshot name … datetime.datetime: the version of the data that
existed as_of the requested point in time"*.

**So ArcticDB's `as_of` is (A).** It selects a *version of the symbol*, never a row-level temporal
match. Anyone saying "ArcticDB does as-of joins" has conflated the two.

| Store | (A) version as-of | (B) as-of join |
|---|---|---|
| ArcticDB | ✅ **native, excellent** (int / snapshot / datetime) | ❌ none |
| Delta Lake, LanceDB | ✅ time travel | ❌ |
| DuckDB · QuestDB | ❌ | ✅ native (`ASOF JOIN` · `ASOF`+`LT`+`SPLICE`) |
| ClickHouse | ❌ | ⚠️ `ASOF JOIN`, restricted |
| kdb+ · polars/pandas | ⚠️ by convention · ❌ | ✅ native `aj` · ✅ in-memory |

🚨 **An honest backtest needs both, and no single tool provides both.** The workable pattern is
**ArcticDB (or date-partitioned Parquet vintages) for (A) + DuckDB or polars for (B)**. Delta Lake and
LanceDB both advertise "versioning" and "time travel" and get recommended in market-data threads —
neither does an as-of *join*.

## 🚨 Trap 3 — ArcticDB's `sorted` field is a quiet correctness bug with a long fuse

✅ Same source file. ArcticDB tracks a per-symbol `sorted` status: `ASCENDING` *"guarantees that
operations such as append, update, and read with `date_range` work as expected"*; for `DESCENDING` and
`UNSORTED`, *"update and read with `date_range` **will not work**"*; `UNKNOWN` means no timestamp index.
`UNSORTED` *"can only be created by calling `write` … or `append_batch` with **`validate_index` set to
False**"*.

🚨 Someone sets `validate_index=False` to speed a slow bulk load. The write succeeds. Months later every
`date_range` read on that symbol returns wrong rows — and `date_range` is **exactly how you slice a
backtest window**. It records `UNSORTED` instead of raising at read time. **Never write market data
with `validate_index=False`; audit `sorted` on existing symbols before trusting them.**

Also ✅: `date_range` on `read()` is **inclusive at both ends**, and `(None, ts)` is allowed. Inclusive-
both-ends is unusual — a loop reading `[d, d+1day]` per day **double-counts the boundary timestamp**.

## QuestDB — the cleanest as-of design of the four

✅ Verified from `questdb/documentation@main` `documentation/query/sql/asof-join.md`. It is the only
system here with a **named join per exact-match semantic**: `ASOF JOIN` takes the most recent row
*"earlier than or equal to"* (exact matches allowed); `LT JOIN` *"behaves like `ASOF JOIN` but
**excludes rows whose timestamp matches exactly**"* ✅ verbatim; `SPLICE JOIN` is *"a full `ASOF JOIN`"*
preserving both sides. So the exact-match decision is a **keyword**, not a default you can forget.

Plus a first-class **`TOLERANCE`** clause ✅ — a left row at `t1.ts` joins `t2.ts` only if
`t2.ts <= t1.ts` **and** `t1.ts - t2.ts <= tolerance`. Units: `n` ns, `T` ms, `s`, `m`, `h`, `d`.

```sql
SELECT market_data.timestamp, market_data.symbol, bids, core_price.*
FROM market_data
LT JOIN core_price ON (symbol) TOLERANCE 50T;   -- strictly prior, max 50 ms stale
```

✅ Also documented: QuestDB joins a microsecond `TIMESTAMP` to a nanosecond `TIMESTAMP_NS` *"without
explicit casting — QuestDB aligns the timestamps internally"*, removing a whole class of unit-mismatch
bug that bites in pandas (`dataframe-engines.md` §Trap 2).

## ClickHouse — ASOF JOIN, but restricted

⚠️ From vendor docs; not executed here. Closest-match operators `>`, `>=`, `<`, `<=` are explicit in the
`ON` clause (good, like DuckDB); *"any number of equality conditions and **exactly one** closest match
condition"*.

🚨 **`ASOF JOIN` is supported only by the `hash` and `full_sorting_merge` join algorithms. It is not
supported in the `Join` table engine.** And for `hash`, the asof column *"can't be the only column in
the `JOIN` clause"* — you **must** supply at least one equality key (e.g. `symbol`). A pure time-only
as-of join therefore requires `full_sorting_merge`. Set the algorithm explicitly; do not let a server
default decide whether your join is even legal.

## 🚨 Trap 4 — pykx is proprietary with 78 stars

✅ `pykx` 4.1.0 (2026-08-19), PyPI classifier **`License :: Other/Proprietary License`**. Its `license`
field is itself a warning ✅ verbatim: *"All files contained within this repository are not covered by a
single license."* — mixed licensing inside one distribution. No sdist.

✅ **`KxSystems/pykx` has 78 GitHub stars** — three orders of magnitude below every alternative here
(polars: 39,646). kdb+ is genuinely the fastest thing in this space and the sell-side/HFT standard, but
the Python-side community is effectively nonexistent: **vendor support will be your only support.**
💵 ❓ KX publishes no price list; per-core figures in forums are unverified folklore — ask KX directly.

## Which is actually a tick store

⚠️ Architectural classification from docs and source, **not benchmarked**.

| Store | Tick? | Verdict |
|---|---|---|
| kdb+ / `pykx` | ✅ | The reference implementation. Licence and cost are the barrier |
| **QuestDB** | ✅ | Designated timestamp, native ASOF/LT/SPLICE, ILP ingest. Genuinely tick-shaped |
| ClickHouse | ⚠️ | An OLAP engine adapted to time series, not a tick database |
| **ArcticDB** | ⚠️ bar-optimal | Versioned dataframe storage over object storage. Superb for the 3000×10y panel; unnatural as a raw trade-by-trade landing zone |
| Parquet + Hive | ✅ batch | Not a database. The honest default for write-once historical tick |
| TimescaleDB | ⚠️ bars | Real SQL and transactions; you inherit Postgres' row-store write path. ⚠️ `psycopg` 3.x is **LGPL-3.0-only** — the one copyleft driver here |
| InfluxDB · LanceDB | ❌ | Observability lineage (v2→v3 split the client into **two PyPI packages** ✅) · a vector DB. Neither does a temporal join |
| Delta Lake | ⚠️ | Table format: ACID + time travel over Parquet. Reproducibility, not query latency |

## Minimal correct usage

```python
from datetime import datetime
import pandas as pd
from arcticdb import Arctic

ac = Arctic("lmdb:///data/arctic")
lib = ac.get_library("bars", create_if_missing=True)

# validate_index=True is the default -- NEVER turn it off for market data.
# UNSORTED symbols make every later date_range read return wrong rows, silently.
lib.write("AAPL", df, validate_index=True)

# as_of selects a VERSION (restatement defence), NOT a row-level temporal match.
old = lib.read("AAPL", as_of=datetime(2024, 3, 15)).data

# date_range is INCLUSIVE at BOTH ends -- per-day loops double-count the boundary.
win = lib.read("AAPL", date_range=(pd.Timestamp("2024-01-02"),
                                   pd.Timestamp("2024-01-02 23:59:59.999999999"))).data
```

## Related

- `asof-joins.md` — the (B) as-of join in full, across pandas/polars/DuckDB/QuestDB/ClickHouse.
- `storage-formats.md` — Parquet's dropped timezone and the float precision limits underneath all of this.
- `dataframe-engines.md` — pandas 3.0's `datetime64[us]` default, which changes what you write here.
