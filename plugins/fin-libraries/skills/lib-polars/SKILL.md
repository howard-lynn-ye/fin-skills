---
name: lib-polars
description: >-
  The polars wheel is now an empty 865 KB py3-none-any shim hard-pinned to polars-runtime-32, so a
  lockfile listing only polars does not pin the engine. TRIGGER - polars, "import polars as pl",
  LazyFrame, scan_parquet, collect(), pl.col, join_asof, group_by, with_columns,
  polars-runtime-32, polars-runtime-64, polars-lts-cpu, polars 2.0.0rc1, "pip download polars",
  vendored or air-gapped polars install, polars wheel has no compiled code, porting pandas
  merge_asof to polars.join_asof, polars sortedness. The runtime split landed at 1.34.0b2 on
  2025-09-26, so install matrices, wheel audits and lockfiles written from memory are wrong. SKIP
  for market-data-engineering, the skill for storage formats and time-series stores. SKIP when the
  question is WHICH library to choose rather than how to use this one - that belongs to the domain
  skill.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# polars

The best single-node dataframe default for large panels — multi-threaded Rust, Arrow columnar, a real
lazy optimiser, and it releases the GIL. It also changed its own packaging, and its as-of join is now
the unsafe one.

| | |
|---|---|
| pip / import | `polars` / `import polars as pl` — **plus `polars-runtime-32`, which is not optional** |
| Version | 1.44.1 (2026-08-26) · 432 releases · **2.0.0rc1 shipped 2026-09-02** |
| Licence | MIT (both `polars` and `polars-runtime-32`) |
| Python | `>=3.10`. `polars` ships **1 wheel, `py3-none-any`**; `polars-runtime-32` ships 9 `cp310-abi3` wheels |
| Status | 39,646★, very active. **Pin `polars<2`** until you have read the 2.0 migration guide |

## The trap that costs you money

**`polars-1.44.1-py3-none-any.whl` is 865,208 bytes and contains no compiled code.** Its
`Requires-Dist` opens with an unconditional hard pin:

```
polars-runtime-32==1.44.1                       # mandatory, NOT an extra
polars-runtime-64==1.44.1 ; extra == "rt64"
polars-runtime-compat==1.44.1 ; extra == "rtcompat"
```

The engine lives in `polars-runtime-32` — MIT, `cp310-abi3` wheels (one covers Python 3.10 through
3.14+) for macOS x86_64/arm64, manylinux + musllinux x86_64/aarch64, win_amd64/arm64, 47–54 MB each.
The split landed at **1.34.0b2 (2025-09-26)**, a 5,966-byte wheel where 1.34.0b1 three days earlier
still shipped ~40 MB binaries; everything from 1.34 on is split. Corroborating: `polars-lts-cpu` is
frozen at **1.33.1 (2025-09-09)**, the version immediately before the split.

Two consequences that bite:

1. **Air-gapped and vendored installs break.** `pip download polars` fetches the 865 KB shim; you must
   also vendor `polars-runtime-32` **for your exact platform** at the identical version. The `==` pin
   at least makes skew a hard resolution failure rather than a misbehaviour.
2. **`py3-none-any` fools platform-audit tooling.** Any script flagging "no wheel for our platform"
   green-lights polars as pure Python while the real 50 MB binary goes unaudited. If you keep an
   install matrix, polars' row must point at `polars-runtime-32`.

## `join_asof` does not check sortedness when you pass `by=`

The folklore says pandas silently gives garbage on unsorted as-of input. **That is out of date.**
`pandas.merge_asof` **raises** `ValueError: left keys must be sorted`. `polars.join_asof`'s own
docstring states the in-memory engine cannot check the sortedness if `by` groups are provided —
**exactly the per-symbol quote join everyone writes.** Unsorted input gives silently wrong rows, no
error. **The real asymmetry:** data sorted by *neither* key is always caught by pandas and never by
polars, so porting a working pandas join to polars can start silently returning wrong rows.

## The two engines disagree on what "sorted" means

- **pandas** wants **one global ascending sort by the `on` key**; its docstring says sorting by any
  additional `by` grouping columns is not required.
- **polars** wants the frame sorted by the `on` key **within each `by` group** — i.e.
  `sort(["symbol","time"])`.

Verified by execution: a frame sorted `["symbol","time"]` (valid polars input) is **rejected by
pandas** with `ValueError: left keys must be sorted`; after a global `sort_values("time")` it works
and gives the correct per-symbol result. **A global sort by the `on` key satisfies both engines — do
that unconditionally and the portability problem disappears.** Then carry the right-hand timestamp
through the join and assert on it, the cheapest guard against a non-prior match.

## Where it fits against the other engines

| Engine | Memory model | Lazy | Out-of-core | Use it when |
|---|---|---|---|---|
| **polars** | Arrow columnar, own pool | `LazyFrame` + optimiser | streaming | Best single-node default for large panels. Streaming coverage of *all* ops at 1.44 is unverified |
| **pandas** 3.x | NumPy blocks (+Arrow opt-in), CoW | eager only | no | Anything that fits in RAM; widest ecosystem. `pd.col()` in 3.0 is initial expression support, **not** a lazy engine |
| **duckdb** | vectorised ~2048-row chunks | SQL planner | spills to disk | SQL-shaped access, and `ASOF JOIN` |
| **dask** / **ray** | partitioned pandas / Arrow blocks | yes | yes | Only with a real cluster; ray has **no musllinux wheels** |

For a 10y × 3000-ticker minute panel — 982,800 bars/ticker, ≈2.95 B rows, ≈153 GB raw — **nothing
fits in RAM in any engine.** Loop one symbol (~51 MB) or one day (~61 MB), or `scan_parquet` with
pushdown. Beware the pivot bomb: a wide close matrix is 982,800 × 3000 × 8 B = **23.6 GB for one
field**, dense, so every untraded minute costs 8 bytes.

## Minimal correct call

```python
import polars as pl

# lazy scan + predicate pushdown; pin polars<2 until 2.0 is reviewed
lf = pl.scan_parquet("panel/**/*.parquet").filter(pl.col("symbol") == "AAPL")

# as-of join: sort GLOBALLY by the `on` key — satisfies polars AND pandas, and is the only
# arrangement polars can validate for you at all when `by=` is present
sig, qt = signals.sort("time"), quotes.sort("time")
j = sig.join_asof(qt, on="time", by="symbol", strategy="backward")

# carry the right-hand timestamp through and assert on it — polars will not warn you
assert j.select((pl.col("quote_time") < pl.col("time")).all()).item(), "matched a non-prior quote"
assert j.height == sig.height, "as-of join dropped rows"

# requirements.txt / lockfile needs BOTH lines, or the engine is not pinned at all:
#   polars==1.44.1
#   polars-runtime-32==1.44.1     # the actual engine; the polars wheel is an empty shim
```

## See also

- `../../../fin-core/skills/market-data-engineering/SKILL.md` §1–2 — the engineering findings list
- `../../../fin-core/skills/market-data-engineering/references/dataframe-engines.md` — reference card
- `../../../fin-core/skills/market-data-engineering/references/asof-joins.md` — the exact-match trap

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`market-data-engineering`** (`../../../fin-core/skills/market-data-engineering/SKILL.md`).

