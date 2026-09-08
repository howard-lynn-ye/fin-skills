# databento

Pay-per-use tick and full-depth data for US equities, futures and options — the only vendor on this
list that sells **raw market data by the gigabyte** instead of a monthly seat. Nothing is derived:
**no greeks, no IV, no adjusted close.** You build every analytic yourself.

| | |
|---|---|
| pip / import | `databento` / `import databento as db` |
| version | **0.86.0 (2026-09-01)** · 111 releases ✅ |
| GitHub | `databento/databento-python` — **297★**, 52 forks, **0 open issues**, pushed 2026-09-01 ✅ |
| Licence | **Apache-2.0** (PyPI `license_expression` + GitHub `spdx_id`) ✅ — **code only** |
| Python | `>=3.10` ✅ · pure-Python wheel; native decoding lives in `databento-dbn`, pinned `~0.69.0` |
| Classifier | still `Development Status :: 4 - Beta` at 0.86.0 ⚠️ |
| Verdict | ✅ **vendor-maintained and actively released** — zero open issues, pushed the day of its release |

Verified 2026-09-04 via the PyPI JSON API, the GitHub REST API, and by reading
`databento/historical/api/{metadata,timeseries}.py` and `databento/common/{dbnstore,enums}.py`.

## 🚨 Traps

🚨 **No greeks and no implied vol — for any asset class.** OPRA.PILLAR gives you MBO / MBP-1 /
MBP-10 / TBBO / trades / OHLCV / definition / statistics / status and nothing else. If you want an
IV surface you solve it yourself from the quotes (see
`../../derivatives-pricing/references/vollib.md`). This is the *tick* option, not the *analytics*
option — the opposite trade-off from ORATS or Tradier, which hand you a modelled surface.

🚨 **Billing is per GB of UNCOMPRESSED raw binary.** `Metadata.get_billable_size` says so verbatim:
*"Request the billable **uncompressed raw binary size**"* ✅. Data arrives zstd-compressed and lands
small on disk; the invoice is computed on the pre-compression figure. Judging cost by the size of
your parquet output will understate it by a large factor.

🚨 **`symbols=None` means ALL SYMBOLS.** Verbatim from the docstring: *"If `'ALL_SYMBOLS'` or `None`
then will select **all** symbols."* ✅ Omitting `symbols` on `GLBX.MDP3` or `OPRA.PILLAR` is a
runaway bill, not an error. Full-depth OPRA is among the highest-volume feeds in existence.
Cap at 2,000 symbols per request (documented limit).

🚨 **`end` is EXCLUSIVE, `start` inclusive**, and both **assume UTC** when you pass a naive value ✅.
Same off-by-one as yfinance, on data that costs money to re-pull.

🚨 **`stype_out` defaults to `"instrument_id"`** ✅ — the wire format carries integer instrument IDs,
not tickers. `to_df()` hides this because `map_symbols=True` is its default, but any code that
consumes the DBN store directly (`.replay()`, `to_ndarray()`) sees integers. Instrument IDs are
**not stable across datasets or over time**; re-resolve them, never cache them as a join key.

🚨 **`to_df(price_type=PriceType.FLOAT)` is the default and it is lossy** ✅. DBN stores prices as
fixed-point int64. `FLOAT` converts to float64 (fine for research, wrong for exact tick arithmetic
and reconciliation); `PriceType.FIXED` keeps the integers; `PriceType.DECIMAL` gives
`decimal.Decimal`. See `../../market-data-engineering/references/storage-formats.md` on float
precision for why this bites at the sub-cent level.

⚠️ **Pricing (vendor page, not re-verified this session):** usage-based **$/GB**, plus subscription
tiers **$199/mo** Standard, **$1,750/mo** Plus, **$4,500/mo** Unlimited, with **$125 of free credits
that expire after 6 months**. Treat the numbers as indicative and check the live page.

⚠️ **Apache-2.0 covers the client, not the data.** Redistribution rights come from your Databento
subscription agreement, and exchange data (OPRA, CME) carries its own licensing. Same code-vs-data
split as `yfinance.md`, with real money attached.

## What it has that the free sources do not

- ✅ **Point-in-time instrument definitions**, README verbatim: *"free of look-ahead bias and
  retroactive adjustments."* This is the single strongest reason to use it for backtests — Yahoo-
  derived sources cannot make that claim (see `yfinance.md` on survivorship).
- **Identical schemas for live and historical**, so a backtest and a live handler share one parser.
- **Full order-book depth (MBO)** — the only reachable-price source in this file for queue-position
  or microstructure work.
- **Smart symbology** for futures rollovers (`stype_in='parent'`, `'continuous'`) rather than
  hand-built roll calendars.
- `Metadata.get_dataset_range` and `get_dataset_condition` tell you coverage and data quality per
  dataset **before** you pay ✅ — use them; the OPRA history start date is ❓ not confirmed here.

## Minimal correct call — cost first, download second

```python
import databento as db

client = db.Historical()                       # reads DATABENTO_API_KEY; never hardcode the key
args = dict(
    dataset="GLBX.MDP3",
    symbols=["ES.FUT"],                        # 🚨 never omit: None means ALL SYMBOLS
    stype_in="parent",
    schema="ohlcv-1m",                         # explicit; the default is 'trades'
    start="2024-01-02T14:30", end="2024-01-02T21:00",   # UTC, end EXCLUSIVE
)
print(client.metadata.get_billable_size(**args), "bytes uncompressed")
print(client.metadata.get_cost(**args), "USD")             # gate on this before get_range

data = client.timeseries.get_range(**args)     # blocks until fully downloaded
df = data.to_df(price_type="decimal", map_symbols=True, tz="UTC")  # FLOAT default is lossy
```

For anything above a few GB use `client.batch.submit_job(...)` instead — `get_range` only returns
once the whole stream has landed.

## Where it sits

| Need | Go to |
|---|---|
| Free daily bars, prototyping | `yfinance.md` (survivorship-biased) |
| Cheap bias-free equity universe with delistings | EODHD — `eodhd.md` |
| Modelled option greeks / IV surface | ORATS or Tradier — see `../../derivatives-pricing/` |
| **Full-depth ticks, point-in-time definitions, pay per GB** | **databento** |

See `_decision-table.md` for the full vendor comparison and free-tier limits.
