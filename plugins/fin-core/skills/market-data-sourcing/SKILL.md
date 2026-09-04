---
name: market-data-sourcing
description: >-
  Choose a market price or reference data vendor and use it without silently corrupting the
  numbers. TRIGGER - download, fetch, pull or load OHLCV, prices, quotes, bars or a ticker
  universe; compare vendors on cost, coverage or free-tier limits; need delisted US or global
  tickers, or a survivorship-free universe; two sources disagree; hitting 429 or rate limits;
  "KeyError: Adj Close"; split and dividend adjustment; trading calendars and holidays. Covers
  yfinance, yahooquery, defeatbeta, EODHD, Tiingo, Twelve Data, Finnhub, Alpha Vantage,
  Polygon/Massive, Databento, openbb, findatapy, financetoolkit, exchange_calendars, and
  alternative data. Also covers 美股 and global 行情数据 requests. SKIP for storing, partitioning or
  as-of joining data you already hold (market-data-engineering); for EDGAR filings, XBRL, CIK and
  macro vintages (fundamental-and-macro-data); and for A-share, 沪深 or 退市 queries
  (china-ashare-data).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-03"
---

# Market data sourcing

Pick the source from the constraint that actually binds — usually *survivorship coverage* or
*licence*, not price or convenience. Then read that library's reference file before writing code;
every one of them has a default that is wrong for research.

## 1. Pick a source

| Your binding constraint | Use | Why |
|---|---|---|
| Free, exploratory, US/global equities | `yfinance` | Best-maintained (1.7.0, 2026-08-26), broadest coverage, no key |
| Free but **429s keep killing me** | `defeatbeta-api`, or `yfinance-cache` | defeatbeta serves a HuggingFace parquet snapshot via DuckDB — **no scraping, no rate limit**; refreshed ~weekly |
| **Survivorship-free universe on a budget** | **EODHD** (~$20–100/mo) | `get_list_of_tickers(code, delisted=1)`. The only cheap genuinely bias-free equity universe |
| Institutional survivorship-free | CRSP (via `wrds`), Norgate, Polygon | |
| Vendor-cleaned EOD, quality reputation | Tiingo | 30+ yrs on free tier — but ToS is **internal use only, no sharing or display** |
| One code path across many vendors | `openbb` | 🚨 **AGPL-3.0** since 2024-05-14. Network copyleft |
| **Free tick-level FX** | `findatapy` (DukasCopy backend) | Nothing else in the free tier offers tick data |
| **Point-in-time macro** | `findatapy` (ALFRED) or `fredapi` | → `fundamental-and-macro-data` |
| Tick/full-depth US equities & futures | `databento` | Actively released (2026-09-01); pay-per-use |
| Options chains, aggregates, delisted | `polygon-api-client` | 🚨 **Polygon.io has rebranded to Massive.com** — `polygon.io/pricing` 301-redirects to `massive.com/pricing`, and the client has had **no PyPI release since 2025-10-30** (the announcement date). Update links and expect the package name to move |
| Symbol/metadata universe, no key | `financedatabase` | 300k+ symbols offline — 🚨 but a *today* snapshot |
| Long-history financial statements | `financetoolkit` (`enforce_source="YahooFinance"` = free, keyless) | ~150 ratios computed transparently |
| Korea / Japan / HK / Vietnam free | `FinanceDataReader` | Only free source with an explicit `KRX-DELISTING` list |
| China A-share | → `fin-china` plugin | Entirely different traps |
| Crypto | → `fin-crypto` plugin | |

**Do not use:** `investpy` (dead 2022, Investing.com blocks it), `investiny` (dead the day it was
released), `yahoo_fin` (broken by Yahoo's crumb auth), `thepassiveinvestor` (archived), the PyPI
`marketstack` package (1★, LGPL, 2022 — call the REST API with `requests` instead),
`alpaca-trade-api` (→ `alpaca-py`).

## 2. The four traps that produce wrong numbers silently

### 2a. Survivorship

**Yahoo-derived sources have no delisted securities.** yfinance's own error class is
`YFTickerMissingError` with a `"possibly delisted; "` prefix — the library *guesses*, because Yahoo
won't say. This applies to yfinance, yahooquery, stockdex, defeatbeta-api, financedatabase,
financetoolkit(free) and Alpha Vantage alike. **Do not build a backtest universe from any of them.**
Full coverage table: `research-integrity-guards` §1.

### 2b. Adjustment defaults disagree across libraries

| Call | Default | Result |
|---|---|---|
| `yf.download()` | `auto_adjust=True` (since 1.0) | **Adjusted, and no `Adj Close` column** |
| `yf.Ticker().history()` | `auto_adjust=True` (since 0.1.26) | Adjusted |
| `yahooquery .history()` | **`adj_ohlc=False`** | 🚨 **Unadjusted — the opposite of yfinance** |
| `tiingo .get_dataframe([list])` | `metric_name='adjClose'` | Adjusted close only, no OHLC |
| Alpha Vantage free | unadjusted (adjusted endpoint is premium) | 🚨 Split-corrupted |

Reconciling yfinance against yahooquery without setting these explicitly produces silent mismatches
around every split. **Always pass the adjustment argument explicitly**, even when the default is
what you want — it documents intent and survives the next default flip.

### 2c. Timezones and off-by-one

- **`yf.download(end=...)` is EXCLUSIVE.** `end='2023-01-01'` gives you a last bar of 2022-12-31.
- 🚨 **yfinance intraday returned UTC before 1.4.0 (2026-05-23) and exchange-local since.** A stored
  intraday dataset built across that boundary has shifted bars. Set `ignore_tz` explicitly.
- `multi_level_index=True` is the default, so **even a single ticker returns a MultiIndex frame**.
  Pass `multi_level_index=False` for a flat one.

### 2d. Licence ≠ data licence

`yfinance` is Apache-2.0, but its README says Yahoo's API is **"intended for personal use only"** and
that your rights to the *data* are governed by Yahoo's ToS. The same split applies to every scraper.
Tiingo's terms are stricter still: *"you may only use the data for your own personal use and you may
not display or share the data with another person or organization."*

🚨 **AGPL-3.0 in this domain:** `openbb`, `dbnomics`. GPL: `edgar-crawler`. If the work will be
served over a network, these are blockers.

## 3. Free-tier reality (verified 2026-09-03)

| Service | Free limit | Confidence |
|---|---|---|
| Yahoo (yfinance et al.) | Undocumented; 429 → `YFRateLimitError`. **No published limit — treat any "N/hour is safe" claim as folklore** | ✅ |
| **Alpha Vantage** | **25 requests/DAY** (was 500) — effectively a demo | ✅ |
| **EODHD** | **20 calls/day**, 1 year of history, **no delisted** (delisted is paid-only) | ✅ |
| **Marketstack** | **100 requests/MONTH**, 1yr, EOD only | ✅ |
| Tiingo | 50/hr, 1,000/day, **500 unique symbols/month**, 1 GB/mo | ⚠️ |
| Twelve Data | 8 credits/min, 800 credits/day, 3 exchanges. **Credits ≠ requests** | ⚠️ |
| FMP (via financetoolkit) | 250/day, **5 years only, US only** | ⚠️ |
| Finnhub | **30 calls/second** global cap; 429 on exceed. Per-minute free limit **not verifiable** — the widely repeated "60/min" is unconfirmed | ✅ / ❓ |

## 4. Canonical safe snippet

```python
import yfinance as yf

df = yf.download(
    ["AAPL", "MSFT"],
    start="2020-01-01", end="2024-01-01",   # end is EXCLUSIVE
    interval="1d",
    auto_adjust=True,        # explicit: the default flipped in 0.2.51 and again at 1.0
    group_by="column",
    multi_level_index=False, # flat columns even for one ticker
    progress=False,          # never print in a pipeline
    threads=True,
)
# NOTE: with auto_adjust=True there is NO 'Adj Close' column — OHLC are already adjusted.
# NOTE: this universe has NO delisted names. Do not backtest a screen on it.
```

## 5. Calendars and identifiers

- Sessions/holidays: `exchange_calendars` (4.13.2, Apache-2.0, genuinely well maintained) or
  `pandas_market_calendars` (5.4.0). Never hand-roll a business-day calendar — half-days and holiday
  drift will break your alignment. **But it has verified defects — see below.**

🚨 **`exchange_calendars`' default date bounds are a MOVING TARGET, and this breaks reproducibility.**
✅ Verified: `GLOBAL_DEFAULT_START` / `_END` are computed from `pd.Timestamp.now()` **at import** as
**today − 20 years** and **today + 1 year**. Running on 2026-09-04, `get_calendar("XTKS")` returns
`first_session=2006-09-04`, `last_session=2027-09-03`. **The same code returns a different calendar
tomorrow.** → **Always pass explicit `start=` and `end=`.**

🚨 **There is no `XNSE`.** ✅ Verified: `"XNSE" in get_calendar_names()` → **False**. India is
**`XBOM`** (BSE) only — used as a silent proxy for NSE, which carries most Indian volume. (`XBSE` is
Bucharest, not Bombay.) Other verified gaps: **Korean CSAT late-open dates stop at 2021-11-18 even on
master**, so `XKRX.session_open("2024-11-14")` wrongly returns 09:00; **`XSES` models no lunch break
ever** despite SGX having one until 2011; and **`XBOM` has zero sessions in 2027** because Indian
holidays are hand-maintained annually. Detail: `../../../fin-asia/skills/asia-pacific-markets/SKILL.md`.
- Never resample intraday bars over wall-clock time across a session break. Compute rolling windows
  over **bar index**.
- Identifier mapping (ticker ↔ CIK ↔ FIGI ↔ PERMNO) is a source of silent joins onto the wrong
  entity — see `research-integrity-guards` §1 and `fundamental-and-macro-data`.

## 6. Reference files

One file per library in `references/`, each with: exact pip name, version + release date, licence,
maintenance verdict, coverage, free-tier limits, **traps**, and a minimal snippet.
Grep them rather than guessing:

```bash
grep -ril "delisted" plugins/fin-core/skills/market-data-sourcing/references/
grep -i -A6 "TRAP" plugins/fin-core/skills/market-data-sourcing/references/yfinance.md
```

Start from `references/_decision-table.md` for the full side-by-side comparison.
