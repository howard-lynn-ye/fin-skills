# Market data — full comparison

Verified 2026-09-03 against PyPI JSON, GitHub API, and each vendor's own pricing page.
✅ = confirmed at a primary source · ⚠️ = secondhand or JS-rendered page · ❓ = could not verify.

## Delisted-security coverage — the decisive axis

| Source | Delisted? | Note |
|---|---|---|
| **EODHD** (paid) | ✅ **YES** | `get_list_of_tickers(code, delisted=1)`. **The only cheap bias-free equity universe.** Delisted is paid-only |
| SEC EDGAR | ✅ YES | Filings for Lehman/Bear/Enron all intact — but `company_tickers.json` is a *current* snapshot |
| FinanceDataReader | ⚠️ Korea only | `StockListing('KRX-DELISTING')` |
| Tiingo | ❓ | Ticker metadata carries `endDate`; free-tier history retention unverified |
| Twelve Data, Finnhub, Marketstack | ❓ | Not documented on any reachable page |
| openbb | ⚠️ provider-dependent | The abstraction does **not** normalize survivorship characteristics |
| **yfinance, yahooquery, stockdex, defeatbeta, financedatabase, financetoolkit(free), Alpha Vantage, findatapy(equities)** | 🚨 **NO** | All survivorship-biased |

## Free-tier limits

| Service | Free limit | Conf. |
|---|---|---|
| Yahoo-backed | Undocumented; 429 → `YFRateLimitError`. **No published limit** | ✅ |
| **Alpha Vantage** | **25 requests/DAY** (was 500) | ✅ |
| **EODHD** | **20 calls/day** + 500 welcome bonus; **1 year history**; no delisted | ✅ |
| **Marketstack** | **100 requests/MONTH**; 1yr; EOD only | ✅ |
| Tiingo | 50/hr, 1,000/day, **500 unique symbols/mo**, 1 GB/mo. 30+ yrs history | ⚠️ |
| Twelve Data | 8 credits/min, 800/day, **3 exchanges**. **Credits ≠ requests** | ⚠️ |
| FMP (via financetoolkit) | 250/day, **5 years, US only** | ⚠️ |
| Finnhub | **30 calls/second** global cap | ✅ (per-minute ❓) |

## Licence

| Risk | Package | Licence |
|---|---|---|
| 🚨🚨 network copyleft | **openbb** | **AGPL-3.0** (relicensed from MIT 2024-05-14, commit "Update the license… to AGPL" #6415) |
| 🚨 | `dbnomics` | AGPL-3.0 |
| ⚠️ weak copyleft | `marketstack` (3rd-party) | LGPL-2.1, 1★, 2022 — **use `requests` instead** |
| ✅ permissive | yfinance, finnhub-python, findatapy, defeatbeta-api | Apache-2.0 |
| ✅ permissive | yahooquery, stockdex, alpha_vantage, twelvedata, financetoolkit, financedatabase, eodhd, tiingo, FinanceDataReader, akshare, yfinance-cache | MIT |
| ✅ permissive | pandas-datareader | BSD-3 (GitHub misreports NOASSERTION) |

**Code licence ≠ data licence.** yfinance's README: Yahoo's API is *"intended for personal use
only"*. Tiingo: *"you may only use the data for your own personal use and you may not display or
share the data with another person or organization."*

## Per-library notes

### yfinance 1.7.0 (2026-08-26) · Apache-2.0 · 25,158★
Most actively maintained free source; ~40% of its changelog is "fix Yahoo changed X".
- **`auto_adjust` history:** ≤0.2.50 `False` → **0.2.51 (2024-12-19) `True`** → 0.2.53–0.2.66 `None`
  sentinel (warns) → **1.0+ hard `True`, warning removed**. With `auto_adjust=True` there is **no
  `Adj Close` column**.
- **`end` is EXCLUSIVE**; `multi_level_index=True` even for one ticker; `progress=True` prints.
- 🚨 **Intraday returned UTC before 1.4.0 (2026-05-23), exchange-local since.**
- `curl_cffi` is a hard dependency; **yfinance <1.5.2 breaks against curl_cffi ≥0.16**.
- `repair=True` requires `pip install yfinance[repair]` (scipy + sklearn).
- `yf.config.network.retries` defaults to **0**.

### pandas-datareader 0.11.1 (2026-06-24) · BSD-3
🚨 **0.11.0 deleted every equity reader** — Yahoo, Stooq, Tiingo, IEX, Quandl, AlphaVantage,
Morningstar, Naver, Nasdaq Trader, `Options`. `DataReader` now accepts only `bankofcanada`, `fred`,
`famafrench`, `oecd`, `eurostat`, `econdb`. Its Fama-French parser was **silently wrong before
0.11.0** — re-pull anything fetched with 0.10.0. `requires_python >=3.11`. Econdb now needs a key.

### yahooquery 2.4.1 (2025-05-15) · MIT · ⚠️ ~16 months no commits
Unique: **bulk multi-symbol in one request** (far better under rate limiting) and Yahoo Premium via
Selenium. 🚨 **`adj_ohlc=False` by default — the opposite of yfinance.** `period="ytd"` default.
Multi-symbol responses can contain **error strings** where a dict is expected. Open 2026 bugs unanswered.

### defeatbeta-api 0.0.60 (2026-06-10) · Apache-2.0 · 744★
Serves Yahoo-derived data from a **HuggingFace parquet dataset queried locally with DuckDB** — *"no
scraping issues and rate limits"*. The structural answer to 429s. 🚨 **Updates ~weekly — not
real-time.** Windows native only since 0.0.60. Inherits Yahoo's survivorship bias.

### EODHD · MIT
🚨 **PyPI is 21 months stale**: published `1.0.32 (2024-12-18)` vs repo main `1.4.0 (2026-08-28)`.
Their changelog admits *"Twenty months of work reached `main` without being published"* and that
**no commit corresponds to the last published package.** Install from git.

### openbb 4.7.2 · AGPL-3.0 · 72,649★
One normalized schema over ~30 vendors. 🚨 **The schema hides provider differences that matter** —
adjustment convention, corporate-action handling and delisted coverage all differ behind the same
call. **Always pin `provider=`.** `openbb-terminal` has been **removed from PyPI** (404).

### findatapy 0.1.42 · Apache-2.0
🔑 **Free tick-level FX from DukasCopy** and 🔑 **ALFRED vintages** — the only library here offering
point-in-time macro. Author's own README: *"a highly experimental alpha project"*. Redis caching
errors are noisy on Windows.

### financetoolkit 2.2.0 / financedatabase 2.4.0 · MIT
`financetoolkit` runs **fully free with no key** via `enforce_source="YahooFinance"`. 🚨 **Silent
FMP→Yahoo fallback** means one DataFrame can mix two conventions — set `enforce_source` explicitly.
FMP fundamentals are **restated, not point-in-time**. `financedatabase` is 300k+ symbols offline but
a **today snapshot** = maximal survivorship bias.

## Dead — do not recommend
`investpy` (README banner: not working; 244 open issues), `investiny` (dead the day it released),
`yahoo_fin` (broken by Yahoo crumb auth), `thepassiveinvestor` (archived), `alpaca-trade-api`
(→ `alpaca-py`), `marketstack` PyPI packages, `quandl` (→ `nasdaq-data-link`, itself stale 2022).
