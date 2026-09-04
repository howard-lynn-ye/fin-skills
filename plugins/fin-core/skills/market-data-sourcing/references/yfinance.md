# yfinance

The default free source, the best-maintained one, and the one whose defaults have changed most.
**~40% of its changelog is "fix Yahoo changed X"** — treat schema drift as the normal operating mode.

| | |
|---|---|
| pip | `yfinance` · **1.7.0 (2026-08-26)** · 150 releases |
| GitHub | `ranaroussi/yfinance` — **25,158★**, 3,412 forks, 104 open issues, pushed 2026-08-27 |
| Licence | **Apache-2.0** (verified three ways: PyPI `info.license`, classifier, and `LICENSE.txt`) |
| Python | `requires_python` is **null** — misleading; the deps require a modern Python |
| Cadence 2026 | 1.4.0 (05-23) · 1.4.1 (05-28) · 1.5.1 (06-28) · 1.5.2 (07-23) · 1.6.0 (08-13) · 1.7.0 (08-26) |

⚠️ I found **no evidence of a historical licence change** — the repo has always shipped Apache-2.0.
If you have seen that claim, treat it as unverified.

## 🚨 Code licence ≠ data licence

README, verbatim: *"yfinance is **not** affiliated, endorsed, or vetted by Yahoo, Inc."* … *"**You
should refer to Yahoo!'s terms of use** … for details on your rights to use the actual data
downloaded."* … *"Remember - the Yahoo! finance API is intended for **personal use only**."*

**Apache-2.0 covers the code only.** Commercial use and redistribution of the *data* is a separate
legal question, and Yahoo's answer is personal-use-only.

## 🚨 The `auto_adjust` history — bisected from source

| yfinance | `yf.download()` default | Released |
|---|---|---|
| ≤ 0.2.50 | `auto_adjust=False` | 2024-11-19 |
| **0.2.51** | **`auto_adjust=True`** ← the flip | **2024-12-19** |
| 0.2.53 – 0.2.66 | `auto_adjust=None` sentinel (warns, behaves as True) | 2025-02-15 → 2025-09-17 |
| **1.0 → 1.7.0** | **`auto_adjust=True`** (hard, warning removed) | 2025-12-22 → now |

`Ticker.history(auto_adjust=True)` has defaulted True **since 0.1.26**; only `download()` ever differed.

🚨 **With `auto_adjust=True` there is NO `Adj Close` column** — OHLC are already adjusted, and
`df['Adj Close']` raises `KeyError` on ≥1.0. Code written before 2025 returns *different numbers*
with no error.

## Current signature

```python
def download(tickers, start=None, end=None, actions=False, threads=True,
             ignore_tz=None, group_by='column', auto_adjust=True, back_adjust=False,
             repair=False, keepna=False, progress=True, period=period_default, interval="1d",
             prepost=False, rounding=False, timeout=10, session=None,
             multi_level_index=True) -> DataFrame | None
```

- **`end` is EXCLUSIVE** — verbatim: *"for end='2023-01-01', the last data point will be on
  '2022-12-31'"*. Classic off-by-one.
- **`multi_level_index=True` (default, since 0.2.48)** → **even a single ticker returns a MultiIndex
  column frame.** Pass `False` for a flat one.
- `group_by='column'` (default) → field-major `('Close','AAPL')`; `'ticker'` → ticker-major.
- `progress=True` prints a progress bar to stdout — **set `False` in pipelines.**
- `period` default is the literal sentinel string `'1mo if start & end None'`.
- `start` defaults to "99 years ago"; `end` to now.

## 🚨 Timezone change in 1.4.0

Docstring for `ignore_tz`, verbatim: *"Default depends on interval. Intraday = False. Day+ = True.
Also controls the returned index's timezone: if True, the index is tz-naive. If False (the intraday
default), the index is converted to the most common exchange timezone among the requested tickers
(**before 1.4.0, this case always returned UTC instead**)."*

**Intraday `download()` returned UTC before 1.4.0 (2026-05-23) and exchange-local since.** A stored
intraday dataset built across that boundary has **shifted bars**. Set `ignore_tz` explicitly.

## 🚨 Survivorship

Yahoo drops delisted symbols. yfinance's own error class is `YFTickerMissingError` with a
`"possibly delisted; "` message prefix — **the library *guesses* delisting because Yahoo won't say.**

**yfinance is survivorship-biased. Do not build a backtest universe from it.** For a bias-free
universe see `_decision-table.md` (EODHD paid, CRSP, Norgate).

## Rate limits and curl_cffi coupling

`yfinance.exceptions.YFRateLimitError` — verbatim: *"Too Many Requests. Rate limited. Try after a
while."* **There is no published Yahoo rate limit**; it is undocumented, IP-based and moves. Treat any
"N requests/hour is safe" claim as folklore.

Crumb/cookie auth landed in 0.2.28. The rate-limit fix history is instructive:

| Version | Changelog entry |
|---|---|
| 0.2.52 | `raise YfRateLimitError if rate limited #2108` |
| 0.2.58 | `Fix false rate-limit problem #2430` |
| 0.2.59 | `Fix the fix for rate-limit #2452` |
| 0.2.60 | `Fix cookie reuse, and handle DNS blocking fc.yahoo.com #2483` |
| 1.0 | `Block curl_cffi version 0.14 #2653` |
| 1.2.1 | `Force curl_cffi>=0.15, because CVE #2743` |
| 1.4.0 | `Make curl_cffi optional with fallback to requests #2802` |
| **1.5.2** | **`Fix yfinance breaking with curl_cffi>=0.16`** |

🚨 **`curl_cffi` is still a hard dependency in 1.7.0** despite 1.4.0's "make it optional" — optional at
*runtime*, still installed. `curl_cffi` is at 0.16.3 (2026-09-02), so **yfinance <1.5.2 breaks against
curl_cffi ≥0.16** — a live pin hazard.

⚠️ **`stockdex` pins `curl_cffi==0.12.0` exactly** and therefore **cannot co-install with yfinance**.

## Other API notes

- `Ticker(ticker, session=None)` — **the old `proxy=` constructor kwarg is gone.** Proxies now go
  through `yf.config.network.proxy`.
- Config surface: `yf.config.network.proxy` (None), **`yf.config.network.retries` (0 — no retries by
  default)**, `yf.config.debug.hide_exceptions` (True), `yf.config.debug.logging` (False),
  `yf.config.locale.lang`/`region`.
- 🚨 **`repair=True` requires `pip install yfinance[repair]`** (scipy + scikit-learn). Plain install
  does not include them. It handles GBp/ZAc/ILA sub-unit 100× mixups, phantom dividends, split repair,
  and capital-gains double-counting (1.1.0).
- `Ticker.get_shares_full(start=None, end=None)` returns a **time series** of shares outstanding —
  distinct from the `Ticker.shares` property.
- New in 1.x: `yf.Search`, `yf.Lookup`, `yf.Market`, `yf.Calendars`, `yf.Auth` (subscription login,
  1.4.0), `yf.WebSocket`/`AsyncWebSocket` (0.2.59), `yf.screen`/`EquityQuery`/`FundQuery`/`ETFQuery`,
  `yf.Sector`/`Industry`, `Ticker.valuation`, `ttm_income_stmt`.
- **30m interval is fetched as 15m and resampled** (documented Yahoo bug workaround). Intraday history
  is capped: *"Intraday data cannot extend last 60 days"*; 1m is ~7 days.
- Classifier still says **`Development Status :: 4 - Beta`** at 1.7.0.

## Canonical safe call

```python
import yfinance as yf

df = yf.download(
    ["AAPL", "MSFT"],
    start="2020-01-01", end="2024-01-01",   # end is EXCLUSIVE
    interval="1d",
    auto_adjust=True,        # explicit: flipped in 0.2.51 and again at 1.0
    group_by="column",
    multi_level_index=False, # flat columns even for one ticker
    progress=False,
    threads=True,
)
# No 'Adj Close' column exists when auto_adjust=True.
# This universe has NO delisted names — do not backtest a screen on it.
```

## Mitigating the 429s

- **`yfinance-cache`** 0.9.3 (2026-08-26 — same day as yfinance 1.7.0; it tracks upstream closely),
  MIT, 116★. *"Intelligent caching, not dumb caching of web requests"* — it understands market
  calendars and only re-fetches genuinely new bars. **The correct first answer to "yfinance keeps
  429-ing me."** Maintained by ValueRaider, a major yfinance contributor.
- **`defeatbeta-api`** — a structurally different answer: a HuggingFace parquet snapshot queried
  locally with DuckDB, **no scraping and no rate limit**, at the cost of ~weekly freshness.
