---
name: lib-yfinance
description: >-
  The default free Yahoo Finance downloader, whose yf.download() now returns pre-adjusted OHLC
  with no Adj Close column at all. TRIGGER - import yfinance as yf, pip install yfinance,
  yf.download, yf.Ticker, Ticker.history, auto_adjust, multi_level_index, ignore_tz, repair=True,
  get_shares_full, yf.Search, yf.Lookup, yf.WebSocket, yfinance-cache; errors "KeyError: 'Adj
  Close'", YFRateLimitError, "Too Many Requests. Rate limited", YFTickerMissingError, "possibly
  delisted", curl_cffi pin conflicts, one ticker returning MultiIndex columns. Memory is stale
  here: auto_adjust flipped at 0.2.51 and hardened at 1.0, intraday timezones changed at 1.4.0,
  the proxy= constructor kwarg is gone, and 1.7.0 shipped 2026-08-26. SKIP for choosing between
  data vendors (market-data-sourcing) and for A-share data (china-ashare-data). SKIP when the
  question is WHICH library to choose rather than how to use this one - that belongs to the domain
  skill.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# yfinance

The default free source, the best-maintained one, and the one whose defaults have changed most.
Roughly **40% of its changelog is "fix Yahoo changed X"** — treat schema drift as normal operation.

| | |
|---|---|
| pip / import | `pip install yfinance` · `import yfinance as yf` |
| Version | **1.7.0 (2026-08-26)** · 150 releases · cadence 1.4.0 (05-23) → 1.7.0 (08-26) |
| Licence | **Apache-2.0** (verified via PyPI `info.license`, classifier, and `LICENSE.txt`) |
| Status | ✅ 25,158★ `ranaroussi/yfinance`, pushed 2026-08-27. Classifier still says **Beta** |

## The trap that costs you money

🚨 **`auto_adjust=True` is now the hard default in `yf.download()`, and there is no `Adj Close` column.** OHLC come back already adjusted; `df['Adj Close']` raises `KeyError` on ≥1.0. Code written before 2025 does not error — **it returns different numbers.**

| yfinance | `yf.download()` default | Released |
|---|---|---|
| ≤ 0.2.50 | `auto_adjust=False` | 2024-11-19 |
| **0.2.51** | **`auto_adjust=True`** ← the flip | **2024-12-19** |
| 0.2.53 – 0.2.66 | `auto_adjust=None` sentinel (warns, behaves as True) | 2025-02-15 → 2025-09-17 |
| **1.0 → 1.7.0** | **`auto_adjust=True`** (hard, warning removed) | 2025-12-22 → now |

`Ticker.history(auto_adjust=True)` has defaulted True **since 0.1.26** — only `download()` differed.
Always pass `auto_adjust` explicitly so the number does not depend on the installed version.

## Signature traps in `download()`

- **`end` is EXCLUSIVE.** Verbatim: *"for end='2023-01-01', the last data point will be on
  '2022-12-31'"*.
- **`multi_level_index=True` since 0.2.48** → **even one ticker returns MultiIndex columns.** Pass
  `False` for a flat frame.
- `group_by='column'` (default) → field-major `('Close','AAPL')`; `'ticker'` → ticker-major.
- `progress=True` writes to stdout — set `False` in pipelines.
- `start` defaults to "99 years ago"; `period`'s default is the literal sentinel string
  `'1mo if start & end None'`.
- **`repair=True` needs `pip install yfinance[repair]`** (scipy + scikit-learn); a plain install
  lacks them. It fixes GBp/ZAc/ILA 100× sub-unit mixups, phantom dividends, split repair, and
  capital-gains double counting (1.1.0).

## 🚨 Intraday timezones moved in 1.4.0

`ignore_tz` docstring, verbatim: *"Default depends on interval. Intraday = False. Day+ = True … if
False (the intraday default), the index is converted to the most common exchange timezone among the
requested tickers (**before 1.4.0, this case always returned UTC instead**)."*
**Intraday `download()` returned UTC before 1.4.0 (2026-05-23) and exchange-local since.** A stored
intraday dataset spanning that boundary has **shifted bars**. Set `ignore_tz` explicitly.

## 🚨 Survivorship, and the data licence

Yahoo drops delisted symbols. yfinance's own `YFTickerMissingError` carries a `"possibly delisted; "`
prefix — **the library guesses at delisting because Yahoo will not say.** Do not build a backtest
universe from it. README, verbatim: *"yfinance is **not** affiliated, endorsed, or vetted by Yahoo,
Inc."* … *"the Yahoo! finance API is intended for **personal use only**."* **Apache-2.0 covers the
code only** — redistributing or commercializing the *data* is a separate legal question.

## Rate limits and the curl_cffi pin

`yfinance.exceptions.YFRateLimitError` — *"Too Many Requests. Rate limited. Try after a while."*
**There is no published Yahoo rate limit**; it is undocumented, IP-based and moves. Any "N
requests/hour is safe" number is folklore. `yf.config.network.retries` defaults to **0**.

🚨 **`curl_cffi` remains a hard dependency at 1.7.0** despite 1.4.0's "make it optional" — optional at
*runtime*, still installed. **yfinance <1.5.2 breaks against curl_cffi ≥0.16** (fixed in 1.5.2), and
`curl_cffi` is at 0.16.3. `stockdex` pins `curl_cffi==0.12.0` and therefore cannot co-install.

First answer to "yfinance keeps 429-ing me": **`yfinance-cache`** (0.9.3, 2026-08-26, MIT) — calendar-
aware caching that only re-fetches genuinely new bars. Structural alternative: **`defeatbeta-api`**, a
HuggingFace parquet snapshot queried with DuckDB — no scraping, no rate limit, ~weekly freshness.

## Other API notes

`Ticker(ticker, session=None)` — **the old `proxy=` constructor kwarg is gone**; proxies go through
`yf.config.network.proxy`. New in 1.x: `yf.Search`, `yf.Lookup`, `yf.Market`, `yf.Calendars`,
`yf.Auth` (1.4.0), `yf.WebSocket`/`AsyncWebSocket`, `yf.screen`/`EquityQuery`, `yf.Sector`/`Industry`,
`Ticker.valuation`, `ttm_income_stmt`, and `Ticker.get_shares_full(start, end)` (a *series*, unlike
the `Ticker.shares` property). **30m bars are fetched as 15m and resampled** (a documented Yahoo
workaround); intraday history is capped — *"Intraday data cannot extend last 60 days"*, 1m ~7 days.

## Minimal correct call

```python
import yfinance as yf
df = yf.download(
    ["AAPL", "MSFT"],
    start="2020-01-01", end="2024-01-01",   # end is EXCLUSIVE
    interval="1d",
    auto_adjust=True,        # explicit: flipped in 0.2.51, hardened at 1.0
    multi_level_index=False, # flat columns even for one ticker
    ignore_tz=True,          # explicit: intraday default changed in 1.4.0
    progress=False,
)
# No 'Adj Close' column exists when auto_adjust=True.
# This universe has NO delisted names — do not backtest a screen on it.
```

## See also

- `../../../fin-core/skills/market-data-sourcing/SKILL.md` — vendor choice, survivorship-free universes
- `../../../fin-core/skills/market-data-sourcing/references/yfinance.md` — the source card
- `../../../fin-core/skills/research-integrity-guards/references/adjustment-conventions.md` — split/dividend conventions

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`market-data-sourcing`** (`../../../fin-core/skills/market-data-sourcing/SKILL.md`).

