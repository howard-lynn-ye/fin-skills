# defeatbeta-api

A structurally different answer to "yfinance keeps 429-ing me": instead of scraping Yahoo, it
queries a **HuggingFace parquet dataset locally with DuckDB**. No HTTP scraping, no cookies, no
crumb, **no rate limit** — paid for with roughly a week of staleness.

| | |
|---|---|
| pip / import | `defeatbeta-api` / `from defeatbeta_api.data.ticker import Ticker` |
| version | **0.0.60 (2026-06-10)** · 30 releases ✅ — still `0.0.x` after 30 releases ⚠️ |
| GitHub | `defeat-beta/defeatbeta-api` — **744★**, 60 forks, 4 open issues, pushed 2026-08-06 ✅ |
| Licence | **Apache-2.0** ✅ (PyPI `license` + GitHub `spdx_id`) — **code only** |
| Python | `>=3.11` ✅ |
| Data | HuggingFace dataset `defeatbeta/yahoo-finance-data`, parquet, read via DuckDB + `cache_httpfs` |
| Verdict | ✅ maintained (created 2025-04, pushed 2026-08) — but pre-1.0, no API stability promise |

Verified 2026-09-04 via the PyPI JSON API, the GitHub REST API, and the repository README.

## 🚨 Traps

🚨 **The data is refreshed roughly weekly.** README verbatim, under *Disadvantages compared to
yfinance*: *"defeat-beta updates data on a periodic basis (**typically weekly**), so it cannot
provide real-time data, unlike `yfinance`."* And in the advantages: *"fetching data periodically
(typically once a week) and uploading it to Hugging Face."*

**"Typically" is doing real work there — there is no published refresh SLA.** Anything that computes
"today's signal" gets a stale answer with no error and no timestamp warning. Always read the maximum
`report_date` your query returned and assert it against your expected as-of date before you act.

🚨 **It is Yahoo-derived, so it inherits Yahoo's survivorship bias in full.** Delisted names are
absent. A snapshot of surviving tickers is a *worse* backtest universe than a live scrape, because
it feels like a curated dataset. **Do not build a screen or a backtest universe from it** — see
`_decision-table.md`; the delisted column is 🚨 NO.

🚨 **`duckdb==1.5.3` is an EXACT pin** ✅ (from `requires_dist`). It cannot co-install with any other
package requiring a different DuckDB. Given how much of this stack uses DuckDB — see
`../../market-data-engineering/references/dataframe-engines.md` — install it in its own environment
by default.

🚨 **It requires `pandas>=3.0.1`** ✅ — that is a floor, not a ceiling, so installing it **forces
pandas 3.x on your whole environment**. pandas 3 makes Copy-on-Write mandatory (chained assignment
silently stops working) and changes Timestamp resolution inference. See
`../../market-data-engineering/references/dataframe-engines.md` before adding it to an existing project.

⚠️ **The hard dependency list is startling for a data client** ✅: `openai>=1.106.1`, `nltk>=3.9.4`,
`matplotlib`, `rich`, `pyfiglet`, `openpyxl`, `psutil`, and **`ipython<=8.37.0,>=8.0.0`** — an upper
bound on IPython that will fight a modern Jupyter install. None of these are extras.

⚠️ **Apache-2.0 covers the client and the loader, not the underlying data.** The content is
Yahoo-derived; Yahoo's terms say personal use only. Republishing it does not become permissible
because it passed through a HuggingFace dataset. Same code-vs-data split as `yfinance.md`.

⚠️ **First query is slow.** `cache_httpfs` caches parquet ranges to local disk; the cold path pulls
from HuggingFace over the network. Sub-second is the warm-cache claim, not the first call.

## What you get that yfinance does not

Beyond OHLCV, the dataset carries derived series computed once and served to everyone —
**TTM EPS, TTM P/E, historical market cap, P/S, P/B, PEG, ROE, ROA, ROIC, WACC, equity multiplier,
asset turnover**, plus **earnings-call transcripts**, SEC filing references, news, and
**revenue by segment and by geography**. Those last two are genuinely hard to assemble yourself.

Because it is parquet under DuckDB, a `Tickers([...])` call returns one long DataFrame with a
`symbol` column rather than N round-trips — the bulk-history workflow is where it wins outright.

## Minimal correct snippet

```python
from defeatbeta_api.data.ticker import Ticker
from defeatbeta_api.data.tickers import Tickers

px = Tickers(["NVDA", "TSLA"], max_workers=2).price()   # one DataFrame, `symbol` column

# 🚨 the data is ~weekly: prove freshness before you use it as "latest"
import pandas as pd
asof = pd.to_datetime(px["report_date"]).max()
assert (pd.Timestamp.utcnow().tz_localize(None) - asof).days <= 10, f"stale snapshot: {asof:%Y-%m-%d}"

t = Ticker("TSLA")
t.quarterly_income_statement().print_pretty_table()
t.roe(); t.wacc()
```

## When to reach for it

| Situation | Answer |
|---|---|
| Bulk historical panel, hundreds of tickers, no key, no 429s | ✅ **defeatbeta-api** |
| Today's close, or anything intraday | ❌ use `yfinance.md` |
| yfinance 429s but you need current bars | **`yfinance-cache`** — calendar-aware caching, see `yfinance.md` |
| A backtest universe that includes delistings | ❌ neither — `eodhd.md` (paid) |
| Fundamentals you can defend as point-in-time | ❌ `../../fundamental-and-macro-data/references/edgartools.md` |

Cross-references: `yfinance.md` · `_decision-table.md` ·
`../../market-data-engineering/references/dataframe-engines.md` (DuckDB and pandas 3.x).
