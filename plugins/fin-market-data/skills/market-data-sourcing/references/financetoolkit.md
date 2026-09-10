# financetoolkit

150+ ratios, technicals, risk and performance metrics with the formulas written out — and it runs
**completely free with no API key at all**. That convenience is also its worst hazard: the source
under the numbers can change without you asking, and one of its defaults changes with your
subscription tier.

| | |
|---|---|
| pip / import | `financetoolkit` / `from financetoolkit import Toolkit` |
| version | **2.2.0 (2026-08-18)** · 72 releases ✅ |
| GitHub | `JerBouma/FinanceToolkit` — **5,288★**, 614 forks, **6 open issues**, pushed 2026-09-01 ✅ |
| Licence | **MIT** ✅ (PyPI `license` + classifier + GitHub `spdx_id` all agree) |
| Python | `>=3.11,<3.16` ✅ · `Development Status :: 5 - Production/Stable` ✅ |
| Verdict | ✅ **healthy and unusually well-triaged** — 6 open issues against 5.3k stars |

Verified 2026-09-04 via PyPI JSON, the GitHub REST API, and by reading
`financetoolkit/toolkit_controller.py` on `main`.

## 🚨 Traps

🚨 **The silent FMP → Yahoo fallback.** README verbatim: *"By default, the Finance Toolkit
prioritizes Financial Modeling Prep for data retrieval. If data acquisition from Financial Modeling
Prep is unsuccessful (e.g., due to plan restrictions or API key issues), **the toolkit automatically
switches to Yahoo Finance as a secondary source**."* The source docstring says the same ✅.

This is not a per-session choice — it is **per request**. A free FMP key covers 5 years, so a
2015→2024 pull returns **FMP rows for the years it can serve and Yahoo rows for the rest, in one
DataFrame, with no marker**. FMP and Yahoo differ on fiscal-period alignment, line-item naming and
currency handling, so the ratio you compute spans two conventions. There is no error and no warning.
**Set `enforce_source` explicitly, always.**

🚨 **An invalid key silently downgrades you to Yahoo.** Read from source: when
`_determine_subscription_plan` flags the key as invalid, the toolkit sets
`self._enforce_source = "YahooFinance"` and logs at ERROR level ✅. In a pipeline that swallows logs,
a typo'd key looks like a successful run with different numbers.

🚨 **`convert_currency`'s default depends on your FMP plan.** Verbatim from source:

```python
self._convert_currency = (
    convert_currency if convert_currency is not None else self._fmp_plan != "Free"
)
```

**On the Free plan currency conversion is OFF; on any paid plan it is ON** ✅. The identical script
run against a free key and a paid key produces different Free Cash Flow Yield and P/E for any
non-USD-reporting company — and the diff looks like a data bug, not a config difference. Pass
`convert_currency=` explicitly. (`sleep_timer` is plan-dependent in the same way.)

🚨 **FMP's free tier is 250 requests/day, 5 years of history, US-listed companies only** ✅
(README verbatim). Not "250 companies" — 250 *requests*, and the toolkit threads several per ticker.

⚠️ **FMP serves restated financials, not point-in-time.** What you get for FY2018 is the figure as
it stands *today*, after every subsequent restatement and reclassification — the very numbers a
2019 investor could not have seen. Any fundamental backtest built on it has look-ahead baked into
the input, and no amount of purged CV fixes that. **For point-in-time US fundamentals use as-filed
SEC data** — `../../fundamental-and-macro-data/references/edgartools.md`.

⚠️ **`benchmark_ticker="SPY"` is added to your ticker list silently** and removed again after
fetching ✅. It costs requests against your daily quota and pulls a second data source into the run.
Set `benchmark_ticker=None` when you do not need Sharpe/Treynor.

⚠️ **Yahoo-sourced mode inherits Yahoo's survivorship bias** — the free path has no delisted names.
See `yfinance.md` and `_decision-table.md`.

## The free, no-key path — which is the point of the library

`api_key` defaults to `os.environ.get("FINANCIAL_MODELING_PREP_API_KEY", "")` ✅. With no key set,
the docstring is explicit: *"the Finance Toolkit will always attempt to acquire data from Financial
Modeling Prep if an API key is set. **If this isn't the case, the data comes from Yahoo Finance.**"*

```python
from financetoolkit import Toolkit

companies = Toolkit(
    tickers=["AAPL", "MSFT"],
    enforce_source="YahooFinance",   # 🚨 pin the source; no key needed, no silent fallback
    start_date="2018-01-01",
    quarterly=False,
    convert_currency=False,          # 🚨 default is plan-dependent — never leave it None
    benchmark_ticker=None,           # skip the silent SPY fetch
    progress_bar=False,
)

ratios = companies.ratios.collect_profitability_ratios()
roe    = companies.ratios.get_return_on_equity()
prices = companies.get_historical_data()
```

`enforce_source="FinancialModelingPrep"` **without** a key raises `ValueError` ✅ — a useful assertion
that your key actually loaded. The same argument is accepted per call on `get_historical_data`,
`get_treasury_data` and the four statement getters, where it overrides the constructor.

## What it is genuinely good at

- **Every formula is in the source and the docs**, so a disputed ratio is auditable rather than
  vendor-defined. The README's own example — MSFT's P/E quoted as 28.93 / 32.05 / 32.66 / 33.09 /
  33.66 / 33.67 / 33.80 / 34.4 by eight vendors on one day — is the argument for the whole project.
- Modules: `ratios`, `models` (Dupont, Altman-Z, Piotroski, WACC, enterprise value), `technicals`,
  `risk` (VaR, ES, EVaR), `performance` (CAPM, Fama-French factors, Jensen's alpha), `fixedincome`,
  `economics` (OECD), `options` (Black-Scholes + greeks), and a standalone `Discovery` screener.
- Caching is incremental — widening a date range fetches only the missing years ✅.
- Ships an **MCP server** (`uvx --from "financetoolkit[mcp]" financetoolkit-mcp`) and a `.mcpb`
  bundle ✅ — it needs an FMP key, so the free Yahoo path is not exposed through MCP.

## Cross-references

`yfinance.md` (the underlying free source and its survivorship problem) ·
`_decision-table.md` (FMP free-tier row, licence table) ·
`../../fundamental-and-macro-data/references/edgartools.md` (as-filed, point-in-time alternative) ·
`../../portfolio-and-risk/references/risk-measures.md` before trusting its VaR/ES defaults.
