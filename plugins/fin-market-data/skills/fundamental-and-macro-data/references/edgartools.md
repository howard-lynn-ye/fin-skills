# edgartools

The default Python library for SEC EDGAR: typed objects for 20+ form types, XBRL-standardized
financial statements, no key and no quota tier. It is also the only **point-in-time** US fundamentals
source in this repo that costs nothing — every EDGAR filing is as-filed, so a 2018 10-K is what a
2018 investor actually saw.

| Field | Value |
|---|---|
| pip / import | `edgartools` / `from edgar import Company, set_identity` |
| version | **5.56.0 (2026-09-02)** · **442 releases** ✅ |
| GitHub | `dgunning/edgartools` — **2,651★**, 473 forks, 46 open issues, pushed 2026-09-04 ✅ |
| Licence | **MIT** ✅ (PyPI `license_expression` + classifier + GitHub `spdx_id` all agree) |
| Python | `>=3.10` ✅ · classifier still `Development Status :: 4 - Beta` at 5.56.0 ⚠️ |
| Verdict | ✅ **the default, and the fastest-moving library in this skill** — pushed the day of this audit |

Verified 2026-09-04 via the PyPI JSON API, the GitHub REST API, and the repository README.

## 🚨 Traps

🚨 **`set_identity()` is mandatory and its absence is a 403, not a helpful error.** The SEC enforces
a User-Agent on every request. See `sec-edgar.md`: **no User-Agent → HTTP 403**, any UA → 200 — the
header is checked for *presence*, not content, and sending a fake browser UA is a policy violation
that gets IPs blocked in practice. Call `set_identity("Your Name your@email.com")` once at import.

🚨 **442 releases means model-recalled snippets are usually wrong.** The library ships several
versions a month and the API has moved repeatedly (`Company(...).financials` vs
`.get_financials()`, `Filing.obj()` return types, XBRL query surface). Any example older than a few
weeks — including anything an LLM produces from memory — must be checked against the installed
version before you trust it. The `Development Status :: 4 - Beta` classifier at 5.56.0 is honest.

🚨 **"XBRL-standardized" is a mapping, and mappings have opinions.** Cross-company comparison works
because edgartools maps issuer-specific us-gaap tags and custom extension tags onto a common
statement layout. That is exactly the step where a company's genuinely unusual line item gets folded
into a standard bucket, or dropped. For anything load-bearing, reconcile against
`Company(...).get_facts()` (raw us-gaap concepts, no remapping) or the filing's own rendered
statement. The convenience is real; the convention is edgartools', not the issuer's.

🚨 **Point-in-time is a property of the *filing*, not of every accessor.** Reading a 2018 10-K gives
you 2018-as-filed. But the XBRL **frames** API and any "latest value for concept X in FY2018" query
return the figure **as it stands today, after restatements** — see `sec-edgar.md`, which flags the
frames endpoint as not point-in-time. If your backtest needs "what was knowable on date D", filter
by the filing's `filed` date, never by fiscal period alone.

⚠️ **22 hard runtime dependencies** ✅ — `httpx`, `lxml`, `pyarrow>=17`, `pydantic>=2`, `pandas>=2`,
`rich`, `orjson`, `nest-asyncio`, `truststore`, `rapidfuzz`, `textdistance`, `rank-bm25` and more.
It is not a light import. The MCP server is an extra (`edgartools[ai]`), not a default.

⚠️ **SEC rate limit is 10 requests/second** ✅ (SEC webmaster FAQ). edgartools is rate-limit aware,
but a parallel loop over thousands of CIKs will still get you throttled or IP-blocked.

⚠️ **`sec.gov/files/company_tickers.json` is a current snapshot — survivorship-biased** ✅. The
*filings* for Lehman, Bear Stearns and Enron are all intact, so EDGAR itself is bias-free; the
convenience ticker list is not. Build historical universes from filings, not from the ticker file.

## Minimal correct snippet

```python
from edgar import Company, set_identity

set_identity("Research Desk you@example.com")     # 🚨 required — omit it and every request 403s

c = Company("AAPL")

# Point-in-time: iterate FILINGS and key off the filed date, not the fiscal period
for f in c.get_filings(form="10-K").head(5):
    print(f.filing_date, f.accession_no)           # `filing_date` is the knowability date
    xb = f.xbrl()                                  # this filing's numbers, as filed

# Convenience path — standardized across companies, but remapped
c.get_financials().income_statement()

# Raw concepts, no remapping — use this to reconcile anything load-bearing
c.get_facts().query().by_concept("Revenues").to_dataframe()
```

## The alternatives, and why they are not the default

| Library | Version (date) | ★ | Licence | Verdict |
|---|---|---|---|---|
| **edgartools** | **5.56.0** (2026-09-02) | 2,651 | **MIT** ✅ | ✅ **the default** — typed objects, XBRL statements, built-in MCP server |
| `sec-edgar-downloader` | 5.1.0 (2026-02-02) | 717 | MIT ✅ | ✅ maintained, pushed 2026-06-22 — but **downloads raw files only**. No parsing, no XBRL, no objects. Correct choice *only* if you want a filing corpus on disk for your own NLP |
| `secedgar` | 0.6.0 (2025-05-09) | 1,412 | Apache-2.0 ✅ | ⚠️ **slowing** — no release in 16 months, last push 2025-12-09. Bulk/daily-index crawling; superseded for most uses |
| `sec-parser` | 0.58.1 (2024-06-09) | 295 | MIT ✅ | ⚠️ **dormant** — no release in 27 months. Parses filing HTML into a *semantic element tree*, which is genuinely distinct work; take the idea, expect to maintain the code |
| `python-xbrl` | 1.1.1 (**2016-12-27**) | 233 | Apache-2.0 ✅ | 🔴 **dead — last release nearly ten years ago.** Predates the current XBRL and inline-XBRL landscape entirely. Do not use |
| `edgar-crawler` | **not on PyPI** ✅ (404; no GitHub releases either — clone only) | 545 | 🚨 **GPL-3.0** ✅ | ⚠️ Extracts 10-K item sections into structured JSON (WWW 2025 paper); last push 2025-07-18. **Copyleft — importing it makes your distributed work GPL.** The only copyleft option in this table |

🚨 **`edgar-crawler` is the licence trap in this group.** Every other EDGAR library here is MIT or
Apache-2.0; edgar-crawler is GPL-3.0. Its item-section extraction is good and there is a real paper
behind it, which is precisely why people vendor it into pipelines without checking. Run it as a
separate command-line step producing JSON, or reimplement the extraction — do not import it.

⚠️ **The MCP ecosystem for EDGAR is AGPL-heavy.** The widely-used SEC EDGAR MCP server is AGPL-3.0,
as are the FRED and OpenBB ones. edgartools' own built-in MCP server (`pip install edgartools[ai]`)
is MIT and is the permissive path to the same data.

## Cross-references

`sec-edgar.md` — the raw endpoints, the 10 req/s limit, the frames `I`-suffix trap, and the DERA
bulk ZIPs · `fredapi.md` — macro series ·
`../../market-data-sourcing/references/financetoolkit.md` — 🚨 its FMP fundamentals are **restated,
not point-in-time**; EDGAR is the fix · `../../backtest-validation/SKILL.md` before backtesting on
any fundamental series.
