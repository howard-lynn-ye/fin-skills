---
name: lib-edgartools
description: >-
  edgartools is the default free SEC EDGAR client - typed objects for 20+ form types, XBRL
  statements, no API key - and it 403s on every request until you call set_identity(). TRIGGER -
  edgartools, "from edgar import Company, set_identity", set_identity, Company("AAPL"),
  get_filings, filing.xbrl(), get_facts, get_financials, accession number, CIK, ticker-to-CIK,
  10-K, 10-Q, 8-K, 13F, Forms 3/4/5, EDGAR full-text search, "HTTP 403" from sec.gov, SEC
  User-Agent required, SEC 10 requests per second, edgartools MCP server, edgartools[ai]. 442
  releases have moved the API repeatedly and the classifier is still Beta, so any snippet recalled
  from memory is probably wrong for the installed version. SKIP for lib-fredapi, which is the
  skill for macro series and revisions. SKIP when the question is WHICH library to choose rather
  than how to use this one - that belongs to the domain skill.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# edgartools

The default Python library for SEC EDGAR, and the only **free point-in-time US fundamentals** source
in this repo — every EDGAR filing is as-filed, so a 2018 10-K is what a 2018 investor actually saw.

| | |
|---|---|
| pip / import | `edgartools` / `from edgar import Company, set_identity` |
| Version | 5.56.0 (2026-09-02) · **442 releases** · repo pushed 2026-09-04 |
| Licence | MIT (PyPI `license_expression` + classifier + GitHub `spdx_id` all agree) |
| Python | `>=3.10` · classifier is still `Development Status :: 4 - Beta` at 5.56.0 |
| Status | The fastest-moving library in this catalog. 2,651★, 46 open issues |

## The trap that costs you money

**`set_identity()` is mandatory, and its absence is a 403, not a helpful error.** The SEC enforces a
User-Agent on every request: **no User-Agent → HTTP 403**, any UA → 200. The header is checked for
*presence*, not content — and sending a fake browser UA is a policy violation that gets IPs blocked
in practice. Call `set_identity("Your Name your@email.com")` once at import, before anything else.

The companion limit: **the SEC caps you at 10 requests/second.** edgartools is rate-limit aware, but
a parallel loop over thousands of CIKs will still get you throttled or IP-blocked.

## Point-in-time is a property of the filing, not of every accessor

Reading a 2018 10-K gives you 2018-as-filed. But the XBRL **frames** API, and any "latest value for
concept X in FY2018" query, return the figure **as it stands today, after restatements** — the frames
endpoint is not point-in-time and cannot be made so.

**If your backtest needs "what was knowable on date D", iterate filings and filter by the filing's
`filing_date`, never by fiscal period alone.**

One more survivorship note: `sec.gov/files/company_tickers.json` is a **current snapshot**. The
*filings* for Lehman, Bear Stearns and Enron are all intact, so EDGAR itself is bias-free — the
convenience ticker list is not. Build historical universes from filings, not from that file.

## "XBRL-standardized" is a mapping, and mappings have opinions

Cross-company comparison works because edgartools maps issuer-specific us-gaap tags and custom
extension tags onto a common statement layout. That is exactly the step where a company's genuinely
unusual line item gets folded into a standard bucket, or dropped.

For anything load-bearing, reconcile against `Company(...).get_facts()` — raw us-gaap concepts, no
remapping — or against the filing's own rendered statement. The convenience is real; the convention
is edgartools', not the issuer's.

## 442 releases means your memory of the API is wrong

The library ships several versions a month and the surface has moved repeatedly:
`Company(...).financials` versus `.get_financials()`, `Filing.obj()` return types, the XBRL query
surface. **Any example older than a few weeks — including anything a model produces from memory —
must be checked against the installed version before you trust it.** The
`Development Status :: 4 - Beta` classifier at 5.56.0 is honest, not stale.

It is also not a light import: **22 hard runtime dependencies**, including `httpx`, `lxml`,
`pyarrow>=17`, `pydantic>=2`, `pandas>=2`, `rich`, `orjson`, `nest-asyncio`, `truststore`,
`rapidfuzz`, `textdistance` and `rank-bm25`. The built-in **MCP server is an extra**
(`pip install edgartools[ai]`), not a default — and it is the MIT path to EDGAR-over-MCP, where the
widely-used third-party SEC EDGAR MCP server is AGPL-3.0.

## Licence note on the neighbours

Every other EDGAR library in this space is MIT or Apache-2.0 except **`edgar-crawler`, which is
GPL-3.0** and is not on PyPI (clone only). Its 10-K item-section extraction is good and there is a
WWW 2025 paper behind it, which is precisely why people vendor it into pipelines without checking.
**Run it as a separate command-line step producing JSON, or reimplement the extraction — do not
import it.** `python-xbrl` is dead (1.1.1, 2016-12-27); `sec-parser` is dormant (2024-06);
`sec-edgar-downloader` is maintained but downloads raw files only, with no parsing and no XBRL.

## Minimal correct call

```python
from edgar import Company, set_identity

set_identity("Research Desk you@example.com")     # REQUIRED — omit it and every request 403s

c = Company("AAPL")

# Point-in-time: iterate FILINGS and key off the filed date, not the fiscal period
for f in c.get_filings(form="10-K").head(5):
    print(f.filing_date, f.accession_no)           # filing_date is the knowability date
    xb = f.xbrl()                                  # this filing's numbers, as filed

# Convenience path — standardized across companies, but remapped
c.get_financials().income_statement()

# Raw concepts, no remapping — use this to reconcile anything load-bearing
c.get_facts().query().by_concept("Revenues").to_dataframe()
```

## See also

- `../../../fin-core/skills/fundamental-and-macro-data/SKILL.md` — the point-in-time join rules
- `../../../fin-core/skills/fundamental-and-macro-data/references/edgartools.md` — the verified
  reference card
- `../../../fin-core/skills/fundamental-and-macro-data/references/sec-edgar.md` — raw endpoints, the
  10 req/s limit, the frames `I`-suffix trap, and the DERA bulk ZIPs
- `../lib-fredapi/SKILL.md` — macro series and their vintages

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`fundamental-and-macro-data`** (`../../../fin-core/skills/fundamental-and-macro-data/SKILL.md`).

