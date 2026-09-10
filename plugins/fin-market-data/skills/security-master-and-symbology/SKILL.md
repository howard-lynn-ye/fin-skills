---
name: security-master-and-symbology
description: >-
  Map ticker, CIK, ISIN, FIGI, SEDOL and CUSIP on (identifier, DATE) rather than on identifier,
  and detect when the entity behind one changed. TRIGGER - ticker to CIK, ISIN to CUSIP, FIGI,
  SEDOL, ISIN or CUSIP check digit, identifier validation; a reused ticker, a renamed company, a
  merger or ticker change breaking a join; security master, symbology, cross-reference table,
  PERMNO, entity resolution; "the fundamentals attached to the wrong company"; building a universe
  from company_tickers.json. Load before any join keyed on a symbol - an identifier is not an
  entity, neither is stable, and the SEC's own name windows both overlap and leave gaps. SKIP for
  FINDING an identifier you do not have yet (finding-and-searching-data), for split/dividend
  adjustment and delisted price history (market-data-sourcing), for the point-in-time vintage of
  the fundamentals (fundamental-and-macro-data), for auditing a finished result
  (research-integrity-guards), and for A-share code changes (china-ashare-data).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-10"
---

# Security master and symbology

The join everyone writes once and never revisits. It has two independent failure modes, and only
the first one is about arithmetic.

## 1. Check digits — cheap, and they do not mean what you think

ISIN, CUSIP, SEDOL and FIGI each carry a self-check digit. All four are pure arithmetic over the
same base-36 alphabet, cost nothing, and catch transcription errors a string join cannot see.

| Scheme | Length | Algorithm | Blind spot |
|---|---|---|---|
| ISIN | 12 | letters expand to two-digit values **first**, then Luhn from the right | adjacent `0<->9` swap |
| CUSIP | 9 | double every second character from the left, fold digits | adjacent `0<->9` swap |
| SEDOL | 7 | weights `(1,3,1,7,3,9)`, `(10 - sum mod 10) mod 10` | — no vowels, so O/0 and I/1 typos are also caught by shape |
| FIGI | 12 | Luhn over base-36 values | adjacent `0<->9` swap |

🚨 **Expanding letters after doubling** is the classic wrong ISIN implementation. It agrees with the
correct one often enough to survive a smoke test.

✅ Verified against identifiers read out of filings on sec.gov on 2026-09-10 — Apple
`US0378331005` / `037833100`, IBM `US4592001014` / `459200101` / FIGI `BBG000BLNNH6`, Amazon
`US0231351067` / `023135106` / `BBG000BVPV84`, BAE Systems `GB0002634946` / CINS `G06940103`,
Vodafone SEDOL `BH4HKS3`. `scripts/identifier_checks.py` validates all fifteen, and
`tests/test_core_identifier_checks.py` re-validates them plus deliberately corrupted variants.

🚨 **A passing check digit means "not obviously mistyped" and nothing more.** Three errors survive
it: a well-formed identifier that does not exist, one that exists and belongs to a different
security than you think, and — because ISIN, CUSIP and FIGI are Luhn-family — an adjacent `0<->9`
transposition, which Luhn provably cannot detect. ✅ Measured by this repo's own test suite: NXP's
real ISIN `NL0009538784` and the typo `NL0090538784` **both validate**.

### 🚨 Never derive one identifier from another

The folklore is "an ISIN is a country code + the CUSIP + a check digit". ✅ Measured over one SEC
N-PX proxy table (CIK 1719812, accession `0001162044-25-000794`, retrieved 2026-09-10): 1,018 rows,
75 distinct `(cusip, isin, name)` triples, **0 check-digit failures on either scheme**, and
**74 of 75 (98.7%)** ISINs do embed the CUSIP as their NSIN.

The remaining 1.3% is the whole point. **NXP Semiconductors NV** carries CUSIP `N6596X109` (a CINS
— the international CUSIP form, letter-led) against ISIN `NL0009538784`. Slicing `isin[2:11]` gives
`000953878`, a well-formed-looking string that is not NXP's identifier under any scheme and will
match nothing, or worse, something. `cusip_from_isin()` validates the slice and returns `None` when
it does not check out. 98.7% is exactly the rate that makes a bug ship.

## 2. 🚨 The real trap: an identifier is not an entity, and neither is stable

### CIK 933136 — six registrants, one continuous filing history

✅ Retrieved from `data.sec.gov/submissions/CIK0000933136.json` on 2026-09-10:

| Window (`from`, `to` as filed) | Registrant |
|---|---|
| 1995-03-29 → 2006-10-12 | WASHINGTON MUTUAL INC |
| 2006-09-19 → 2012-03-20 | WASHINGTON MUTUAL, INC |
| 2012-01-25 → 2015-05-08 | WMI HOLDINGS CORP. |
| 2015-05-13 → 2018-09-14 | WMIH CORP. |
| 2018-10-10 → 2025-10-01 | Mr. Cooper Group Inc. |
| current | **Maverick Merger Sub 2, LLC** |

A failed thrift, an empty bankruptcy shell, a mortgage servicer, and now a merger vehicle — one
CIK. ✅ Its `companyfacts` `Assets` series is **123 facts across 49 accessions**, period ends
**2012-12-31 → 2025-06-30**: $339,916,000 in the first (the shell) to $18,499,000,000 in the last
(the servicer). Keyed on the CIK alone, that is one continuous balance-sheet history of an entity
that changed three times. ⚠️ Note the correction to the version of this story circulating in
this repo's own research notes: XBRL began ~2009 and WaMu died in 2008, so `companyfacts` does
**not** reach the thrift. The *filing* history does — 1,366 filings from 1995-01-25 to 2008-06-17
sit under the same CIK.

🚨 And the two SEC APIs now disagree about the current name: on 2026-09-10 the Submissions API says
`Maverick Merger Sub 2, LLC` while `companyfacts` still reports `entityName: "Mr. Cooper Group
Inc."`. There is no single authoritative "name of this CIK".

### Renaming is normal, not exotic

✅ Measured 2026-09-10 over a seeded random sample of **400 CIKs** (`random.Random(0).sample`) drawn
from `company_tickers.json`, each fetched from the Submissions API:

- **194 of 400 (48.50%)** carry at least one `formerNames` record. Nearly half of currently listed
  filers have changed their registered name.
- Names per CIK: 206 have one, 122 have two, 50 three, 15 four, 5 five, one has seven and one has
  eight.
- Of the **72** with more than one former name, **14 (19%) have OVERLAPPING windows.**

### 🚨 `resolve()` must return a list, because the primary source gives 0, 1 or 2 answers

Apple's own record (`CIK0000320193.json`, 2026-09-10) has both pathologies:

```
APPLE COMPUTER INC       1994-01-26 -> 2007-01-04
APPLE COMPUTER INC/ FA   1997-07-28 -> 1997-07-28     <- from == to, nested inside the above
APPLE INC                2007-01-10 -> 2019-08-05
Apple Inc.               (current, no start date anywhere in the file)
```

- **1997-07-28 has TWO registrant names.** Code that assumes one silently takes whichever sorted
  first.
- **2007-01-05 to 2007-01-09 has NONE** — five days covered by no name at all. Returning the
  nearest window instead of nothing is an invention.
- ⚠️ Two conversions the SEC's JSON forces and does not document: `to` is the **last day** the name
  was in effect, not an exclusive bound (read as exclusive, the `from == to` record becomes a
  zero-width window that covers nothing and vanishes); and the **current name has no start date**,
  so its window opens where the newest former one closes — an inference, and this library labels it
  as one.
- ⚠️ The window stamps mix conventions: over that 400-CIK sample the `from`/`to` values carry three
  distinct times of day — `04:00:00Z` (316), `05:00:00Z` (188) and `00:00:00Z` (96), i.e. midnight
  Eastern in two DST states plus midnight UTC. Truncate to the date; never compare two of them as
  instants.

### 🚨 Ticker → CIK is many-to-one, and the extras are not shares

✅ Measured on `sec.gov/files/company_tickers.json`, retrieved 2026-09-10: **10,407 rows, 8,013
distinct CIKs, 1,441 of them (17.98%) carry more than one ticker.** 544 tickers (5.2%) contain a
dash.

| CIK | Filer | Tickers | Includes |
|---|---|---|---|
| 927971 | BANK OF MONTREAL | **33** | `BERZ`, `BNKD`, `BNKU`, `AIQD`, `AIQU` — leveraged **ETNs** |
| 1026214 | FEDERAL HOME LOAN MORTGAGE | 25 | `FMCCG`, `FMCCH`, ... preferred series |
| 70858 | BANK OF AMERICA | 17 | `BML-PG`, `MER-PK` — Merrill and BankAmerica legacy preferreds |
| 19617 | JPMORGAN CHASE | 9 | `JPM-PC` preferred, `VYLD` and `AMJB` **ETNs** |
| 1652044 | Alphabet | 4 | `GOOG`, `GOOGL`, `GOOGM`, `GOOGN` |

A naive ticker → CIK → fundamentals join attaches **Bank of Montreal's balance sheet to a 3x
bank-sector ETN**. Filter with `company_tickers_exchange.json` and drop the `-P*` lines, and even
then confirm the security type rather than assuming.

### 🚨 `company_tickers.json` contains no dates, so it cannot answer a historical question

The file has no date field of any kind. The only honest validity window for a mapping built from it
opens at the moment you downloaded it. `from_company_tickers()` therefore **requires** a
`retrieved_at` and refuses to invent one, and the consequence is intended:

```python
from fin_skills.discovery import SecurityMaster, from_company_tickers, Scheme
m = SecurityMaster(from_company_tickers(rows, retrieved_at="2026-09-10"))
m.resolve("JPM", Scheme.TICKER, "2008-01-02")     # []  <- correct, and it is a real answer
```

To cover history you must assert the evidence, and `widen(a, start=..., why=...)` records the
assertion so a reviewer can disagree with it. ✅ The same file is survivorship-biased in the other
direction: Lehman, Bear Stearns and Enron all return an empty ticker list, so a universe built from
it re-introduces exactly the bias EDGAR's filings would have avoided
(`fundamental-and-macro-data` §2.5).

## 3. Using it

```python
from fin_skills.discovery import SecurityMaster, from_sec_former_names, Scheme

master = SecurityMaster(from_sec_former_names("933136", "Maverick Merger Sub 2, LLC", former))
[a.name for a in master.resolve("933136", Scheme.CIK, "2007-06-29")]  # ['WASHINGTON MUTUAL, INC']
[a.name for a in master.resolve("933136", Scheme.CIK, "2019-12-31")]  # ['Mr. Cooper Group Inc.']

master.overlaps()      # windows of one identifier that both cover some date
master.gaps()          # stretches where resolve() legitimately returns nothing
```

`resolve_one()` raises `AmbiguousIdentifier` when two entities claim the date and
`UnknownIdentifier` when none does — never the nearest window.

### The guard: fail when a mapping is used outside its window

```python
report = master.check_usage([{"identifier": "JPM", "scheme": "ticker", "as_of": "2008-01-02"}])
report.passed          # False
print(report.summary())
# FAIL  symbology_windows  1 usage(s), 1 outside their window
#   error: ticker:JPM @ 2008-01-02: used outside every validity window; this identifier is
#          mapped over [2026-09-10, open) -> cik:0000019617
```

It fails on four things: a usage outside every window, an identifier the master has never seen (an
*unverified* join key, not merely an unmatched one), a date claimed by two entities, and a
malformed identifier. It warns when the only mapping came from a dateless snapshot.

## 4. Checklist

1. **Validate the shape** — free, and it catches typing, not truth.
2. **Never derive one scheme from another.** 98.7% is a bug waiting to ship.
3. **Key every mapping on `(identifier, date)`**, half-open `[start, end)`.
4. **Choose a stable surrogate you control** (`Scheme.LOCAL`, or a PERMNO if you license CRSP).
   Every borrowed identifier can be reassigned by whoever issues it.
5. **Handle zero and two answers.** The primary source produces both.
6. **Run `check_usage()` on your join keys** before the join, not after the result looks odd.

## Scripts

- `scripts/identifier_checks.py` — the four check digits, `cusip_from_isin()`, the SEC name-window
  arithmetic (`name_windows`, `window_overlaps`, `window_gaps`, `label_as_of`), and
  `multi_ticker_rate()` / `embedding_rate()` as functions over a supplied file, so every number
  above is reproducible on today's copy rather than trusted from this page. Offline, no writes.

## ❓ Not verified

The SEDOL weighted-modulus constants and the FIGI Luhn-over-base-36 rule are implemented from the
published algorithms and validated only by the identifiers cited above — no LSE or OpenFIGI
specification document was fetched, so ⚠️ both algorithms are secondhand even though the examples
pass. The claim that `to` in `formerNames` is inclusive is an inference from Apple's
`from == to` record and the 2007 gap; the SEC documents neither. No FIGI/SEDOL/PERMNO mapping is
shipped or endorsed here — this library ships no data. Whether a ticker meant the same security on
a past date is, in the end, not answerable from any free source: that is what a paid security
master is for, and the honest interim answer is an empty `resolve()`.
