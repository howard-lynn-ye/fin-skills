# Re-verification queue for this skill

Network access failed partway through the research that produced `../SKILL.md`, so its claims sit in
two tiers. This file records exactly which ones need a second pass, so the ⚠️ markers in the skill
are a queue rather than a permanent hedge.

**Research date: 2026-09-04.** Everything below was still outstanding when the file was written.

## What is already citation-grade — do not re-do

✅ Fetched directly from the PyPI JSON API and the full PyPI simple index (884,814 names) before the
outage:

- every package version, release date, licence field, dependency pin and `requires_python` in
  `../SKILL.md` §0
- the two load-bearing quotes: `optopsy`'s README stating AGPL-3.0 and disclaiming assignment risk,
  and `py-vollib`'s description stating it "intentionally contains no library code"
- the keyword scan of `optopsy`'s README (`margin` 0, `exercise` 0, `settle` 0, `multiplier` 0)
- the negative result: no other options *backtester* exists on PyPI

## Ranked re-fetch list

| # | Fetch | Why it matters | Open conflict |
|---|---|---|---|
| ~~1~~ | ~~`optopsy`'s `LICENSE` file on GitHub~~ | **CLOSED 2026-09-05** — fetched `raw.githubusercontent.com/goldspanlabs/optopsy/main/LICENSE`: *GNU AFFERO GENERAL PUBLIC LICENSE Version 3*. AGPL confirmed; the GPL-3.0 search result was wrong | Also confirmed while here: `py-vollib` 1.0.12's wheel is 1,484 bytes, summary "Deprecated transition package for vollib" |
| 2 | OCC Rules PDF, **Rule 805(d)** and By-Laws **Article VI §11 / §11A** | The exercise-by-exception threshold and the contract-adjustment rules are the two things a backtest hard-codes | 🚨 **Pin a hashed copy.** The OCC Rules PDF is live-updated and returned two different Last-Modified dates for one URL, so an unhashed citation cannot be reproduced |
| 3 | `thetadata.net/pricing` | Cost decides whether it is an option at all | 🚨 **Irreconcilable**: $25/$60/$200 in one source against $80/$160 in another  — **tried 2026-09-05**: page is JS-rendered, static HTML carries no prices; needs a browser session |
| 4 | `massive.com/pricing?product=options` | Polygon renamed to Massive 2025-10-30; third-party pages describing its tiers contradict the vendor | Entry price, history depth, and whether options appear on the free tier are all disputed  — **tried 2026-09-05**: 319 KB of static HTML, no price strings; JS-rendered, needs a browser |
| 5 | `alphavantage.co/premium/` | This is the cheapest API-accessible full historical chain with greeks, so it is the recommendation most readers will act on | Dollar figures were not extractable  — **partial 2026-09-05**: static HTML shows monthly tiers at $149.99, $199.99, $249.99 (and higher), but the rate-limit ↔ price mapping is not parseable from static markup, and whether `HISTORICAL_OPTIONS` is served on a free key is still unconfirmed |
| 6 | Databento per-GB historical OPRA rates | The two-date history split (quotes 2023-03-28, trades 2013-04-01) is verified; the cost is not | — |
| ~~7~~ | ~~`DESCRIBE option_chain` on DoltHub~~ | **CLOSED 2026-09-05** — SQL API, branch `master` (`main` does not exist). Columns: `date, act_symbol, expiration, strike, call_put, bid, ask, vol, delta, gamma, theta, vega, rho`. **Open interest: absent. Volume: absent.** `vol` is `decimal(5,4)`, i.e. implied volatility | Both disqualifying unknowns resolved in the disqualifying direction. Recorded in `../SKILL.md` §6 table |
| 8 | Cboe RG08-73 and the SEC notice E8-7120 | The `$0.01 × multiplier` threshold and the units correction | Cboe and OIC material both print the units error being corrected, so the primary text is the only settlement |
| 9 | GitHub stars and last-commit dates for every package in §0 | Staleness is the claim, and star counts came back search-rounded | Treat the numbers in the research as indicative, not exact |

## How to close an item

Re-fetch, then in `../SKILL.md` change the claim's ⚠️ to ✅ and move
`metadata.verified_on` forward. If the primary source **disagrees** with what the file says, that is
the more valuable outcome — correct it and say what changed, per `CONTRIBUTING.md`.

If a re-fetch shows a claim was wrong, it belongs in the commit message, not only in the diff. This
repo has reversed several claims that way already.
