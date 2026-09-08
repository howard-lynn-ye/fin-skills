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
| ~~2~~ | ~~OCC Rules PDF and By-Laws~~ | **CLOSED 2026-09-05** — both fetched with curl (urllib gets 403) and hashed. `occ_rules.pdf` (getmedia/9d3854cd-b782-450f-bcf7-33169b0576ce): 1,492,449 bytes, Last-Modified 2026-08-26 20:50:55 GMT, sha256 `facaed0569bc45f3a7c1872a0be33f6365a69daf094107c720a8b91f53043a9e`. `occ_bylaws.pdf` (getmedia/3309eceb-56cf-48fc-b3b3-498669a24572): 1,658,160 bytes, Last-Modified 2026-04-24 15:41:11 GMT, sha256 `812f5e4c43de81ccda1c165b43115757e4c320f088da7720616dacdca0548c62`. Quoted verbatim in `../SKILL.md`: Rule 805(d)(2) incl. the multiple-applied-to-closing-price clause; Rule 1804(c) ($1.00 per contract at multiplier ≠ 1, $0.01 at multiplier 1); Interpretation .02 (administrative, not binding on customer accounts); By-Laws Art. VI §3A(a)(3) ($.125 per unit floor) and Interp. .01 (policy-or-practice, regardless of size). Rule 805 last amended 2025-12-31 (SR-OCC-2025-017) | The research cited Art. VI §11/§11A for adjustments; in this copy the dividend rule is in §3A. §11/§11A were not read. Cite the full hashes |
| ~~3~~ | ~~`thetadata.net/pricing`~~ | **CLOSED 2026-09-05** (browser): Value **$40**, Standard **$80**, Pro **$160** per month; 4 / 8 / 12 years of history by tier | The $25/$60/$200 source was wrong; $80/$160 was right minus the entry tier |
| ~~4~~ | ~~`massive.com/pricing?product=options`~~ | **CLOSED 2026-09-05** (browser): Basic **$0** (5 calls/min, 2 yrs, EOD), Starter **$29**, Developer **$79** (4 yrs), Advanced **$199** (5+ yrs, real-time) | Matches the research agent's vendor-page reading exactly; the third-party claims were the wrong ones |
| 5 | `alphavantage.co/premium/` | This is the cheapest API-accessible full historical chain with greeks, so it is the recommendation most readers will act on | **Partial 2026-09-05** (browser + curl): free key = **25 requests per day** (vendor text); premium tiers seen at $149.99 / $199.99 / $249.99 but the rate-limit mapping is in a form widget, not page text. Still open: whether `HISTORICAL_OPTIONS` is served on a free key |
| 6 | Databento per-GB historical OPRA rates | The two-date history split (quotes 2023-03-28, trades 2013-04-01) is verified; the cost is not | **Partial 2026-09-05** (browser): subscription tiers confirmed — Standard $199/mo, Plus $1,750/mo, Unlimited $4,500/mo; $125 credit, 6-month expiry. Per-GB is only in the interactive estimator |
| ~~7~~ | ~~`DESCRIBE option_chain` on DoltHub~~ | **CLOSED 2026-09-05** — SQL API, branch `master` (`main` does not exist). Columns: `date, act_symbol, expiration, strike, call_put, bid, ask, vol, delta, gamma, theta, vega, rho`. **Open interest: absent. Volume: absent.** `vol` is `decimal(5,4)`, i.e. implied volatility | Both disqualifying unknowns resolved in the disqualifying direction. Recorded in `../SKILL.md` §6 table |
| ~~8~~ | ~~Cboe RG08-073~~ | **CLOSED 2026-09-05** — `cdn.cboe.com/resources/regulation/circulars/regulatory/RG08-073.pdf`, 24,066 bytes, sha256 `939825b597f0943ab814ba1cfa4e1e6bff38c5db95430241a165ca56c22f918a`, circular dated June 13, 2008: threshold *"from $.05 to $.01 in a clearing member's customer, firm, and market maker account"*, *"effective for the June 2008 expiration, which is Saturday, June 21st"* | SEC notice E8-7120 not fetched; the Cboe circular plus OCC Rule 805(d)(2) and 1804(c) (item 2) are the operative primary texts and all are hashed |
| ~~9~~ | ~~GitHub stars and last-commit dates~~ | **CLOSED 2026-09-05** via PyPI `project_urls` → `gh api` (no guessed repo names): lean-cli 326★ / 2026-09-04; optopsy 1,470★ / last commit 2026-04-02; backtrader 23,156★ / 2024-08-19; vectorbt 9,026★ / 2026-08-02; opstrat 168★ / 2024-01-11; py_vollib_vectorized 162★ / 2024-12-02; vollib 435★ / 2026-05-29 | optionlab, mibian, wallstreet, Optlib publish no GitHub URL on PyPI — left unverified rather than guessed. **Found while here:** `michaelchu/optopsy` redirects to `goldspanlabs/optopsy` (one repo, transferred), and its LICENSE commit log shows the GPL-3.0 → AGPL-3.0 switch on **2026-02-23** |

## How to close an item

Re-fetch, then in `../SKILL.md` change the claim's ⚠️ to ✅ and move
`metadata.verified_on` forward. If the primary source **disagrees** with what the file says, that is
the more valuable outcome — correct it and say what changed, per `CONTRIBUTING.md`.

If a re-fetch shows a claim was wrong, it belongs in the commit message, not only in the diff. This
repo has reversed several claims that way already.
