---
name: asean-markets
description: >-
  TRIGGER - ASEAN, SGX, Singapore, Bursa Malaysia, SET Thailand, Thai foreign board or NVDR, IDX Indonesia, PSE Philippines, HOSE HNX UPCoM Vietnam, foreign ownership room, Southeast Asian sessions and settlement. SKIP for Japan (japan-markets), Hong Kong (hong-kong-markets), India (india-markets), Korea or Taiwan (korea-taiwan-markets), mainland China (china-trading-stack), and regional stack selection (asia-pacific-markets).
license: MIT
compatibility: Python 3.10+; offline scripts require numpy and pandas.
metadata:
  version: "0.1.0"
  verified_on: "2026-09-14"
allowed-tools: Read Bash(python:*)
---

# ASEAN markets

Foreign ownership room, a foreign share line and an executable offer are different facts.
A price series can be economically inappropriate for a foreign investor even when every
return calculation is correct. Start with the security, investor category and venue.

## Session and settlement routing

The table gives local-time regular-day windows used in the offline routing example.
These include auction time in some venues; the resulting minute count is not a guarantee
of continuous executions. Special calendars and security-specific mechanisms remain separate.

| Market | Snapshot windows | Segments to distinguish | Verification scope |
|---|---|---|---|
| SGX | 09:00–12:00, 13:00–17:00 | Mainboard / Catalist | ✅ SGX rulebook checked 2026-09-14: lunch exists |
| Bursa | 09:00–12:30, 14:30–17:00 | Main / ACE / LEAP | ✅ Bursa ground rules checked 2026-09-14 |
| SET | Nominal 10:00–12:30, 14:00–16:30 | SET / mai; local, foreign and NVDR lines | ✅ SET checked 2026-09-14; actual opens random within preceding five minutes |
| IDX | 09:00–12:00, 13:30–15:50; Friday 09:00–11:30, 14:00–15:50 | Main / Development / Acceleration / New Economy; Watchlist mechanisms differ | ⚠️ Retained 2026-09-10 audit; current source unavailable |
| PSE | 09:30–12:00, 13:00–14:45 before pre-close | Main / SME | ⚠️ Official page checked 2026-09-14 contains conflicting schedules |
| Vietnam | HOSE snapshot 09:00–11:30, 13:00–15:00 | HOSE and HNX are separate venues; UPCoM has separate rules | ⚠️ Retained 2026-09-10 audit; do not reuse HOSE hours for every segment |

Sources: [SGX rulebook archive](https://rulebook.sgx.com/rulebook/archive),
[Bursa ground rules](https://www.bursamalaysia.com/sites/5bb54be15f36ca0af339077a/assets/5cd16a4f5b711a52c3201c37/BM-Index-Series-Ground-Rules.pdf),
[SET trading hours](https://www.set.or.th/en/market/information/trading-procedure/trading-hours),
[IDX mechanism](https://www.idx.co.id/en/trading/trading-hours-and-mechanism/), and
[PSE investing page](https://www.pse.com.ph/investing-at-pse/).
The IDX route returned **503 in a browser** as well as failing programmatic access on
2026-09-14. PSE's detailed table shows the window above while older prose on the same
page says 13:30–15:30. Obtain a dated exchange notice before building a trading calendar;
the helper exposes the table snapshot for research routing only.

✅ T+2 transition notices re-read 2026-09-14: SGX **2018-12-10**
([rule 9.1A.1](https://rulebook.sgx.com/rulebook/91a1)), Bursa **2019-04-29**
([transition circular](https://www.bursamalaysia.com/sites/5bb54be15f36ca0af339077a/assets/5cc04de639fba211ef1f1a59/1T2Settlement-POsCircular.pdf)),
SET **2018-03-02** ([NVDR amendments](https://www.set.or.th/nvdr/en/info/letter.html)),
and PSE **2023-08-24** ([SCCP announcement](https://www.pse.com.ph/sccps-t2-settlement-cycle-goes-live-on-august-24/)).
⚠️ IDX's **2018-11-26** transition remains an earlier-audit input. The helper switches
these covered modern regimes from T+3 to T+2 and accepts explicit holidays. It is not
a custodian calendar. Vietnam requires an explicit settlement lag: its 2022 change to
intraday securities delivery must not be mislabelled as the original T+2 introduction.

## Foreign room does not imply a universal cap or price premium

✅ [SET investor guide](https://www.set.or.th/en/trade-in-set) and
[foreign-investor handbook](https://media.set.or.th/set/Documents/2022/Sep/SET_Foreign_Handbook.pdf),
re-read 2026-09-14: local shares held by foreigners, registered foreign shares and NVDRs
do not confer identical rights. NVDRs generally preserve financial benefits without voting
rights; their eligibility and issuer-specific limits require separate checks. That investor
page also contains stale settlement prose, so use the dated settlement notice above.

Do not hardcode a country-wide ownership limit from one company's limit. Sector statutes,
issuer articles, investor classification and current foreign holdings matter. Remaining room
constrains a purchase that increases aggregate foreign ownership; an eligible transfer of an
existing foreign holding may use a different route. A foreign board's existence does not
prove that anyone offers the required quantity. The script returns `None` when that capacity
is unknown, and its positive-room result passes only the ownership gate.

For a local price `L` and foreign-line premium `p`, a holding-period return is
`(L_exit / L_entry) * (1 + p_exit) / (1 + p_entry) - 1`.
A constant premium cancels. Deducting its level again from the foreign-line return double
counts the effect. A real foreign line can trade at a discount, be stale or have no offers.

## Reproduce the numerical trap

Run `python scripts/asean.py`. No account, network, optional market package or live order is
used. Fixed seed **20260910**; all holdings and prices are synthetic. A chosen **49%** cap,
positive exponential premium and ownership drift are scenario inputs, not country rules or
estimated market behaviour. Measured output from 2026-09-14:

| Scenario output | Result |
|---|---:|
| Holding windows, 60 observations each | 1,190 |
| Mean foreign-minus-local percentage-return gap | 81.0 bp |
| Worst / best gap | −1,307.0 / 1,906.9 bp |
| No room for a new aggregate-ownership purchase | 71 / 1,250 observations (5.7%) |
| Fixed 20% premium at both endpoints | 0.0 bp return gap |

These overlapping windows do not estimate independent sampling uncertainty. The scenario
does not model FX, taxes, dividends, bid/ask spreads or available order size. No output is a
claim about actual foreign-investor performance. `session_minutes` likewise measures the
explicit snapshot windows, not trade counts.

## Data routes and handoff

Use exchange security masters, corporate-action notices, foreign-room reports and licensed
historical data for execution studies. SGX/Bursa/SET/IDX/PSE/HOSE web disclosures have
different publication formats and retrieval conditions; a public page is not automatically
a free realtime API or a complete delisted universe. Preserve original reports and timestamps.
Retail broker APIs need the relevant venue entitlement; symbol suffixes alone do not verify
the share line, rights or availability. Attach issuer-specific caps and eligibility to the
dated security master rather than a global country constant.

Regional routing: [asia-pacific-markets](../asia-pacific-markets/SKILL.md). For A/H and Connect
use [hong-kong-markets](../hong-kong-markets/SKILL.md); for mainland rules use
`china-trading-stack`. Daily price bands are also venue/product/date-specific: never infer
universal symmetric limits for this region from a single market.

- [Mainland trading](../../../fin-china/skills/china-trading-stack/SKILL.md): compare fill gates without copying local rules.
- [Market data](../../../fin-market-data/skills/market-data-sourcing/SKILL.md): source access, adjustments and provenance.
