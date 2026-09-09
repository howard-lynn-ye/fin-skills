---
name: libor-transition-and-fallbacks
description: >-
  Work out what a US dollar LIBOR contract actually falls back to under the LIBOR Act, and why
  the spread everyone quotes is the easy half. TRIGGER - LIBOR fallback, LIBOR transition,
  Regulation ZZ, 12 CFR 253, LIBOR Act, Board-selected benchmark replacement, tenor spread
  adjustment, 26.161 bp, 0.26161, 11.448 bp, 71.513 bp, ISDA 2020 IBOR Fallbacks Protocol,
  Fallback Rate (SOFR), CME Term SOFR, 30-day Average SOFR, 90-day Average SOFR, FFELP ABS
  fallback, FHFA-regulated-entity contract, consumer loan LIBOR fallback, "what does 3-month
  LIBOR become", "SOFR plus 26 bp", legacy LIBOR swap repapering, synthetic LIBOR. SKIP for
  computing a compounded SOFR coupon and its lookback conventions (sofr-and-rfr-compounding),
  for building an OIS curve (ois-discounting-and-multi-curve), and for US settlement and
  calendar rules (../../../fin-core/skills/us-market-rules).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# LIBOR transition and fallbacks

**Everyone remembers the five statutory tenor spread adjustments. Almost nobody remembers that
the base rate they attach to depends on what kind of contract it is.** "SOFR + 26.161 bp" is
ambiguous *as written in US law*: 12 CFR 253.4 gives **five different base rates for the same
spread**.

Every figure below is printed by `scripts/fallbacks.py` (runs in **0.3 s**; no optional
dependencies). ✅ Measured means this file produced it on 2026-09-09. Every rule is quoted from
**eCFR 12 CFR part 253 (Regulation ZZ)**, fetched 2026-09-09.

> **The rule: the tenor spread adjustment in 12 CFR 253.4(c) is the SAME for every contract
> type; the base rate is not. Name the contract type before you name the rate.**

## 1. ✅ The five statutory tenor spread adjustments — 12 CFR 253.4(c)

Verbatim: *"(1) 0.00644 percent for overnight LIBOR; (2) 0.11448 percent for one-month LIBOR;
(3) 0.26161 percent for three-month LIBOR; (4) 0.42826 percent for six-month LIBOR; and
(5) 0.71513 percent for 12-month LIBOR."*

| tenor | spread | bp | per quarter on 10,000,000 |
|---|---|---|---|
| overnight | 0.00644% | **0.644** | 161.00 |
| 1M | 0.11448% | **11.448** | 2,862.00 |
| **3M** | **0.26161%** | **26.161** | **6,540.25** |
| 6M | 0.42826% | **42.826** | 10,706.50 |
| 12M | 0.71513% | **71.513** | 17,878.25 |

- ✅ **12 CFR 253.2 covers only those five tenors.** The definition of LIBOR there says it
  *"Does not include the one-week or two-month tenors of U.S. dollar LIBOR."* 🚨 A 1-week or
  2-month LIBOR contract has **no** Board-selected replacement and is outside the statute.
- ✅ **The LIBOR replacement date is "the first London banking day after June 30, 2023"** —
  2023-07-03.

## 2. 🚨 One spread, FIVE base rates

✅ From 12 CFR 253.4:

| contract type | base rate for a 3-month LIBOR contract | observation window | cite |
|---|---|---|---|
| **derivative** | ISDA **Fallback Rate (SOFR)** | compounded **in arrears** over the period | 253.4(a)(1) |
| **non-consumer cash** | 3-month **CME Term SOFR** | forward-looking, set **in advance** | 253.4(b)(1)(ii) |
| **consumer loan** | 3-month CME Term SOFR | same, but the **spread ramps for one year** | 253.4(b)(2) |
| **FHFA-regulated entity** | 🚨 **30-day Average SOFR** — for *every* tenor | **backward**, 30 days ending at the reset | 253.4(b)(3)(i)(B) |
| **FFELP ABS** | 🚨 **90-day Average SOFR** for 3M; 30-day for 1M, 6M and 12M | **backward** | 253.4(b)(4) |

🚨 **Read the last two rows again.** A **six-month** LIBOR contract held by an FHFA-regulated
entity falls back to **30-day** Average SOFR + 42.826 bp: a six-month forward rate is replaced by
a thirty-day backward-looking average, and the spread is still the six-month one. The statute
says so.

✅ Four administrators publish those five rates (12 CFR 253.2, "Relevant benchmark
administrator"):

| base rate | administrator |
|---|---|
| Fallback Rate (SOFR) | Bloomberg Index Services Limited |
| CME Term SOFR | CME Group Benchmark Administration, Ltd. |
| the consumer-loan replacement | Refinitiv Limited — "USD IBOR Cash Fallbacks", "Consumer" |
| 30-day / 90-day Average SOFR | Federal Reserve Bank of New York |

⚠️ **CME Term SOFR is licensed and not redistributable**, which is why `fallback_rate` takes it
as an argument rather than computing it. The other four are free.

## 3. 🚨 What the base rate costs — measured

✅ One 3-month LIBOR contract, 10,000,000 notional, period **2026-07-01 → 2026-10-01**, on a
seeded SOFR path around 4.30% with a +25 bp step on 2026-06-18 and a −25 bp step on 2026-09-17.
CME Term SOFR is stood in with a **perfect forward** (4.540911%) — the friendliest possible
assumption for the cash contract:

| contract type | base | spread | all-in | window | vs derivative | cash |
|---|---|---|---|---|---|---|
| derivative | 4.5409% | 0.2616% | **4.8025%** | in arrears | — | — |
| non-consumer cash | 4.5409% | 0.2616% | 4.8025% | term, set in advance | +0.00 bp | 0.00 |
| **FHFA entity** | 4.4483% | 0.2616% | **4.7099%** | 30 days ending 2026-07-01 | 🚨 **−9.26 bp** | **−2,365.75** |
| **FFELP ABS** | 4.3791% | 0.2616% | **4.6408%** | 90 days ending 2026-07-01 | 🚨 **−16.18 bp** | **−4,134.13** |

✅ Over four quarterly resets across 2026:

| period | derivative | cash (term) | FHFA 30d | FFELP 90d | FHFA bp | FFELP bp |
|---|---|---|---|---|---|---|
| 2026-01-01 → 04-01 | 4.5837% | 4.5837% | 4.5525% | 4.5796% | −3.12 | −0.41 |
| 2026-04-01 → 07-01 | 4.6405% | 4.6405% | 4.5832% | 4.5837% | −5.73 | −5.68 |
| 2026-07-01 → 10-01 | 4.8025% | 4.8025% | 4.7099% | 4.6408% | −9.26 | −16.18 |
| **2026-10-01 → 27-01-01** | 4.6018% | 4.6018% | 4.7110% | 4.7975% | **+10.91** | 🚨 **+19.56** |
| **year total interest** | **472,227** | 472,227 | 470,422 | 471,554 | −1,805 | −673 |

- 🚨 **The largest single-reset gap is 19.56 bp — against a statutory spread adjustment of
  26.161 bp.** The base rate is worth as much as the number everyone argues about.
- 🔑 **The sign flips.** A backward-looking average lags: it is *below* the in-arrears rate after
  a hike and *above* it after a cut. That is why the year total nets down to −1,805 while single
  quarters are ±20 bp. **A hedge sized on the annual number will be wrong every quarter.**
- ✅ The cash contract matches the derivative here **only because Term SOFR was given perfect
  foresight**. A real forward-looking term rate does not know about an unscheduled move, and the
  gap is whatever the surprise was.

## 4. ✅ The consumer transition is a ramp, not a step — 12 CFR 253.4(b)(2)(i)

For a consumer loan, during the one-year period beginning on the LIBOR replacement date, the
spread *"transitions linearly for each business day"* from **the difference between the relevant
CME Term SOFR and the relevant LIBOR tenor determined as of the day immediately before the LIBOR
replacement date** to the statutory adjustment.

✅ The arithmetic, with an **illustrative** day-before difference of 15.0 bp (⚠️ not a quoted
value — the endpoint, 26.161 bp, is the statutory one) over 252 business days:

| business day | 0 | 63 | 126 | 189 | **252** | 300 |
|---|---|---|---|---|---|---|
| spread (bp) | 15.000 | 17.790 | 20.581 | 23.371 | **26.161** | 26.161 |
| quarterly effect on 10,000,000 | −2,790.25 | −2,092.69 | −1,395.12 | −697.56 | **0.00** | 0.00 |

🚨 **A consumer loan and an otherwise identical business loan carried different spreads on every
day between 2023-07-03 and 2024-07-03**, converging only at the end. Any historical repricing,
restatement or interest-recalculation over that window has to know which one it is holding.
✅ Reg ZZ also deems Refinitiv's published "USD IBOR Cash Fallbacks" for "Consumer" products
*equal* to those rates, so there is a published series and you should use it rather than
re-derive the ramp.

## 5. What Regulation ZZ does and does not do

- ✅ It applies **only where the contract has no workable fallback of its own** — 253.3(b) lists
  the exceptions, including a contract that already specifies a non-LIBOR replacement.
- ✅ **Benchmark replacement conforming changes** (253.5) become "an integral part of the LIBOR
  contract" automatically, and for a non-consumer contract the calculating person may make
  further technical changes in their reasonable judgment. **Two counterparties can therefore
  make different conforming changes to the same trade** and both be compliant.
- ✅ For derivatives the fallback is determined on the **derivative transaction fallback
  observation day**, and if the rate is unavailable then, *"the most recently available
  publication ... shall be used"* — a stale-rate rule that is in the regulation and not in most
  implementations.
- ⚠️ **This is US dollar LIBOR only.** Sterling and yen "synthetic LIBOR" ran under FCA powers on
  a different timetable and are not covered by anything here.

## 6. What the script gives you

`scripts/fallbacks.py` — numpy only at import; no optional libraries.

| Function | Does |
|---|---|
| `TENOR_SPREADS` / `tenor_spread(tenor)` | §1; raises on the excluded 1W and 2M tenors |
| `BOARD_SELECTED_BASES` / `BENCHMARK_ADMINISTRATORS` | §2, with the CFR citation on each row |
| `fallback_rate(contract_type, tenor, path, start, end, ...)` | §3 — the all-in rate by contract type |
| `fallback_comparison(...)` | the §3 table, in bp and in cash |
| `compounded_sofr(path, start, end, holidays)` | the one arithmetic behind all the SOFR bases |
| `consumer_transition_spread(day, transition_days, initial, statutory)` | §4 |
| `synthetic_sofr(..., steps=)` | the seeded path with policy steps |
| `LIBOR_REPLACEMENT_DATE`, `COVERED_TENORS`, `EXCLUDED_TENORS` | the dated constants |

## Where this sits

- `../sofr-and-rfr-compounding/SKILL.md` — how each of these SOFR bases is actually computed:
  the index, the averages, and 🚨 the four few-bp errors in compounding one.
- `../ois-discounting-and-multi-curve/SKILL.md` — repricing a legacy swap after the fallback,
  and 🚨 why the par rate check passes while the annuity is 1.461% wrong.
- `../yield-measures-and-bill-quotes/SKILL.md` — the other family of quotes that are not what
  they look like.
- `../../../fin-core/skills/us-market-rules/SKILL.md` — the US regulatory and calendar layer this
  sits inside.
- `../../../fin-models/skills/term-structure-models/SKILL.md` — building the curve these rates
  project off.
- `../../../fin-core/skills/derivatives-pricing/SKILL.md` — 🚨 `rateslib` is not open source;
  QuantLib is the permissive route for repricing the swaps this skill describes.
