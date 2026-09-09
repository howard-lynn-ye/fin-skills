---
name: corporate-bond-data-and-trace
description: >-
  Use FINRA TRACE corporate bond data without inheriting the two things it does not tell you -
  the 15-minute reporting window and the size caps that censor volume. TRIGGER - TRACE, FINRA
  trade reporting, corporate bond tape, bond transaction data, "how fast are bond trades
  reported", one-minute TRACE reporting, Rule 6730, 15-minute reporting, dissemination cap,
  "5MM+", "1MM+", capped trade size, bond volume, TRACE academic or historic files, bond
  turnover, Amihud illiquidity on bonds, bond VWAP, "why is my bond volume so low", corporate
  bond liquidity screen, WRDS bond data, PyBondLab. SKIP for turning a bond price into a spread
  (credit-spread-measures), for CDS quotes and upfronts (cds-mechanics-and-upfront), for the
  rating that put the bond in an index (ratings-transitions-and-migration), for equity tick data
  and TAQ (intraday-microstructure), and for choosing a market data vendor or API in general
  (market-data-sourcing).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Corporate bond data and TRACE

**Two facts about TRACE are load-bearing and a model trained through 2025 gets both wrong.**
Reporting is **15 minutes, not one** — the SEC approved a one-minute rule in September 2024 and
FINRA abandoned it. And disseminated **size is right-censored** at $5MM (investment grade) or
$1MM (non-investment grade), so TRACE volume is a lower bound, not a measurement, and every
statistic with volume in it inherits that.

Measured figures below are printed by `scripts/trace.py` (runs in **1.3 s**, numpy + scipy, one
seeded synthetic tape, no network). ✅ Measured means this file produced it on 2026-09-09 with
numpy 2.2.6, scipy 1.13.0, Python 3.11.3. Regulatory facts are marked ✅ source-verified with
the page and the date they were read.

> **The rule:** TRACE reporting is **15 minutes**, and a disseminated size is a **lower bound**.
> "5MM+" is a **censored observation** — treat it as one, or every volume statistic you build is
> biased in a known direction.

## 1. 🚨 It is still 15 minutes. It was never one minute.

✅ **Source-verified at finra.org on 2026-09-09.** FINRA **Regulatory Notice 25-17**, *FINRA
Adopts Amendments to Rule 6730 (Transaction Reporting) to Streamline Allocation Reporting for
BD/IAs*, published **2025-12-04**, effective **2026-06-08**, says:

> "FINRA is not moving forward with its proposal to reduce the 15-minute TRACE reporting outer
> limit to one minute."

and restates the standing requirement as reporting "as soon as practicable but no later than
within 15 minutes of the time of execution".

🚨 **Why this is the trap and not a footnote:** the SEC approved the one-minute amendment in
2024, so the entire 2024–2025 written record — press coverage, vendor blogs, compliance
newsletters — says "TRACE is moving to one minute". A model that read that record and stopped
there will state it as current fact. It is not; the amendment was withdrawn before it took
effect.

✅ **The current outer limits**, from FINRA's own TRACE reporting-timeframes page (read
2026-09-09):

| security | outer limit | rule |
|---|---|---|
| Corporate and agency bonds | **within 15 minutes of execution** | FINRA Rule 6730(a)(1) |
| Asset-backed securities | within 15 minutes of execution | FINRA Rule 6730(a)(3) |
| Most other securitized products | same day, TRACE business hours | FINRA Rule 6730(a)(3)(A) |
| US Treasury securities | generally within **60 minutes** | FINRA Rule 6730(a)(4)(A) |

**Treasuries are 60 minutes, not 15.** A pipeline that applies one latency assumption across a
TRACE feed is wrong on at least one asset class.

### ✅ What 15 minutes of staleness is worth

A print you read is at least as old as the reporting limit, so the window is the size of the
price move you cannot see. ✅ Measured, 5% annualised price volatility, 6.5-hour sessions:

| window | sd of the move you did not see | vs one minute |
|---|---|---|
| 1 minute | 1.59 bp | 1.00x |
| **15 minutes** | **6.18 bp** | **3.87x** |
| 60 minutes (Treasuries) | 12.35 bp | 7.75x |

**3.87x more price uncertainty per print than the one-minute world people write about.** Any
event study, execution benchmark or "was this trade at a fair level" test that timestamps a
print at its dissemination time is that much too confident.

## 2. 🚨 Disseminated size is censored: 2% of prints hide 16% of the volume

✅ **Source-verified at finra.org on 2026-09-09:** transaction size caps of **$5 million for
investment grade** and **$1 million for non-investment grade**; above the cap the tape shows
**"5MM+"** or **"1MM+"** and never the true size.

✅ Measured on one seeded tape — 3,000 trades over 250 days, lognormal sizes (median $267,878),
**the same trades under both caps** so the cap is the only difference:

| cap | printed as | trades capped | **volume hidden** | mean-size error | VWAP error |
|---|---|---|---|---|---|
| $5MM (IG) | `5MM+` | **1.97%** | **−16.28%** | −16.28% | **−3.92 bp** |
| $1MM (HY) | `1MM+` | 17.90% | **−47.49%** | −47.49% | −4.10 bp |

🚨 **Under 2% of the prints carry 16% of the volume at the investment-grade cap, and nearly half
of it at the high-yield cap.** The censoring is invisible in a trade count and enormous in a
volume total.

✅ **The median survives exactly and the mean does not** — $267,878 either way, because fewer
than half the trades are capped. **If you need one number off a censored tape, use the median.**

✅ **The VWAP moves too, by −3.92 bp.** Large corporate-bond trades execute at *tighter*
half-spreads than small ones, so under-weighting them pulls the volume-weighted average price
away from the mid and toward the retail end of the tape. It is not a rounding error on a
transaction-cost benchmark.

## 3. 🚨 An Amihud ratio is biased up — but not by the volume fraction

Amihud illiquidity is `mean over days of |daily return| / daily dollar volume`. Volume is the
denominator, censoring shrinks it, so the measure can only go **up**. ✅ Measured on the same
tape:

| cap | true | from TRACE | **overstated by** | what `1/(1 − hidden)` predicts |
|---|---|---|---|---|
| $5MM | 0.000395 | 0.000407 | **+2.89%** | +19.45% |
| $1MM | 0.000395 | 0.000558 | **+41.08%** | +90.44% |

🚨 **The direction is certain and the magnitude is not what you would guess.** Scaling the bias
by the hidden-volume fraction overshoots by **6.7x** at the investment-grade cap. Amihud averages
a *ratio* across days and is dominated by low-volume days — which are precisely the days with no
capped print — so the correction is not a constant factor and cannot be applied as one.

### 🚨 And it contaminates the cross-section, which is what people actually use it for

✅ Measured: two bonds with **identical true liquidity** (the same tape), one investment grade
and one high yield, differing only in which cap applies.

| | ratio HY / IG |
|---|---|
| true Amihud | **1.0000** |
| Amihud from TRACE | **1.3712** |

🚨 **The high-yield name looks 37.12% more illiquid than the investment-grade name with no
difference in liquidity at all** — the $1MM cap censors more than the $5MM cap does. A liquidity
screen, a liquidity factor, or a bid-ask proxy sorted on TRACE volume is sorting partly on
**credit rating**, because the rating chooses the cap.

## 4. ✅ The fix: "5MM+" is a censored observation, so estimate with it

A capped print is not a $5MM trade and not a missing value. It is the statement
`size > 5,000,000`, and that is exactly the likelihood contribution a right-censored (Tobit-style)
fit uses: `log f(x)` for the prints you see, `log(1 − F(cap))` for the ones you do not.
✅ Measured, `censored_lognormal_mle` on the same tape:

| cap | true mean | naive mean | **MLE mean** | naive error | **MLE error** |
|---|---|---|---|---|---|
| $5MM | 770,802 | 645,288 | **766,992** | **−16.28%** | **−0.49%** |
| $1MM | 770,802 | 404,748 | **783,475** | **−47.49%** | **+1.64%** |

✅ At the $1MM cap the fit recovers **sigma = 1.4820** against the true **1.5000**, and a total
volume of **$2.350bn** against **$2.312bn** true — where reading the tape straight gives
**$1.214bn**, 47% low. **A 47% error becomes a 1.6% error from the same data**, because the
capped prints were used as information rather than thrown away or taken at face value.

⚠️ The fit is only as good as the shape you assume; a lognormal is the standard choice for trade
sizes and is what the simulation draws from, so this is an upper bound on how well it can do.
The transportable point is the *estimand*: never sum a column that contains `5MM+`.

## 5. What else the tape does not say

- ⚠️ **A print is not a mid.** TRACE reports executions, one side at a time, with no quote
  attached. Building a return series from consecutive prints mixes a bid-side print with an
  offer-side print and manufactures bid-ask bounce; §2's VWAP result is a version of the same
  problem. See `../../../fin-core/skills/execution-cost-analysis/SKILL.md`.
- ⚠️ **Uncapped size is available only in the historic/academic files**, on a lag, not in the
  real-time feed. The real-time feed is permanently censored — the cap is a dissemination rule,
  not a delay.
- 🚨 **`PyBondLab` 0.2.0 (2026-04-08) is MIT in the PyPI metadata and `NOASSERTION` from the
  GitHub API** (GiulioRossetti94/PyBondLab). Read the repository's LICENSE file before treating
  it as MIT. It implements published bond-return filters, which are worth reading whatever you
  decide about the licence.

## 6. What the script gives you

`scripts/trace.py` — numpy and scipy only. No network: the regulatory facts are quoted with
their source, the tape is synthetic and seeded (`SEED = 20260909`).

| Function | Does |
|---|---|
| `reporting_timeframes()` / `dissemination_caps()` | §1, §2, the FINRA table with its rule numbers |
| `staleness(annual_vol, minutes)` | §1, the unseen move per reporting window |
| `simulate_tape(n_days, trades_per_day, seed)` | the seeded tape; half-spread shrinks with size |
| `apply_cap(size, cap)` | what TRACE would have disseminated |
| `censoring_report(tape, cap)` | §2, every statistic the cap changes |
| `amihud(tape, size)` / `amihud_bias(tape, caps)` | §3, true vs reported |
| `cross_sectional_distortion(tape)` | §3, the IG-vs-HY ranking distortion |
| `censored_lognormal_mle(reported, cap)` | §4, the Tobit-style fit |
| `recovery_report(tape, cap)` | §4, naive vs MLE against the truth |

## Where this sits

- `../credit-spread-measures/SKILL.md` — what to do with the price once you have it, and why
  "the spread" is six numbers. A spread computed off a one-sided last print inherits the
  bid-offer measured in §2.
- `../cds-mechanics-and-upfront/SKILL.md` — the other quote on the same credit. CDS is quoted
  and cleared, not printed on a tape; the bond-CDS basis is a trade, and it is also the place a
  stale 15-minute bond print shows up as fake basis.
- `../ratings-transitions-and-migration/SKILL.md` — the rating decides which dissemination cap
  applies, so §3's cross-sectional distortion is indexed by rating.
- `../../../fin-core/skills/market-data-sourcing/SKILL.md` — vendor choice, keys and coverage.
  FRED's ICE BofA series (`BAMLC0A0CM`, `BAMLH0A0HYM2`) are index-level and free; TRACE
  historic/academic files and WRDS are the trade-level routes.
- `../../../fin-core/skills/intraday-microstructure/SKILL.md` — the equity-tape versions of the
  same problems (bid-ask bounce, trade signing, timestamp alignment). Bonds differ mainly in
  that the tape is delayed and the size is censored.
- `../../../fin-core/skills/execution-cost-analysis/SKILL.md` — benchmarking a fill against a
  VWAP that §2 shows is 3.92 bp off.
- `../../../fin-models/skills/credit-risk-models/SKILL.md` — the bond price in §2 is the input
  to an implied hazard rate; the 15-minute staleness is inside that PD before the model starts.
