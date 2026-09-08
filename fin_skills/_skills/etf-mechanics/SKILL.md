---
name: etf-mechanics
description: >-
  Why an ETF's price series does not behave like the index it tracks - daily-reset leverage, NAV
  vs price, distributions, holdings files and fees. TRIGGER - "why is my 3x ETF down when the
  index is flat", TQQQ decay, SQQQ, leveraged ETF long term, inverse ETF, volatility drag, daily
  reset; premium to NAV, discount to NAV, iNAV, creation/redemption, "bond ETF trading below NAV";
  ETF distribution, capital gains distribution, return of capital, "ETF dropped on the ex-date",
  phantom drop; ETF holdings file, constituents CSV, index reconstitution, Russell rebalance, "I
  used today's holdings for the backtest"; expense ratio drag, tracking difference vs tracking
  error, "ETF returned less than the index". SKIP for downloading price series and vendor
  adjustment defaults (market-data-sourcing) - holdings files stay here, for UNG, USO or VIXY roll
  yield and contango (futures-continuous-contracts), for auditing a finished backtest
  (research-integrity-guards), and for weights, Sharpe or drawdown (portfolio-and-risk).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-08"
---

# ETF mechanics

**An ETF is a fund with a share price. The index is a formula.** The two differ by five things a
price series never shows: the daily reset of a leveraged product, the gap between NAV and price,
cash that leaves as distributions, holdings that change, and fees. Every computed number below is
printed by `scripts/leveraged_reset.py` (numpy/pandas, seed 0). Issuer and regulator facts carry
the date they were read.

## 1. 🚨 "3x the index" is a one-day statement

✅ The issuer's own words (proshares.com, read 2026-09-08): TQQQ *"seeks daily investment results,
before fees and expenses, that correspond to three times (3x) the daily performance of the
Nasdaq-100 Index"*; SQQQ is the same sentence with *"three times the inverse (-3x)"*. Both pages
add: *"For any holding period other than a day, your return may be higher or lower than the Daily
Target. These differences may be significant."*

✅ FINRA Regulatory Notice 09-31 (2009-06-11): *"Most leveraged and inverse ETFs 'reset' daily"*.
Its example — index 100 → 101 → 100 costs an inverse ETF 0.02%; 100 → 110 → 100 costs it 1.82% —
is reproduced exactly by the script (§A: −1x on "+10% then back to flat" = **−1.82%**).

### A flat year is not free — but the honest size is smaller than the folklore

✅ §B: a random 252-day path whose realized vol is pinned and whose index return is **exactly 0**:

| ann. vol | 2x | **3x** | −1x | −3x |
|---|---|---|---|---|
| 16% (S&P-like) | −2.53% | **−7.40%** | −2.53% | −14.26% |
| 25% | −6.07% | **−17.14%** | −6.06% | −31.37% |
| 40% (single-name 3x, or a bad Nasdaq year) | −14.82% | **−38.32%** | −14.81% | −62.09% |

The popular "up 10%, down 10%" examples use daily moves of 5–10%; a real S&P year at 16% vol costs
a 3x product **7.4%**, not 50%. What decides the number is **realized variance**, and it grows
with the square of leverage: the −1x product loses about as much as 2x, the −3x loses twice what
3x loses.

### The analytic form, and what it does and does not tell you

`W_L ≈ (1 + R)^L · exp(−(L² − L)/2 · Σr²)` — the product's wealth is the index wealth to the
power L, times a drag set by the realized variance of the path. For L = 3 the exponent is
−3 × Σr². ✅ §C: across 2,000 random flat 25%-vol years the second-order form is off by a mean
**−0.030 pp** and at most **0.382 pp** against a **−17.10%** drag — the approximation is good.

🚨 **It is a statement about the path already taken.** Σr² is realized variance, so the formula
predicts nothing about next year unless you also forecast the variance — which is the whole
problem, restated.

### "Decay" is the wrong word: the gap changes sign

✅ §D, one random shape at 16% vol, index return pinned per row:

| index R | 3x product | 3 × R | gap |
|---|---|---|---|
| −30% | −68.30% | −90.00% | **+21.70 pp** |
| −20% | −52.63% | −60.00% | +7.37 pp |
| −10% | −32.50% | −30.00% | −2.50 pp |
| 0% | −7.39% | 0.00% | −7.39 pp |
| +10% | +23.27% | +30.00% | −6.73 pp |
| +20% | +60.02% | +60.00% | +0.02 pp |
| +30% | +103.39% | +90.00% | +13.39 pp |
| +50% | +212.16% | +150.00% | **+62.16 pp** |

Near zero the variance drag dominates; in a strong trend compounding dominates and the 3x product
**beats** three times the index — in both directions (a −30% year loses 68%, not 90%). The daily
reset is path dependence, not a fee.

### Holding-period sensitivity — and it is the median that suffers, not the mean

✅ §E, 20,000 zero-drift Monte Carlo paths, gap = 3x return − 3 × index return:

| horizon | 16% vol, median gap | 25% vol | 40% vol | P(gap < 0), 16% |
|---|---|---|---|---|
| 1 day | **0.00 pp** | 0.00 pp | 0.00 pp | 0.0% |
| 5 days | −0.03 pp | −0.08 pp | −0.21 pp | 62.8% |
| 21 days | −0.28 pp | −0.68 pp | −1.72 pp | 66.7% |
| 63 days | −0.96 pp | −2.33 pp | −5.77 pp | 67.4% |
| 126 days | −1.98 pp | −4.72 pp | −11.22 pp | 67.5% |
| 252 days | −3.98 pp | −8.89 pp | −18.23 pp | 67.7% |

The **mean** gap is ~0 at every horizon (`E[Π(1 + 3r)] = 1` when `E[r] = 0`); at 40% vol and one
year the median 3x outcome is **−51.06%** while the mean is **+0.86%**. Volatility drag lowers the
typical outcome, not the expected one — a right-skewed lottery, which is why "it always decays"
and "expected return is 3x" are both wrong and both commonly said.

### The expense ratio is the small cost

✅ §F, the flat 16%-vol year: variance drag alone **−7.40%**; add financing of the borrowed 2 × NAV
at 4.00%/yr → **−14.53%** (−7.13 pp); add a 0.82%/yr expense ratio → **−15.23%** (−0.70 pp). The
4.00% is an input — the fund pays what its swap counterparties charge, and that is not published
as one number. ✅ TQQQ's own holdings page (as of 2026-09-04) lists Nasdaq-100 index swaps with
ten bank counterparties, each shown at 16–30% exposure, over a partial equity basket; ✅ yfinance's
`funds_data.top_holdings` shows **3 lines totalling 24.1%** of TQQQ — the leverage is invisible
there.

**What this does NOT show:** no swap spread, rebalancing cost or tracking noise, so every cost
figure is a lower bound; the drag numbers transfer to any product with the same realized variance;
the sign of the gap does not transfer anywhere.

## 2. NAV vs price

NAV is `(Σ shares × price + cash − liabilities) / shares outstanding`, struck once a day from the
holdings file. ✅ §G, five lines: NAV **59.7696**, close 59.79, premium **+0.034% = +3.4 bp**.
An authorized participant's round trip — creation fee 1.7 bp + basket half-spread 2.0 bp + hedge
1.0 bp = **4.7 bp** — is wider than that, so nobody creates or redeems, and the premium is noise.
(Those three inputs are chosen for the example; measure yours.) Arbitrage keeps the premium inside
the cost of the basket, which for a liquid US-equity ETF is a few basis points.

✅ **What the issuer must publish** — Rule 6c-11(c)(1), 17 CFR 270.6c-11 (text read 2026-09-08 at
the LII mirror; ecfr.gov and federalregister.gov refused automated fetches; substance confirmed in
SEC press release 2019-190, 2019-09-26): each business day, **before the opening of regular
trading**, every portfolio holding (ticker, CUSIP, description, quantity, weight); the prior day's
NAV, market price and premium/discount; a table and line graph of premiums/discounts for the last
calendar year; the median bid-ask spread over the prior 30 calendar days sampled every 10 seconds;
and an explanation if the premium or discount exceeded **2% for more than seven consecutive
trading days**. This is the data you use to check an ETF, and it is free.

✅ Issuer pages, figures as of 2026-09-04, read 2026-09-08:

| ETF | NAV | close | premium | 30-day median spread | holdings |
|---|---|---|---|---|---|
| IVV (S&P 500) | 774.0352 | 773.92 | −0.01% | 0.01% | 504 |
| SPY (S&P 500, a unit investment trust) | 770.33 | 770.19 | −0.01% | 0.00% | daily file |
| LQD (IG corporate bonds) | 105.4852 | 105.48 | 0.00% | 0.01% | 3,137 |
| **EWJ (Japan)** | 98.0752 | 98.28 | **+0.22%** | 0.01% | 167 |

### 🚨 The stale-NAV trap: an international "premium" is mostly the clock

EWJ's holdings stopped trading in Tokyo hours before the US close; the NAV is struck from those
last prices (✅ iShares states FX is taken *"as of the close of business on the New York Stock
Exchange"*; ⚠️ whether it fair-values the Tokyo closes is not stated on the page). The ETF
trades at the US-hours fair value. The difference prints as a premium, and it is not mispricing.

✅ §G simulates it (home-hours moves 0.8%/day, US-hours moves 0.6%/day): the premium's std is
**0.59%** — it *is* the US-hours move — and **corr(premium today, NAV return tomorrow) = +0.59**
while **corr(premium today, price return tomorrow) = 0.00**. A backtest that "buys the discount
and waits for NAV to catch up" is right that NAV catches up and wrong that you can trade at NAV.
Judge an international ETF's premium against its own history table (the 6c-11 disclosure above),
not against IVV's.

### Fixed income: the NAV is the stale price, not the ETF

Bond NAVs come from evaluated (vendor) prices (✅ iShares: *"The vendor price is not necessarily
the price at which the Fund values the portfolio holding"*), and many bonds never trade on a given
day. In stress the ETF trades continuously and the NAV lags. ⚠️ March 2020 (press reports via
search, 2026-09-08): LQD traded at discounts of roughly 3–5% around 2020-03-20 and at a ~3%
premium on 2020-03-23 after the Fed's announcement. A "5% discount" on a bond ETF in a crisis is
information about the NAV, not an arbitrage — and the 6c-11 history table is where you see how
often it happens.

⚠️ **iNAV / IIV**: Rule 6c-11 did not require an intraday indicative value; exchange listing rules
required a 15-second IIV until they were amended (law-firm summaries of the adopting release,
2019). Where it exists it is computed from last prints, so it carries the same staleness as NAV.

## 3. Distributions and the phantom ex-date drop

Three kinds of cash leave a fund: **income**, **capital-gains distributions** (✅ IRS Topic 404:
*"always reported as long-term capital gains"*), and **return of capital**, which ✅ *"reduces the
adjusted cost basis of your stock"* and becomes a capital gain once basis reaches zero. ✅ Under
Rule 19a-1 (17 CFR 270.19a-1, LII mirror) a payment not wholly from net income must come with a
written statement of what portion is income, realized gains, or paid-in capital — that notice is
where "12% yield" turns out to be 5% income and 7% of your own money back.

**A distribution moves the price by its full amount whatever its tax label.** ✅ §H:
`1.00/share with 0.60 ROC moves the price by the full 1.00; only 0.40 is income`. An unadjusted
close series therefore shows a drop that never happened to a holder:

- ✅ 10 years, 1.8%/yr paid quarterly plus one 3% capital-gains payout: total-return CAGR
  **+1.80%**, unadjusted-price CAGR **−0.24%** — a **2.04 pp/yr** gap for 2.01%/yr paid out.
- ✅ The capital-gains day prints **−4.21%** on a true **−0.76%** day.
- ✅ Ordinary quarterly dividends are *not* outliers (1 of the 10 worst unadjusted days), but for
  an income product (4.8% vol, 0.35% monthly payout = 1.2 daily sigmas) **5 of the 10 worst days
  are ex-dates**. A stop-loss or reversal signal on that series fires on payouts.

**Use a total-return series, backward-adjusted, and never a forward-adjusted one** — the
conventions, the yfinance `Adj Close == Close` tell and the per-library defaults are in
`../research-integrity-guards/references/adjustment-conventions.md`; this skill does not repeat
them. ✅ Verified live with yfinance 1.7.0 (2026-09-08): `Ticker("IVV").history(period="2y",
auto_adjust=False, actions=True)` carries `Dividends`, `Stock Splits` **and `Capital Gains`**
columns — 8 dividend rows, 0 capital-gains rows; on the latest ex-date (2026-06-15) `Adj Close ==
Close == 756.35`, while 2025-12-16 shows `Close 679.84` vs `Adj Close 676.22`. That is the
present-anchored series the reference file warns about. ⚠️ Equity index ETFs rarely distribute
capital gains because in-kind redemption removes low-basis lots — a mechanism, not measured here;
IVV's zero rows are consistent with it, not proof.

## 4. Holdings files and reconstitution

**Where a holdings file comes from.** ✅ Under 6c-11 it is the issuer's daily website file (§2).
Read 2026-09-08: IVV offers a CSV of 504 holdings as of 2026-09-04; SPY a daily XLSX; TQQQ/SQQQ
a CSV as of 2026-09-04. ⚠️ **Vanguard's ETF share classes (VOO and siblings) do not rely on
6c-11**, and Vanguard's filings describe holdings for those funds as monthly with a 15-day lag —
the phrase *"complete portfolio holdings as of the end of the most recent month"* returns 15 hits
in Vanguard Index Funds' 485BPOS filings on EDGAR full-text search (latest 2025-04-29), but the
full sentence was not read; investor.vanguard.com renders nothing without a browser. Treat "daily
holdings" as issuer-specific until you have the file with its **as-of date** in hand.

🚨 **`yfinance` does not give you a holdings file.** ✅ 1.7.0 (released 2026-08-26), run
2026-09-08: `Ticker(t).funds_data.top_holdings` returns **10 rows for IVV (37.7% of the fund), 3
for TQQQ (24.1%), 10 for EWJ**, columns `Name` and `Holding Percent`, **no as-of date**, no
history. It is Yahoo's "top ten today". ✅ The iShares `...ajax?fileType=csv` URL that circulates
in scrapers returned the product web page, not a CSV, from this environment on 2026-09-08 — the
download link is a page, not an API. Download from the page, keep the as-of date, and snapshot it
(`../market-data-engineering/SKILL.md` §8).

### 🚨 Today's file is a current list, and using it for history is survivorship bias

✅ §J, 300 stocks with **identical expected returns**, 30% idiosyncratic vol, cap-weighted top-100
index rebalanced monthly for 10 years: point-in-time membership CAGR **+5.34%**; today's 100
members held throughout **+8.74%** — **+3.40 pp/yr of pure selection**, with 35 of today's 100
names not in the index at the start. Nothing here has any alpha; the gap is the file. Real churn
and dispersion differ, so measure it on your index, but the sign never changes. This is gate 1 of
`../research-integrity-guards/SKILL.md` — a holdings file is exactly "a ticker list built from
today's constituents". Build membership from dated files (`../research-integrity-guards/scripts/
pit_universe.py`) and record the snapshot id.

### Reconstitution is a schedule, and the schedule is a trade

- ✅ FTSE Russell (LSEG press release 2026-05-22): the Russell US indexes are reconstituted
  **semi-annually, June and December, from 2026** (previously annual); rank day **2026-04-30**,
  effective after the close on **Friday 2026-06-26**; **$217.2 billion** traded at the June 2025
  reconstitution close.
- ⚠️ S&P 500: rebalanced quarterly after the close on the third Friday of March, June, September
  and December, with constituent changes made as needed and announced days ahead (spglobal.com
  returned 403 to every fetch; from search snippets of the S&P U.S. Indices methodology).
- ⚠️ Petajisto, *Journal of Empirical Finance* 18(2), 2011 (abstract read 2026-09-08): the
  index-turnover cost of buying additions after they have run up and selling deletions after they
  have fallen has a lower bound of **21–28 bp/yr for the S&P 500 and 38–77 bp/yr for the Russell
  2000** over 1990–2005, peaking in 2000.

**Two dates matter.** Prices move on the announcement; index funds trade at the close of the
effective date. A backtest that adds a name at the effective-date close is paying what the index
funds paid, which is the right price for an index-tracking strategy and the wrong one for a
strategy that claims to front-run it.

## 5. Expense ratio, tracking difference, tracking error

- **Tracking difference (TD)** = fund return − index return over a period. **This is the money.**
- **Tracking error (TE)** = annualized standard deviation of the periodic return differences.
  Dispersion around the TD, not a cost.

✅ §I, one 10-year index path, three funds:

| fund | TD /yr | TE /yr | 10-year wealth vs index |
|---|---|---|---|
| A: 0.95% fee, full replication | **−1.05%** | 0.00% | −9.06% |
| B: 0.03% fee, sampled (0.50%/yr noise) | −0.11% | **0.49%** | −0.95% |
| C: 0.20% fee, +0.05% lending income, 0.10% noise | −0.13% | 0.10% | −1.12% |

Fund A has zero tracking error and loses 9% of terminal wealth; fund B has the largest tracking
error and costs a tenth of that. A screen sorted on TE picks the wrong fund. Two details the table
also shows: A's TD is −1.05%, not −0.95%, because the fee accrues on wealth that grew; and TE
makes the TD you *measure* noisy — a 0.50% TE over 10 years is **±0.16%/yr** in the realized TD,
so B's −0.11% is its 0.03% fee plus noise. ✅ 0.95%/yr for 10 years at a 0% index return is
**−9.10%** of terminal wealth.

✅ Expense ratios from issuer pages, 2026-09-08: SPY 0.0945% (gross), IVV 0.03%, QQQ 0.18%,
LQD 0.14%, EWJ 0.49%, TQQQ 0.82% net / 0.97% gross, SQQQ 0.95% net / 0.99% gross. For a leveraged
product the expense ratio is the small line — §1 shows financing at 4% costing ten times the fee.

## 6. Rolling-futures ETFs: already measured elsewhere

UNG, USO, VIXY and their kin hold futures and pay the roll. Do not backtest the spot series and
trade the ETF: `../../../fin-futures-fx/skills/futures-continuous-contracts/SKILL.md` §5 measured
**UNG −23.25 pp/yr against NG=F and VIXY −45.9 pp/yr against ^VIX**, with GLD as the near-zero
control. Everything about roll yield, contango and continuous contracts lives there, not here.

## 7. Where this sits

- `../market-data-sourcing/SKILL.md` — getting the price series, and which vendor adjusts what.
- `../research-integrity-guards/SKILL.md` §1 — survivorship, the general form of §4 above; and
  `../research-integrity-guards/references/adjustment-conventions.md` — the rules §3 relies on.
- `../../../fin-futures-fx/skills/futures-continuous-contracts/SKILL.md` §5 — futures ETFs.
- `../portfolio-and-risk/SKILL.md` — once you have a correct total-return series, the metrics.

## 8. Scripts

`scripts/leveraged_reset.py` — sections A–J: the two-day reset arithmetic, the pinned flat and
trending years, the analytic-vs-exact check, the 20,000-path holding-period table, financing and
expense drag, NAV/premium and the stale-NAV correlations, the phantom ex-date drop, TD vs TE, and
the today's-holdings look-ahead. numpy and pandas only, seed 0, ASCII output, about ten seconds.
