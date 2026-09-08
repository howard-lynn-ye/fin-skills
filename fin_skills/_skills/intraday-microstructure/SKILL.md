---
name: intraday-microstructure
description: >-
  Measure the market at the tick level and know when the measure is lying. TRIGGER - build dollar
  bars, volume bars, tick bars, imbalance bars; classify trades as buyer or seller initiated,
  Lee-Ready, tick rule, bulk volume classification, BVC; quoted vs effective vs realized spread,
  price impact, Kyle's lambda, Amihud illiquidity, order flow imbalance, OFI, VPIN; reconstruct
  the order book from MBO, market by order, L2 vs L3, queue position, NBBO, odd lots, dark prints;
  "my tick strategy works on bars but not on ticks", exchange clock vs vendor clock, the 5-second
  rule. SKIP for downloading tick data or picking a vendor (market-data-sourcing), for as-of joins
  and tick storage (market-data-engineering), for the cost of your own fills
  (execution-cost-analysis), for indicator look-ahead on bars (signal-construction), for how an
  engine fills orders or what spread to assume in a bar backtest (backtesting-engines) and for
  short-sale or margin rules (us-market-rules).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-08"
---

# Intraday microstructure

**Every microstructure estimator has a failure mode that returns a plausible number.** A trade
classifier that is 79% right looks fine until you learn it flips exactly the trades that moved the
price. So this skill scores each estimator on a synthetic stream where the answer was **planted**,
and reports the size of the failure as a number rather than as a warning.

✅ Every number below is printed by `scripts/microstructure_measures.py` (numpy 2.2.6, pandas
2.2.3, seed 20260908, about 20 s). It scores *estimators*, not markets: nothing here is a claim
about how real order flow behaves.

Getting tick data is `../market-data-sourcing/SKILL.md` (Databento MBO in
`../market-data-sourcing/references/databento.md`); joining quotes to trades without look-ahead
is `../market-data-engineering/SKILL.md` §2. This file starts where those stop.

## 1. Bars: activity time helps only when variance arrives with trades

Four bar types from one stream of 733,383 trades over 40 sessions, thresholds set so each type
yields about the same number of bars as 1-minute time bars (matched counts, or the comparison is
unfair). Two worlds, same price mechanism:

| World A - **94%** of daily variance arrives with trades | ex. kurt | \|z\|>3 | half-hour var ratio | bars/day |
|---|---|---|---|---|
| time 1-min | **7.13** | 1.794% | 2.4 | 388-390 |
| tick (47 trades) | 2.45 | 1.213% | 1.2 | 213-831 |
| volume (11,832 sh) | **1.87** | 0.911% | 1.3 | 209-824 |
| dollar ($1.25M) | **1.87** | 1.021% | 1.2 | 209-811 |

| World B - **16%** of daily variance arrives with trades; the rest is news and jumps on the clock | ex. kurt | \|z\|>3 |
|---|---|---|
| time 1-min | 54.80 | 0.836% |
| tick (42 trades) | 57.15 | 0.807% |
| volume (10,584 sh) | 58.95 | 0.841% |
| dollar ($1.06M) | 62.53 | 0.770% |

*(Gaussian: ex. kurt 0, |z|>3 0.270%; var ratio = max/min mean squared return across the 13
half-hours, 1 = an even clock)*

- **In world A the effect is real and modest**: excess kurtosis 7.13 → 1.87, and the intraday
  variance ratio 2.4 → 1.2 — activity bars even out the U-shaped clock. That is the whole
  mechanism (the mixture-of-distributions story, ⚠️ Clark 1973 / Ané-Geman 2000), and it only
  works when variance comes *with* trades.
- **In world B they do nothing**: kurtosis 54.80 vs 57.15 / 58.95 / 62.53 — within sampling
  noise of each other, because a handful of jump bars set the number. Whether your instrument is
  A or B is an empirical question about your data, not a doctrine.
- 🚨 **Dollar bars are not better than volume bars here** — identical 1.87 in world A, because
  the price level moves 15% in 40 sessions and the two thresholds select almost the same trades.
  Dollar bars earn their keep across *years* and splits, not weeks. Do not expect a fortnight of
  tick data to show a dollar-bar advantage.
- 🚨 **The bars/day column is the price**: one fixed threshold gives 209-831 bars on different
  days. Sample sizes float, cross-asset alignment breaks, and **a threshold fitted on the whole
  sample already knows future volume** — set it from trailing data, e.g. trailing ADV / desired
  bars per day.
- 🚨 **Stamp the bar at its END.** A volume bar completes when the threshold is crossed, not when
  it starts; index it by the last trade's timestamp or you leak the whole bar. The generic rule
  and the indicator classes that inherit it are in `../signal-construction/SKILL.md` §4.

Imbalance and run bars (⚠️ López de Prado 2018) follow the same threshold logic with an
expectation you must estimate — one more fitted quantity that has to come from the past.

## 2. Trade classification: the lag belongs to your clock, not the rule

The rules, as implemented in the script (attributions ⚠️ secondhand, the mechanics ✅ executed):

| Rule | Mechanism |
|---|---|
| Quote rule | price above the prevailing mid = buy, below = sell, at the mid = **unclassified** |
| Tick rule | uptick = buy, downtick = sell, zero tick carries the previous sign |
| **Lee-Ready** (⚠️ 1991) | quote rule, tick rule only at the midpoint |
| EMO (⚠️ Ellis-Michaely-O'Hara 2000) | quote rule only for trades at the touch, tick rule elsewhere |
| BVC (⚠️ Easley-López de Prado-O'Hara 2012) | per **bar**: buy fraction = Φ(ΔP / σ_ΔP); needs no quotes and no trade sides |

✅ 185,480 trades with a known initiator: 71.4% at the touch, 8.8% at the midpoint, 7.4%
price-improved, 12.4% walked the book. Quotes re-centre a median 3 ms after a trade. The vendor
clock stamps trades late by a median 80 ms (95th percentile 414 ms), so under it the trade
arrives *after* the quote update it caused.

| Lee-Ready accuracy | lag 0 | lag 1 s | **lag 5 s** |
|---|---|---|---|
| exchange timestamps | **91.0%** | 88.3% | 79.4% |
| vendor timestamps | 82.6% | **88.4%** | 79.5% |

Per location, exchange timestamps at lag 0: touch **100.0%**, midpoint **41.6%**, price-improved
48.3%, walked 100.0%. Tick rule alone: 84.5% overall, 41.5% at the midpoint. EMO: 90.7% at lag 0,
82.4% at 5 s.

- 🚨 **The 5-second convention takes Lee-Ready from 91.0% to 79.4% on exchange timestamps.** Lee
  and Ready recommended it for 1988 NYSE data where quotes were recorded ahead of the trades that
  caused them (⚠️ the paper) — which is precisely the *vendor clock* case above, where a 1-second
  lag does recover 88.4%. ⚠️ Later work (Bessembinder 2003; Holden-Jacobsen 2014) finds no lag or
  a sub-second lag best on modern timestamps. **Pick the lag from your timestamps, not from the
  literature.**
- 🚨 **Midpoint trades are the floor, and the tick rule is worse than a coin flip there
  (41.5%).** A midpoint print after a buy at the ask is a downtick, so the tick rule calls it a
  sell — and with persistent order flow it is usually another buy. Dark-pool and midpoint-cross
  prints (§6) land exactly here.
- **Quote updates lagging the trade** (vendor clock, lag 0): touch accuracy falls to 92.6% and
  midpoint to 29.0%, because the mid has already moved in the trade's direction and the print now
  sits at or below it. The errors are not random — see what that does to λ in §4.

**BVC vs Lee-Ready, at the bar level** (2,340 volume bars of 20,000 shares): correlation with the
true buy fraction **0.865 vs 0.955**, mean absolute error **0.142 vs 0.026**. BVC's ceiling is
how much of the bar's price change the order flow explains — the thing you do not know on real
data. It is the tool for when you have bars and no quotes, not a free upgrade.

## 3. Spreads: quoted, effective, realized, and impact = effective − realized

For a trade at price `p` with initiator sign `q` (+1 buy), prevailing mid `m` and the mid `Δ`
later `m_Δ`:

```
quoted    = ask − bid                     (time-weighted)
effective = 2 q (p − m)                    what the taker paid
realized  = 2 q (p − m_Δ)                  what the maker kept
impact    = 2 q (m_Δ − m) = effective − realized
```

✅ Quoted $0.0184 = 1.87 bps; effective mean $0.0188, share-weighted $0.0273 (size walks the
book). By horizon, realized / impact: 1 s $0.0087 / $0.0101, 30 s $0.0059 / $0.0129, 5 min
$0.0060 / $0.0126 — flat after the first seconds, because this simulator's temporary impact is a
walked book that re-centres in milliseconds.

| trade size | n | effective | realized | impact | impact bps |
|---|---|---|---|---|---|
| odd lot 1-99 | 53,959 | 0.0155 | 0.0107 | 0.0049 | 0.49 |
| 100 | 42,345 | 0.0158 | 0.0061 | 0.0097 | 0.99 |
| 200-400 | 56,999 | 0.0181 | 0.0044 | 0.0137 | 1.39 |
| 500-900 | 20,735 | 0.0259 | 0.0002 | 0.0257 | 2.61 |
| 1,000+ | 6,746 | 0.0442 | **−0.0003** | 0.0445 | 4.53 |

The impact was planted linear in size, so this table checks the estimator, not the market.
Realized spread reaches zero at 1,000+ shares: the maker's half-spread is consumed by the
permanent impact of size.

**The reversion test in `../execution-cost-analysis/SKILL.md` §5 is realized-spread thinking
from the taker's side**: measure the mid `T` after the last fill; what reverts was temporary
impact (the maker's realized spread), what stays was permanent (your impact). Same decomposition,
opposite sign convention. Use `Δ` consistently or the two will not reconcile.

## 4. Kyle's lambda, order-flow imbalance, Amihud

`Δmid = λ · (net signed volume) + noise` over 78 five-minute intervals per session (⚠️ Kyle 1985
for the model; the regression is plain OLS).

| sides used for the signed volume | mean λ̂ / planted | min λ̂ (1e-5 $/share) |
|---|---|---|
| true initiator | **1.018** (planted inside ±2 se on 90% of days) | 0.51 |
| Lee-Ready, exchange ts, lag 0 (91.0% accurate) | 1.043 | 0.52 |
| Lee-Ready, exchange ts, **lag 5 s** (79.4%) | 0.747 | 0.49 |
| Lee-Ready, **vendor ts**, lag 0 (82.6%) | **0.404** | **−1.14** |

🚨 **Classification error is not noise in the regressor.** Two classifiers of similar accuracy
(79.4% and 82.6%) give λ̂ of 0.747 and 0.404 of the planted value, and the vendor-clock one turns
**negative on the high-λ days** (planted 2.5 → λ̂ −1.14 and −0.99; planted 2.0 → −0.47), because
the flipped sides are precisely the trades that moved the quote. An attenuation correction assumes
random error and would not save you here. Check the clock before you estimate anything signed.

**OFI** (⚠️ Cont-Kukanov-Stoikov 2014) needs only L1 quotes — bid/ask price and size changes —
and no trade sides: `e = 1{b≥b₋}·q_b − 1{b≤b₋}·q_b₋ − 1{a≤a₋}·q_a + 1{a≥a₋}·q_a₋`, summed per
interval. ✅ Mean R² 0.59 vs 0.78 for signed volume here — **meaningless as a ranking**, because
depth changes in this simulator carry no information by construction. The script shows the
computation; its edge on real books is the paper's claim, not this file's.

**Amihud** (⚠️ 2002) is `|r_day| / $volume`, the low-frequency proxy that needs only daily bars —
its whole appeal. ✅ Over 40 sessions with a planted λ cycling by day, rank correlation with the
planted value: Kyle from trade prices **0.98**, Amihud **0.42**. One day of Amihud is one noisy
draw of |flow| × λ; average it over months before ranking anything.

## 5. VPIN — ⚠️ the debate is unresolved, do not sell it as a crash predictor

VPIN (⚠️ Easley-López de Prado-O'Hara 2012) is BVC applied to consecutive volume buckets:
`VPIN = Σ|V_buy − V_sell| / (n · V)` over a window of `n` buckets. ✅ On the synthetic stream:
mean 0.434, max 0.649 — a number with no event behind it.

⚠️ The empirical claim that VPIN rose ahead of the 2010 flash crash (Easley et al.) was disputed
by Andersen and Bondarenko (2014), who report that its predictive content depends on the
bucketing and classification choices and is largely captured by volume and volatility; the
authors replied.
This file takes no side. What is ✅ measurable: VPIN inherits BVC's error (correlation 0.865 with
the true buy fraction here) and every degree of freedom in the bucket size and window is a trial
to record in `../backtest-validation/scripts/trial_ledger.py`.

## 6. Reconstructing the book from market-by-order data

✅ Field semantics verified 2026-09-08 in the DBN source (`databento/dbn`,
`rust/dbn/src/record.rs`, `enums.rs`, `flags.rs`; pushed 2026-09-04, Apache-2.0):

| Field | What it is | Why you need it |
|---|---|---|
| `ts_event` | *matching-engine-received* timestamp, ns | the exchange clock |
| `ts_recv` | *capture-server-received* timestamp, ns | the vendor clock; `ts_in_delta` = matching-engine-*sending* time as ns before `ts_recv` |
| `order_id` | order ID assigned at the venue | queue position is impossible without it |
| `action` | **A**dd, **C**ancel, **M**odify, clea**R** book, **T**rade, **F**ill, **N**one | the book replay |
| `side` | **A**sk / **B**id / **N**one — the resting side, or the *aggressor* for a trade | trade sides without classification |
| `price` | int64, 1 unit = 1e-9 | exact tick arithmetic; the float conversion is lossy |
| `sequence` | venue sequence number | gap detection |
| `flags` | `LAST` = last record of the event for the instrument; `SNAPSHOT`; `BAD_TS_RECV`; `MAYBE_BAD_BOOK` = unrecoverable gap; `TOB`; `MBP` | when the book is valid |

- 🚨 **`T` and `F` "do not affect the book"** (verbatim in the enum docs). A reconstructor that
  decrements the resting order on the trade record double-counts when the venue's `C`/`M` for
  that order arrives. Apply book changes only on `A`/`C`/`M`/`R`.
- 🚨 **Apply an event atomically at `LAST`.** Records of one venue event arrive as several
  messages; a snapshot taken mid-event shows a book that never existed. On `MAYBE_BAD_BOOK` the
  book is garbage until an `R` (clear) or a snapshot.
- **Queue position** is *(size ahead of you at your price when you joined) − (cancels ahead of
  you since)*. Only order-level data tells you whether a cancel came from ahead or behind you.
  **L2 (market-by-price) cannot**: it aggregates every order at a level into one size, so from a
  cancel you learn only that the level shrank. That is why L2-based backtesters ship
  queue-position *models* — `hftbacktest` describes itself as accounting for "queue positions ...
  utilizing full tick data ... (Level-2 and Level-3)" (✅ GitHub description 2026-09-08). With MBO
  you replay the queue; with L2 you assume it.
- **The NBBO excludes odd lots by definition.** ✅ 17 CFR 242.600(b), read at eCFR 2026-09-08:
  "bid or offer" is a price at which a member "is willing to buy or sell **one or more round
  lots**", "odd-lot" is an order smaller than a round lot, and the NBBO is the best of those bids
  and offers. **Round lot is now tiered by price**: 100 shares at $250.00 or less, 40 at
  $250.01-$1,000, 10 at $1,000.01-$10,000, 1 above — ✅ the SEC states in the Federal Register
  (2026-06-17) that this took effect **2025-11-03**. So a quoted-spread series for a high-priced
  stock has a structural break at that date, and the odd-lot quotes inside the NBBO — the best
  prices actually available — are visible only in venue depth or in the "odd-lot information" the
  SIPs must disseminate by ✅ **2028-05-01** (same source).
- **Consolidated vs venue-specific.** The SIP/consolidated NBBO is assembled from each exchange's
  best quote (✅ 242.603(b)); a direct feed is one venue's book. The two disagree by the SIP's
  latency and by which quotes are protected: ✅ a *protected* quotation is the best bid or offer
  **of an exchange** (242.600(b)) — depth is never protected.
- **Trade-throughs.** ✅ A trade-through is an execution at a price worse than a protected quote
  (242.600(b)); Rule 611 requires policies to prevent them, with exceptions for intermarket sweep
  orders, benchmark trades, and a quote that was at that price within the last one second
  (242.611(b)). So prints outside the NBBO are legitimate and common in tick data; a "clean"
  filter that deletes them deletes real volume. ⚠️ **Status changing:** the SEC proposed on
  2026-06-17 to **rescind Rule 611** (Release 34-105655; comments closed 2026-08-17). No final
  rule was in the Federal Register as of 2026-09-08. Key any trade-through logic to a date.
- **Dark prints in the tape.** ✅ Off-exchange executions (ATS/dark pools, internalisers) are
  reported to a FINRA facility — TRF, ADF or ORF (FINRA trade-reporting FAQ, read 2026-09-08) —
  and appear on the consolidated trade tape but in **no exchange's book**. ⚠️ They carry FINRA's
  participant code rather than an exchange's (not re-verified at the plan specs). Reconcile a
  venue book against the consolidated tape and these are the prints at the midpoint that no book
  update explains — the §2 failure case.
- **Odd-lot prints were not on the consolidated tape** until the CTA/UTP plan amendments the SEC
  approved on ✅ 2013-11-06 (Federal Register orders 2013-26556/26557: odd-lot transactions were on
  the list "not to be reported for inclusion on the consolidated tape"). ⚠️ Implementation date
  not verified — trade counts and volume series have a break in late 2013.

Regulatory text above is engineering context, **not legal advice**: check the primary release
before relying on it, and see `../us-market-rules/SKILL.md` for the rules that decide whether a
strategy is executable at all.

## 7. Traps that produce plausible numbers

| Trap | Symptom | Fix |
|---|---|---|
| 🚨 Bar stamped at its start | strategy works on bars, dies on ticks — it traded on the bar it was still inside | index by bar END; `../signal-construction/SKILL.md` §4 |
| 🚨 Quote at the same timestamp as the trade | 100% classification accuracy in-sample, garbage live — the quote is the *post*-trade update | strictly-prior quote: `searchsorted(side="left") − 1`, or `merge_asof(allow_exact_matches=False)`; `../market-data-engineering/SKILL.md` §2 |
| 🚨 Vendor clock treated as exchange clock | Lee-Ready 82.6% not 91.0%; λ̂ flips sign | classify on `ts_event`, not `ts_recv`; sort by the clock you use — vendor delays reorder trades |
| 🚨 Float prices | "at the midpoint" equality tests fail on float64 prices and half-ticks | int64 fixed-point (DBN's 1e-9 units); `../market-data-engineering/SKILL.md` §4 |
| 🚨 Threshold from the full sample | activity bars that "knew" the day's volume | trailing estimate only |
| ⚠️ Survivorship in tick archives | vendors sell today's symbol list; delisted names and old tickers are absent from the archive you can buy | point-in-time definitions (`../market-data-sourcing/references/databento.md`); `../market-data-sourcing/SKILL.md` §2a |
| ⚠️ Round-lot / odd-lot regime | spread and NBBO series jump at 2025-11-03 and late 2013 | key the definition to the date (§6) |

## 8. Libraries (checked at PyPI and GitHub 2026-09-08)

| Package | State | Use for |
|---|---|---|
| `hftbacktest` | ✅ **2.4.4** (2025-12-10), MIT, Python ≥3.11, GitHub pushed 2025-12-23, 4,633★ | L2/L3 replay with queue-position and latency models, crypto-first examples |
| `RiskLabAI` | ✅ **3.1.0** (2026-08-26), Python ≥3.12,<3.15 | tick/volume/dollar bars and the AFML stack — see `../backtest-validation/references/afml-stack.md` |
| `databento` / `databento-dbn` | ✅ 0.86.0 / 0.69.0 (2026-09-01), Apache-2.0 | MBO/MBP/TBBO/trades, both clocks in every record — `../market-data-sourcing/references/databento.md` |
| `nautilus_trader` | ✅ 1.231.0 (2026-08-02), LGPL-3.0-or-later, Python ≥3.12 | L2/L3 book in an event-driven engine — `../backtesting-engines/references/nautilus-trader.md` |
| `mlfinlab` | 🔴 **404 on PyPI**; GitHub `hudson-and-thames/mlfinlab` last pushed 2023-10-02, licence `NOASSERTION` | the AFML bar code most tutorials paste from — not installable from PyPI today |
| `mlfinpy` | ⚠️ 0.1.2 (2024-10-09), **2 releases**, MIT, GitHub pushed 2025-01-23 | an mlfinlab fork; treat as unmaintained until it releases |
| `tardis-dev` | ✅ 5.0.0 (2026-08-23) | crypto tick and book replay; ⚠️ not evaluated here |

There is no maintained, general-purpose Python microstructure library with verified Lee-Ready /
spread / λ implementations. **The script in this skill is the reference implementation**, and it
is short because the estimators are small; the data handling is the hard part.

## 9. Scripts

`scripts/microstructure_measures.py` — parts 1-4 above from two synthetic streams with planted
answers: a 40-session vectorised trades-only stream (bars, Amihud) and a 10-session stream with a
tick-rounded market maker, four execution locations, a vendor clock and a planted λ per day
(classification, spreads, λ, OFI, VPIN). numpy + pandas only, fixed seed, ASCII output, about
20 s. Change `LAMBDAS`, `PERSIST`, `DELAY_MS` or the world definitions in `part1_bars()` to see
how the failures scale — that is what it is for.
