---
name: us-market-rules
description: >-
  US trading rules that decide whether a strategy is executable at all - short-sale restrictions,
  margin, settlement, day-trading limits, wash sales, and what a data licence lets you keep.
  TRIGGER - can I short this, locate, hard to borrow, borrow fee, short interest, Reg SHO, uptick
  rule, SSR, short sale restricted; PDT, pattern day trader, the 25k rule, day trade limit; Reg T,
  initial or maintenance margin, margin call, buying power, leverage limit; T+1, settlement, free
  riding, good faith violation, cash account; wash sale, 30 day rule, tax-loss harvesting,
  after-tax returns; "can I redistribute this data"; may_cache, may_redistribute, non-display use,
  market data licence; presenting or publishing backtested performance, Marketing Rule,
  hypothetical performance disclosure. Two of the most-cited rules moved in 2024-2026, so a
  training-prior answer is usually stale. SKIP for A-share T+1 and price limits
  (china-ashare-data) and for the order-safety mechanics of actually sending an order
  (broker-execution-apis).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# US market rules

Engineering context, **not legal advice** — confirm against the primary release before relying on
it. Verified 2026-09-04.

**Key every rule to a date.** A backtest over 2015 must apply the 2015 rulebook; a live path in 2026
applies the current one. The two most-cited rules in trading tutorials both moved recently, so a
model answering from its training prior will usually be wrong about them.

## 1. 🚨 Changed recently — most tutorials are stale

| Rule | Status | Consequence |
|---|---|---|
| **Pattern Day Trader (PDT)** | 🚨 **ELIMINATED effective 2026-06-04** (SEC Release **34-105226**), replaced by intraday margin | **Keep PDT in the backtester for pre-2026-06-04 periods; remove it from live paths.** Any strategy previously rejected for needing >3 day-trades/5 days on a <$25k account needs re-evaluating |
| **Settlement** | ✅ **T+1 since 2024-05-28** (was T+2) | Free-riding and good-faith-violation windows shortened; cash-account strategies that relied on T+2 float no longer work |
| **SEC predictive-data-analytics / AI proposal** | ✅ **WITHDRAWN 2025-06-17** | Do not design to it |
| **Marketing Rule (206(4)-1)** | In force, with **March 2025 and January 2026 FAQs** | Governs how hypothetical/backtested performance may be presented — §4 |

## 2. 🚨 Reg SHO Rule 201 breaks the backtests that claim the most edge

After a security drops **10% intraday** from the prior close, short sales may only execute **above
the national best bid** for the rest of that day *and the next*.

**Shorting into a crash is frequently not executable at the price the backtest assumed.** This hits
mean-reversion and short-the-spike strategies precisely where they book their best trades — the
fills that carry the P&L are the ones the rule would have blocked or repriced.

If your strategy shorts weakness, model the SSR trigger or state plainly that you did not. It is not
a small correction: the constraint binds only on the days the strategy makes its money.

## 3. What a short position actually requires

**A locate.** A short sale requires the broker to locate borrowable shares before execution. In a
backtest this is a silent assumption; in life it is a hard gate that fails exactly on crowded names.

**A borrow fee.** Hard-to-borrow names carry a rate that can dwarf the alpha, and the rate is not
constant — it moves with demand, which correlates with the signal that made you want the short.

🚨 **A short backtest with no borrow-cost model is not a backtest.** See the `borrow` column in
`../backtesting-engines/references/_engine-matrix.md` — every free engine is ❌ on it, so the model
has to come from you.

⚠️ **This library does not yet carry verified sources for borrow rates or short-interest data.**
That is a known gap, not an omission by design; candidates to evaluate are the IBKR shortable-shares
feed, FINRA short interest, and SEC threshold/fails-to-deliver lists.

## 4. Reg T, margin, and buying power

Initial margin **50%** for equities; maintenance minimum **25%** (brokers set higher, and the
broker's number is the one that liquidates you). This determines achievable leverage and the size
of a margin-call cascade in a drawdown.

A backtest that levers 3x without checking whether Reg T permits it at that account type is
modelling an account that cannot exist.

## 5. Wash sales

A loss is disallowed if a substantially identical security is repurchased within **±30 days**.

Irrelevant to gross backtest returns. **Decisive for after-tax returns in a taxable account**, and
it interacts badly with high-turnover strategies and with tax-loss-harvesting logic that is trying
to do the opposite. If you report after-tax numbers, this is not optional.

## 6. Presenting backtested results

The **Marketing Rule** treats hypothetical performance — which includes backtests — as requiring
policies reasonably designed to ensure relevance to the intended audience, plus disclosure of the
criteria and assumptions used and the risks and limitations.

Whenever a backtest number leaves your machine:

- Label it **hypothetical**, not actual trading
- State costs, universe, period, rebalance frequency, and whether it is net of fees
- Disclose survivorship, look-ahead, and **the trial count**
- Do not present the best of many variants as if it were the only one tried

`../research-integrity-guards/scripts/result_manifest.py` emits most of this by construction.

## 7. 🚨 Market-data licensing is not the code's licence

Vendors differ on four separate permissions, and **none of them is implied by the package's open
source licence**:

| Permission | Meaning |
|---|---|
| `may_cache` | May you store the data locally at all, and for how long |
| `may_redistribute` | May you pass it to another person or system |
| `may_derive` | May you publish derived values (an index, a signal, a chart) |
| `display_only` | Is non-display / algorithmic use prohibited or separately priced |

Concrete cases: **Yahoo** is *"intended for personal use only"* — yfinance's Apache-2.0 licence
covers the code, not the data. **Tiingo**: *"you may only use the data for your own personal use and
you may not display or share the data with another person or organization."* **IBKR** market data is
per-exchange, per-account, with non-display use priced separately. **Chinese scrapers** (akshare,
efinance, adata) pull from sites whose ToS prohibit systematic extraction; 反不正当竞争法 Art. 12 and
数据安全法 have both been applied to this.

**Nobody encodes these as machine-readable metadata.** That is a real gap this library could fill.

## 8. Non-US

Rules here are US-centric.

- **A/H-shares** — T+1 equities, price limits by board, ST ±5% — live in
  `../../../fin-china/skills/china-ashare-data/SKILL.md`
- **Japan, Korea, HK, India** — `../../../fin-asia/skills/asia-pacific-markets/SKILL.md`
- **EU/UK MiFID II** best-execution and research-unbundling obligations, and PDT-equivalents in
  other jurisdictions, are **not covered** — check locally

## 9. Where this bites in the rest of the library

| If you are | Read this because |
|---|---|
| Sending orders (`../broker-execution-apis/SKILL.md`) | §1 settlement and §4 margin decide what the account can actually do |
| Choosing a backtest engine (`../backtesting-engines/SKILL.md`) | §2 and §3 are constraints no free engine models for you |
| Auditing a result (`../research-integrity-guards/SKILL.md`) | §6 is the reporting standard; §2 is a common reason a short-side edge is fictional |
| Building an LLM trading agent (`../../../fin-llm/skills/llm-finance-agents/SKILL.md`) | Personalized investment advice is regulated under the Investment Advisers Act — explain mechanics, do not recommend securities or allocations as suitable for a particular person |
