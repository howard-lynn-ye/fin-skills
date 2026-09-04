# Regulatory facts that change strategy design

Rules change. **Key every rule to a date** so a backtest over 2015 applies the 2015 rulebook and a
live path in 2026 applies the current one. The two most-cited rules in trading tutorials both moved
recently.

Verified 2026-09-04. This is engineering context, **not legal advice** — confirm against the primary
release before relying on it.

## Changed recently — most tutorials are stale

| Rule | Status | Consequence |
|---|---|---|
| **Pattern Day Trader (PDT)** | 🚨 **ELIMINATED effective 2026-06-04** (SEC Release **34-105226**), replaced by intraday margin | **Keep PDT in the backtester for pre-2026-06-04 periods; remove it from live paths.** Any strategy previously rejected for needing >3 day-trades/5 days on a <$25k account needs re-evaluating |
| **Settlement** | ✅ **T+1 since 2024-05-28** (was T+2) | Free-riding and good-faith-violation windows shortened; cash-account strategies that relied on T+2 float no longer work |
| **SEC predictive-data-analytics / AI proposal** | ✅ **WITHDRAWN 2025-06-17** | Do not design to it |
| **Marketing Rule (206(4)-1)** | In force, with **March 2025 and January 2026 FAQs** | Governs how hypothetical/backtested performance may be presented |

## Rules that shape what a strategy can even do

**Reg SHO Rule 201 (the alternative uptick rule).** After a security drops **10% intraday** from the
prior close, short sales may only be executed **above the national best bid** for the rest of that
day and the next. 🚨 **This breaks exactly the mean-reversion backtests that claim the most edge** —
shorting into a crash is frequently not executable at the price the backtest assumed. If your
strategy shorts weakness, model this or state that you did not.

**Reg T / margin.** Initial margin 50% for equities; maintenance minimum 25% (brokers set higher).
Determines achievable leverage and the size of a margin-call cascade in a drawdown.

**Wash sale (US tax).** A loss is disallowed if a substantially identical security is repurchased
within **±30 days**. Irrelevant to gross backtest returns, decisive for after-tax returns in a
taxable account, and it interacts badly with high-turnover strategies and tax-loss-harvesting logic.

**Locate and borrow.** A short requires a locate; hard-to-borrow names carry a borrow fee that can
dwarf the alpha. A short backtest without a borrow-cost model is not a backtest.

## What an LLM must not do

**Personalized investment advice is regulated** under the Investment Advisers Act. An agent may
explain mechanics, analyze data, and implement a user's stated strategy; it must not recommend
specific securities or allocations as suitable for that particular person. Say plainly that you are
not a licensed adviser and stay on the mechanics.

## Presenting backtested results

The **Marketing Rule** treats hypothetical performance — which includes backtests — as requiring
policies reasonably designed to ensure relevance to the intended audience, plus disclosure of the
criteria and assumptions used and the risks and limitations. Practically, whenever a backtest number
leaves your machine:

- Label it clearly as hypothetical, not actual trading.
- State the assumptions: costs, universe, period, rebalance frequency, and whether it is net of fees.
- Disclose the limitations — survivorship, look-ahead, the trial count.
- Do not present the best of many variants as if it were the only one tried.

The result card in `../../fin-core/skills/research-integrity-guards/scripts/result_manifest.py`
emits most of this by construction.

## Market-data licensing

Vendors differ on four separate permissions, and they are not implied by the code's licence:

| Permission | Meaning |
|---|---|
| `may_cache` | May you store the data locally at all, and for how long |
| `may_redistribute` | May you pass it to another person or system |
| `may_derive` | May you publish derived values (an index, a signal, a chart) |
| `display_only` | Is non-display / algorithmic use prohibited or separately priced |

Concrete cases: **Yahoo** is *"intended for personal use only"* — yfinance's Apache-2.0 licence
covers the code, not the data. **Tiingo**: *"you may only use the data for your own personal use and
you may not display or share the data with another person or organization."* **IBKR** market data
is per-exchange, per-account, with non-display use priced separately. **Chinese scrapers** (akshare,
efinance, adata) pull from sites whose ToS prohibit systematic extraction; 反不正当竞争法 Art. 12 and
数据安全法 have both been applied to this.

**Nobody encodes these as machine-readable metadata.** That is a real gap this library can fill.

## Non-US

Rules here are US-centric. A/H-share rules (T+1 equities, price limits by board, ST ±5%) live in
`../../fin-china/skills/china-ashare-data`. EU/UK MiFID II best-execution and research-unbundling
obligations, and PDT-equivalents in other jurisdictions, are **not covered** — check locally.
