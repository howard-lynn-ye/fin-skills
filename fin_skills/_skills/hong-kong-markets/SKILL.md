---
name: hong-kong-markets
description: >-
  TRIGGER - HKEX, 港股, 每手 board lot, 碎股 odd lot, Stock Connect, 滬港通, 深港通, northbound quota, southbound quota, VCM, CAS, A/H premium, ADR ratio, typhoon trading. SKIP for Japan (japan-markets), India (india-markets), Korea or Taiwan (korea-taiwan-markets), ASEAN (asean-markets), regional selection (asia-pacific-markets), and the mainland leg (china-trading-stack).
license: MIT
compatibility: Python 3.10+; offline scripts require numpy and pandas.
metadata:
  version: "0.1.0"
  verified_on: "2026-09-14"
allowed-tools: Read Bash(python:*)
---

# Hong Kong markets

Load the issuer's dated board lot before sizing an order. Measure portfolio weights against
account NAV, including cash, and treat Connect quota exhaustion as state rather than a balance.

## Lots and measured portfolio error

✅ [HKEX securities-market FAQ](https://www.hkex.com.hk/Global/Exchange/FAQ/Securities-Market/Trading/Securities-Market-Operations?sc_lang=en),
checked 2026-09-14: issuers determine board lots. Odd lots exist and can trade in the special-lot
market; they are not accepted for automatic matching in the ordinary board-lot book. Therefore
an odd-lot position is possible, but an ordinary-book backtest cannot assume it gets that fill.
Use dated [security lists](https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx).

✅ Measured 2026-09-14 by `scripts/hong_kong.py`: 60 synthetic names with a realistic spread
of lot sizes. The recovered source-era frequency weights are fixed scenario inputs; the original
workbook is not bundled, so the script does not claim a current count of listed securities.

| Account capital (HKD) | Zero positions after rounding | Maximum NAV weight error | Unspent cash |
|---|---|---|---|
| 1,000,000 | 32 / 60 | 166.7 bp | 63.96% |
| 5,000,000 | 0 / 60 | 75.2 bp | 11.64% |
| 20,000,000 | 0 / 60 | 25.1 bp | 3.01% |
| 100,000,000 | 0 / 60 | 4.5 bp | 0.56% |

🚨 Dividing holdings by deployed capital instead of NAV produces a different statistic.
The script returns `deployed_weight` separately; rounding down does not make a surviving name
NAV-overweight. Actual weight errors are sawtoothed as capital changes: do not bisect them as
if they were monotone. `capital_for_weight_error` returns a conservative **sufficient bound**
from maximum ticket value, not the smallest feasible account. Measured bound for this scenario
and 100 bp is HKD 5,792,000.

## Sessions, settlement and weather

✅ [HKEX trading hours](https://www.hkex.com.hk/Services/Trading-hours-and-Severe-Weather-Arrangements/Trading-Hours/Securities-Market?sc_lang=en),
checked 2026-09-14: ordinary equities trade 09:30–12:00 and 13:00–16:00. Pre-opening is
09:00–09:30; closing auction ends randomly during 16:08–16:10. The extended morning window
is for designated securities, not ordinary-stock lunch trading. Half days and auction phases
must be handled separately from continuous bars.

⚠️ Dated rules retained from the 2026-09-10 audit: ordinary equity settlement is T+2; taxable
stock transfers have per-side stamp duty of 0.1%, raised to 0.13% from 2021-08-01 and restored
to 0.1% from 2023-11-17. Use security-specific exemptions and the government's current rate
schedule for actual costs; `stamp_duty_rate` is not a tax engine.

✅ [HKEX severe-weather implementation](https://www.hkex.com.hk/News/Market-Communications/2024/240618news?sc_lang=en)
announced trading through severe weather from 2024-09-23 (source audit 2026-09-10).
The script distinguishes the change. Its historical weather description is not a reconstruction
of intraday suspension timestamps: signal type, hoisting/lowering times and exchange notices
are required for that. Do not delete all typhoon days from a current calendar.

⚠️ VCM tiers retained from the 2026-09-10 HKEX audit: Large/Mid/SmallCap 10%/15%/20%, with
five-minute cooling-off. VCM is a temporary band around a reference trade, not a blanket daily
price limit or a complete halt. Security eligibility, exemptions, auction limits and current
parameters require the exchange's [VCM material](https://www.hkex.com.hk/Services/Trading/Securities/Overview/Trading-Mechanism/Volatility-Control-Mechanism?sc_lang=en).

## Connect: order value, trade value and the exhaustion latch

✅ [HKEX Connect hub](https://www.hkex.com.hk/Mutual-Market/Connect-Hub/Stock-Connect?sc_lang=zh-HK)
and [SEHK Rules, 14A/14B daily-quota clauses](https://www.hkex.com.hk/-/media/HKEX-Market/Services/Rules-and-Forms-and-Fees/Rules/SEHK/Whole_SEHK_e.pdf),
checked 2026-09-14: northbound quota is RMB 52 billion per route; southbound RMB 42 billion.
Balance subtracts **buy orders**, adds **sell trades** and released adjustments. It is not gross
turnover and not simply net executed buys. Northbound exhaustion during continuous trading
blocks new buys for the rest of the day even if the balance later recovers. Opening-auction
recovery is different; already accepted orders are unaffected.

✅ Measured: 40 bn buy orders and 35 bn sell trades leave 47 bn northbound balance. Pass
`exhausted_in_continuous=True` after exhaustion; the gate then remains closed with that positive
balance. This is a quota check only, not order approval.

🚨 Join the **published dated eligibility lists**, route, account eligibility and actual Connect
calendar. Do not infer an eligible universe merely from an index name or intersect only exchange
trading days: clearing and currency holidays also matter. Eligibility expansion changes over time.
ETFs were already included in both directions in July 2022; the recovered claim that southbound
ETFs began in 2026 was wrong ([HKEX 2022 notice](https://www.hkex.com.hk/eng/prod/dataprod/Documents/22-06-28%20IV%20notice%20SB%20Brokers%20Prog%20Update%20%28Eng%29.pdf),
checked 2026-09-14).

## A/H and ADR relationships

✅ Measured 2026-09-14: H shares at HKD 40, A shares at CNY 42.5 and HKD 1.08/CNY imply a
14.75% A/H premium. At USD 21.4 per ADR, five ordinary shares per ADR and HKD 7.8/USD imply
HKD 33.384 per ordinary share. Use the direction of the ADR ratio and contemporaneous FX.
A/H lines are not generally interchangeable through Connect; an apparent price gap alone
is not executable arbitrage. ADR conversion requires the depositary's actual programme terms.

## Where this sits

- [Regional router](../asia-pacific-markets/SKILL.md).
- [Mainland trading rules](../../../fin-china/skills/china-trading-stack/SKILL.md): the other Connect leg.
- [Market data](../../../fin-market-data/skills/market-data-sourcing/SKILL.md): dated lots and share ratios.
