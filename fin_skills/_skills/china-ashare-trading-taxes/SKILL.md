---
name: china-ashare-trading-taxes
description: >-
  A-share stamp duty is charged to the seller only and halved on 2023-08-28, and dividend tax is a
  step function of holding period - a turnover penalty written into the tax code that a flat
  symmetric cost model cannot express. TRIGGER - A股印花税, 证券交易印花税, stamp duty, stamp tax,
  seller side only, 单边征收, 印花税减半, 0.05%, 2023-08-28, A股股息红利税, 股息红利差别化,
  dividend tax by holding period, 持股期限, 一个月, 一年, 财税〔2015〕101号, dividend capture
  China, A-share transaction cost model, "what does an A-share round trip actually cost",
  backtesting Chinese equities net of tax. Modelling assumptions for backtests, not tax advice,
  and for an individual resident investor only. SKIP for US wash sales and lot matching
  (wash-sale-rules, tax-lot-matching-and-cost-basis), for reporting an after-tax Sharpe
  (after-tax-backtesting), and for A-share data, T+1 and price limits (china-trading-stack).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# China A-share trading taxes

**Two taxes decide what an A-share round trip costs, and a flat symmetric basis-point model can
express neither of them. One is charged to the seller only and halved in the middle of any sample
that spans 2023-08-28. The other charges you for holding briefly.**

These are **modelling assumptions for a backtest, not tax advice**, and everything here is for an
**individual resident investor**. Institutions, funds, QFII/RQFII and Stock Connect investors are
treated differently and none of that is in this file. Confirm current law with a qualified
professional before relying on any of it.

✅ Measured comes from `scripts/ashare_taxes.py` — numpy + pandas, seed 20260909, one seeded
2021→2025 path that deliberately spans the rate cut, **under 1 s**. ✅ source-verified material
was read on **2026-09-09** at 国家税务总局 (`fgk.chinatax.gov.cn`, `www.chinatax.gov.cn`) and
财政部 (`www.mof.gov.cn`).

## 1. ✅ source-verified — stamp duty is one-sided, and it moved

**中华人民共和国印花税法, Article 3**, in force from **2022-07-01**, verbatim:

> 本法所称证券交易，是指转让在依法设立的证券交易所、国务院批准的其他全国性证券交易场所交易的股票和
> 以股票为基础的存托凭证。**证券交易印花税对证券交易的出让方征收，不对受让方征收。**

*Securities transaction stamp duty is levied on the transferor, and not on the transferee.* The
attached 印花税税目税率表 sets the rate at **千分之一 (0.1%) of 成交金额**, and Article 5 makes the
transaction amount the tax base.

**财政部 税务总局公告2023年第39号《关于减半征收证券交易印花税的公告》**, issued 2023-08-27,
verbatim: *"为活跃资本市场、提振投资者信心，**自2023年8月28日起，证券交易印花税实施减半征收**。"*
0.1% × 50% = **0.05%, still seller-side only.**

**财政部 release of 2008-09-19**, verbatim: *"调整为单边征税，即对买卖、继承、赠与所书立的A股、B股
股权转让书据的**出让方**按千分之一的税率征收证券（股票）交易印花税，**对受让方不再征税**。"* That is
the date the duty stopped being two-sided. ⚠️ The document number is widely reported as
**财税明电〔2008〕2号**; the MOF page itself does not carry it, so treat the number as secondhand
and the substance as verified. (⚠️ The research this skill started from named 财税〔2008〕96号 —
that is wrong.)

| effective from | rate | charged to | authority |
|---|---|---|---|
| 2008-09-19 | 0.10% | **seller only** | 财政部 release 2008-09-19 |
| 2022-07-01 | 0.10% | **seller only** | 印花税法 art. 3 + 税目税率表 |
| **2023-08-28** | **0.05%** | **seller only** | **财政部 税务总局公告2023年第39号** |

🚨 **Before 2008-09-19 the duty was two-sided and the rate moved several times** (it was raised to
0.3% in 2007). This library did **not** verify those rates, so `stamp_duty_rate()` **raises** for
any date before 2008-09-19 rather than extrapolating the table backwards.

⚠️ **ETFs and funds are outside the tax.** Article 3 defines 证券交易 as transferring **股票 and
share-based 存托凭证**. A fund unit is neither. ✅ Measured: `stamp_duty(1_000_000, "S", ...,
instrument="etf")` returns **0.00** where the same trade in shares returns **500.00**. A strategy
implemented through an A-share ETF has a genuinely different tax structure from the same strategy
in the constituents — which is a real reason to prefer one implementation, and one no
constituent-level backtest sees.

## 2. ✅ Measured — what a flat symmetric model gets wrong

Seeded path **2021-01-04 → 2025-09-05**, 100,000 shares, round-tripped at several frequencies.
"Flat 10 bps/side" is the model most A-share backtests use; "real stamp duty" is the tax alone.

| hold (days) | trades | gross notional | flat 10 bps/side | **real stamp duty** | ratio |
|---|---|---|---|---|---|
| 5 | 487 | 847,924,422 | 847,924 | **326,019** | 2.6x |
| 21 | 117 | 203,666,730 | 203,667 | **77,042** | 2.6x |
| 63 | 39 | 68,059,289 | 68,059 | **24,944** | 2.7x |
| 244 | 9 | 14,985,663 | 14,986 | **4,681** | 3.2x |
| never | 1 | 2,086,219 | 2,086 | **0** | n/a |

🔑 **The point is not that 10 bps is too big.** A calibrated 10 bps a side is a defensible *total*
cost — commission, 过户费, spread and impact all live in there too. The point is the **structure**:

- **It is symmetric and the tax is not.** The buy side pays **zero** duty. A model that charges
  both sides equally misprices any strategy whose buys and sells are not balanced — an
  accumulating book, a liquidating book, or any comparison between a long-only sleeve and a
  turnover-matched one.
- **It is constant and the tax is not.** ✅ Measured on the same monthly blotter, split at the
  cut date: **10.0 bps of sell notional before 2023-08-28, 5.0 bps from it.** A sample that spans
  that Monday contains two cost regimes, and a single number is wrong on one side of it whichever
  number you choose. Backtests run before and after 2023 are not comparable on cost without
  saying which rate they used.

⚠️ **Not modelled here, because they are fees rather than taxes**: 佣金 (broker commission, with
its per-order minimum), 过户费 (transfer fee, charged on both sides), 规费, and the T+1 constraint
that changes what a "round trip" even means. Those belong to
`../../../fin-china/skills/china-trading-stack/SKILL.md`.

## 3. ✅ source-verified — the dividend tax is a step function of holding period

**财税〔2015〕101号**《财政部 国家税务总局 证监会关于上市公司股息红利差别化个人所得税政策有关问题
的通知》, issued 2015-09-07, **effective for record dates from 2015-09-08**:

| 持股期限 (holding period) | inclusion in taxable income | statutory rate | **effective rate** |
|---|---|---|---|
| **超过1年** (over one year) | exempt | — | **0%** |
| **1个月以上至1年（含1年）** | 50% | 20% | **10%** |
| **1个月以内（含1个月）** | 100% | 20% | **20%** |

🚨 **Both boundaries are inclusive on the *taxed* side.** ✅ Measured — the boundary is one day
wide:

| bought | sold | bucket | rate |
|---|---|---|---|
| 2024-01-10 | 2024-02-10 | `<=1m` | **20%** |
| 2024-01-10 | 2024-02-11 | `1m-1y` | **10%** |
| 2024-01-10 | **2025-01-10** | `1m-1y` | **10%** — exactly one year is still taxed |
| 2024-01-10 | 2025-01-11 | `>1y` | **0%** |

🔑 **Calendar months, not 30 days.** `holding_bucket()` uses `pd.DateOffset`, so a purchase on
2024-01-31 sold on 2024-02-29 is inside one month while a 30-day counter would say otherwise.

⚠️ **The timing is the part that breaks a cash-flow model.** 财税〔2015〕101号 defers the
withholding: the listed company withholds **nothing** at payment, and 中国证券登记结算公司 computes
and withholds the tax **when the shares are sold**, using the holding period on a first-in
first-out basis. So the dividend arrives **gross**, the tax comes out of a **later** sale, and a
backtest that nets the tax on the ex-date has both the amount and the date wrong.

## 4. ✅ Measured — the turnover penalty is in the tax code

Same seeded path, same 2.5% dividend yield, same one dividend a year, same shares. The only
difference is how often the position is round-tripped.

| hold (days) | bucket | gross dividend | tax | **effective rate** | **drag, bps/yr** |
|---|---|---|---|---|---|
| 5 | `<=1m` | 234,380 | 46,876 | **20.0%** | **48.1** |
| 21 | `<=1m` | 234,380 | 46,876 | **20.0%** | **48.1** |
| 63 | `1m-1y` | 234,380 | 23,438 | **10.0%** | **24.1** |
| 244 | `1m-1y` | 234,380 | 23,438 | **10.0%** | **24.1** |
| **never** | `>1y` | 234,380 | **0** | **0.0%** | **0.0** |

🚨 **48 basis points a year, on a 2.5% yield, purely for rebalancing monthly instead of not
rebalancing.** That is larger than the stamp duty on the same strategy, it does not appear in any
transaction-cost model, and it does not shrink as you trade more cheaply — it is not a cost of
trading, it is a cost of *not holding*.

🔑 **The step is where the design lives.** Moving a monthly rebalance to a 32-day one takes the
dividend rate from 20% to 10%; moving an annual rebalance one day later takes it from 10% to 0%.
If a strategy's holding period is anywhere near a boundary, the boundary is worth more than most
of its signal work — and it is a schedule change, not a signal change.

⚠️ **A dividend-capture strategy is the extreme case and the arithmetic is brutal**: it collects
the dividend, pays 20% of it, and eats the ex-date drop in full. Model both halves before
concluding anything about it.

## 5. What this skill does not cover

| Not here | Why |
|---|---|
| **Capital gains on A-shares** | Individual resident investors' gains on listed A-shares have been **exempt** under a long-running temporary policy rather than a permanent rule. This library did not verify its current status on 2026-09-09, so the module **does not model capital gains tax at all** rather than assuming zero. Check it |
| Institutions, funds, QFII/RQFII, Stock Connect | Different regimes entirely, including withholding treaties |
| 佣金, 过户费, 规费, T+1 | Fees and market structure — `../../../fin-china/skills/china-trading-stack/SKILL.md` |
| B shares, H shares, NEEQ (新三板) | Different rules; 财税〔2015〕101号's NEEQ paragraph (Article 4) was repealed by **公告2019年第78号** |
| 限售股 (restricted shares) | Different dividend and transfer rules |

⚠️ **Dated claim, re-check before use.** Everything above was verified on **2026-09-09**. The
stamp duty rate in particular is an active policy lever — it was cut in 2008 and halved again in
2023, both times as market support — so it is exactly the kind of number a model's training prior
will get wrong. The most recent related instrument this library found is 财政部 税务总局 中国证监会
公告2026年第8号 (2026-01-14), which extends the innovative-enterprise CDR pilot policies through
2027 **by reference to 财税〔2015〕101号** and changes neither the dividend brackets nor the duty
rate.

## 6. Scripts and where this sits

`scripts/ashare_taxes.py` — `stamp_duty` / `stamp_duty_rate` (dated table, seller side only,
raises before 2008-09-19, zero for funds), `rate_history_table`, `holding_bucket`,
`dividend_tax_rate` / `dividend_tax`, `flat_symmetric_cost`, `real_stamp_duty`,
`compare_cost_models`, `dividend_drag`, and the seeded `make_ashare_path` /
`make_turnover_blotter`. numpy + pandas, seed 20260909, under 1 s.

- A-share data, T+1, price limits, suspension and the broker stack —
  `../../../fin-china/skills/china-trading-stack/SKILL.md` and
  `../../../fin-china/skills/china-ashare-data/SKILL.md`.
- Reporting the result with the jurisdiction and rate attached, so it is comparable to a US
  number — `../after-tax-backtesting/SKILL.md`, whose guard exists for exactly this.
- The US analogues: lot matching (`../tax-lot-matching-and-cost-basis/SKILL.md`), the wash sale
  (`../wash-sale-rules/SKILL.md`), and 60/40 (`../section-1256-and-derivatives-tax/SKILL.md`) —
  **none of which has an A-share equivalent**, which is itself the reason to keep the
  jurisdiction in the report line.
- Turning any of these drags into a breakeven and a capacity limit —
  `../../../fin-core/skills/backtest-validation/scripts/cost_curve.py`.
