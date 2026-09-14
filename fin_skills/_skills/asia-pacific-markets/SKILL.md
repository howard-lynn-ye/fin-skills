---
name: asia-pacific-markets
description: >-
  TRIGGER - choose an Asia-Pacific data or trading stack outside mainland China, compare Asian venues, multi-market calendars, regional survivorship and currency alignment. SKIP for Japan (japan-markets), Hong Kong or Connect (hong-kong-markets), India (india-markets), Korea or Taiwan (korea-taiwan-markets), Southeast Asia (asean-markets), and mainland A-shares (china-ashare-data, china-trading-stack).
license: MIT
compatibility: Read-only market selection; linked offline scripts require Python 3.10+.
metadata:
  version: "0.2.0"
  verified_on: "2026-09-14"
allowed-tools: Read Bash(python:*)
---

# Asia-Pacific market router

Choose the venue before choosing a Python wrapper. The same symbol string can identify a
different currency, share line, board lot, calendar or investor entitlement. Use the dated
market skill for the actual mechanism; the offline demonstrations are not live market feeds.

| Task | Read next | Failure to check |
|---|---|---|
| Japan, JPX, J-Quants, Nikkei/TOPIX | [japan-markets](../japan-markets/SKILL.md) | Lunch and closing auctions, tick class, reference prices, price adjustment factors |
| Hong Kong, Stock Connect, A/H or ADRs | [hong-kong-markets](../hong-kong-markets/SKILL.md) | Per-security lots, cash-inclusive weights, quota state, share ratios |
| India, NSE/BSE, broker APIs | [india-markets](../india-markets/SKILL.md) | Per-security price bands, auction eligibility, dated taxes and index-contract rules |
| Korea/Taiwan, KRX/TWSE | [korea-taiwan-markets](../korea-taiwan-markets/SKILL.md) | Historical short-sale eligibility, limit prices versus available liquidity |
| Singapore, Malaysia, Thailand, Indonesia, Philippines, Vietnam | [asean-markets](../asean-markets/SKILL.md) | Lunch/Friday windows, dated settlement, investor ownership room and share lines |

For mainland China use `china-ashare-data` and `china-trading-stack`. Australian or New
Zealand work needs its own current exchange evidence; none of these demonstrations supplies
an ASX/NZX calendar. For package-specific instructions, opt into `fin-libraries`.

## Checks common to a regional study

- Keep a point-in-time security master: listings, delistings, mergers, renamed tickers,
  share classes, board lots and investor eligibility. A current ticker list cannot establish
  historical universe coverage. An empty response is not evidence that a security never existed.
- Store the original exchange timezone, holiday calendar and auction/bar convention.
  UTC alignment and forward filling can make a closed market appear contemporaneous with
  an open one. A library calendar is an implementation to verify against dated notices.
- Distinguish adjusted return series from executable quotes. Split/dividend adjustments,
  corporate-action base prices, available queues, ownership gates and borrow availability
  belong to different layers. Do not use a synthetic adjustment as an order price.
- Record data publication time and revision time. A historical fiscal period is not the
  date its financial statement became available. Check a provider's actual revision and
  delisting coverage instead of assuming that “historical” means point in time.
- Confirm the provider's account, entitlement, redistribution licence and service limits.
  A Python client does not create an exchange-data licence or cross-border trading access.

Verification scope: the linked market skills distinguish primary sources re-read on
2026-09-14, earlier repository audits, unavailable sources and synthetic model assumptions.
They contain the reproducible scripts and source-specific limitations.

Related handoffs: [mainland trading](../../../fin-china/skills/china-trading-stack/SKILL.md)
and [market-data sourcing](../../../fin-market-data/skills/market-data-sourcing/SKILL.md).
