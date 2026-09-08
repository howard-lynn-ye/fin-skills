# Backtesting engines — verified metadata

Verified 2026-09-03 against the PyPI JSON API, the GitHub REST API, and (where marked) the
libraries' own source. `[V-SRC]` = read in the source · `[V-DOC]` = stated in official docs ·
`[V-API]` = from the PyPI/GitHub API.

## Metadata

| Project | PyPI | Version | Released | ★ | Licence (actual) | Last commit | Status |
|---|---|---|---|---:|---|---|---|
| polakowo/vectorbt | `vectorbt` | 1.1.0 | 2026-07-05 | 8,978 | **Apache-2.0 + Commons Clause** `[V-SRC]` | 2026-08-02 | Active — revived, v1.0 in Apr 2026 |
| kernc/backtesting.py | `backtesting` | 0.6.6 | 2026-07-22 | 8,927 | **AGPL-3.0** | 2026-08-05 | Active |
| mementum/backtrader | `backtrader` | 1.9.78.123 | **2023-04-19** | 23,132 | GPL-3.0 | **2023-04-19** | 🔴 **Unmaintained ~3.4 yrs** |
| backtrader2/backtrader | — | — | — | 268 | GPL-3.0 | pushed 2024-03-24 | Fork, also stale |
| stefan-jansen/zipline-reloaded | `zipline-reloaded` | 3.1.1 | 2025-07-19 | 1,932 | Apache-2.0 | 2025-11-13 (dependabot only) | ⚠️ **Maintenance-only** |
| edtechre/pybroker | `lib-pybroker` | 2.0.1 | 2026-08-28 | 3,524 | **Apache-2.0 + Commons Clause** `[V-SRC]` | 2026-08-28 | Active |
| pmorissette/bt | `bt` | 1.2.0 | 2026-04-25 | 2,973 | MIT | 2026-09-01 | Active |
| enzoampil/fastquant | `fastquant` | 0.1.8.1 | 2023-01-04 | 1,755 | MIT | pushed 2023-09-15 | 🔴 Dead |
| mhallsmoore/qstrader | `qstrader` | 0.3.0 | 2024-06-24 | 3,455 | MIT | 2024-06-24 | 🔴 Dormant ~2 yrs |
| Lumiwealth/lumibot | `lumibot` | 4.5.88 | 2026-09-01 | 2,037 | **GPL-3.0** `[V-SRC]` | 2026-09-02 | Very active |
| blankly-finance/blankly | `blankly` | 1.18.25b0 | 2023-07-23 | 2,473 | LGPL-3.0 | pushed 2024-12-30 | 🔴 Dead |
| nautechsystems/nautilus_trader | `nautilus_trader` | 1.231.0 | 2026-08-02 | 28,351 | LGPL-3.0-or-later | 2026-09-03 | Very active, bi-weekly |
| QuantConnect/Lean | (engine) | — | — | 21,471 | Apache-2.0 | 2026-09-03 | Very active |
| QuantConnect/lean-cli | `lean` | 1.0.229 | 2026-08-28 | 326 | Apache-2.0 | 2026-09-02 | Very active |
| microsoft/qlib | `pyqlib` | 0.9.7 | 2025-08-15 | 48,253 | MIT | 2026-07-23 | Active, slow releases |
| microsoft/RD-Agent | — | v0.8.0 | 2025-11-03 | 14,480 | MIT | 2026-09-03 | Very active |
| freqtrade/freqtrade | `freqtrade` | 2026.8 | 2026-08-31 | 53,985 | GPL-3.0 | 2026-09-03 | Very active, monthly |
| jesse-ai/jesse | `jesse` | 3.1.0 | 2026-09-02 | 8,413 | MIT | 2026-09-02 | Very active |
| hummingbot/hummingbot | `hummingbot` | 20260729 | 2026-07-29 | 19,781 | Apache-2.0 | 2026-07-30 | Active |
| Drakkar-Software/OctoBot | `OctoBot` | 2.1.1 | 2026-03-29 | 6,510 | GPL-3.0 | 2026-08-10 | Active |
| ricequant/rqalpha | `rqalpha` | 6.3.0 | 2026-07-23 | 6,744 | **Custom — non-commercial only** `[V-SRC]` | 2026-09-01 | Active |
| vnpy/vnpy | `vnpy` | 4.4.0 | 2026-05-14 | 45,089 | MIT | 2026-08-06 | Active |
| wondertrader/wondertrader | — | — | — | 6,314 | MIT | 2026-09-01 | Active |
| wondertrader/wtpy | — | — | — | 1,487 | MIT | pushed 2025-08-06 | Slowing |

## Install constraints

| Engine | Python | Windows wheel |
|---|---|---|
| **nautilus_trader** | 🚨 **>=3.12,<3.15** | ✅ cp312/313/314 only — **will not install on 3.11** |
| hummingbot | >=3.10.12 | 🚨 **none** (sdist only) |
| rqalpha | >=3.8 | 🚨 **none, 0 wheels** |
| vectorbt, backtesting, bt, freqtrade, jesse, lumibot, pyqlib | — | ✅ pure-python |

## Maintenance verdicts

**Dead — do not start new work:**
`backtrader` (master's last commit is 2023-04-19 "Version 1.9.78.123"; repo not archived, README not
updated, forum support long gone — **23k stars makes it the most over-recommended dead library in
the domain**; its ideas survive, the code should not start a project), `fastquant` (2023), `blankly`
(2024-12; last real release a 2022 beta), `qstrader` (2024-06), `catalyst` (archived),
`mlfinlab` (off PyPI).

**Watch-list — works today, maintainer bandwidth is the risk:**
- **zipline-reloaded** — the only 2025 commits on `main` are Dependabot CI bumps. It is a
  *compatibility-maintenance* project, not a feature project. Fine as a frozen, correct equity
  backtester; not fine if you need new asset classes.
- **wtpy** — last push 2025-08.

## Licence detail that surprises people

- **Commons Clause** on `vectorbt` and `lib-pybroker` is an *addendum* to Apache-2.0: it removes the
  right to **sell** the software, which includes selling a product or service whose value derives
  substantially from it. Internal use and research are fine; a hosted product is not.
- **AGPL-3.0** on `backtesting.py` is network copyleft — serving it obliges source disclosure of
  the combined work.
- **RQAlpha's licence is custom and non-commercial only** — not any standard OSI licence.
- GitHub's licence detector reports `NOASSERTION` for several of these because the LICENSE file
  carries prepended copyright text. **Trust the LICENSE file, not the API field.**

## Architecture at a glance

| Engine | Model | Speed | Multi-asset | Live path |
|---|---|---|---|---|
| vectorbt | vectorized (Numba/Rust) | 🥇 fastest for sweeps | yes | none |
| backtesting.py | bar loop, single asset | fast | ❌ single asset | none |
| zipline-reloaded | event-driven, daily/minute | moderate | yes | none |
| PyBroker | bar loop + walk-forward | fast | yes | none |
| bt | weight/rebalance tree | fast | yes | ❌ not order-level |
| nautilus_trader | event-driven, L2/L3 book | moderate, Rust core | yes | ✅ **same code** |
| LEAN | event-driven | moderate | yes | ✅ |
| freqtrade / jesse | bar loop, crypto | fast | pairs | ✅ |

**The rule of thumb:** vectorized engines are for *searching* a parameter space; event-driven
engines are for *believing* the result. Use both — sweep with vectorbt, then re-run the survivor in
an event-driven engine and check the numbers agree. When they don't, the vectorized one is usually
the optimistic one.

## What each engine does NOT model

| | partial fills | volume cap | margin | borrow | corporate actions | delistings |
|---|---|---|---|---|---|---|
| vectorbt | ❌ | ❌ | limited | ❌ | ❌ | ❌ |
| backtesting.py | ❌ | ❌ | limited | ❌ | ❌ | ❌ |
| zipline-reloaded | ✅ | ✅ **2.5% of bar volume** | ✅ | ❌ | ✅ | ✅ |
| PyBroker | ❌ | ❌ | limited | ❌ | ❌ | ❌ |
| nautilus_trader | ✅ | ✅ book-based | ✅ | partial | ❌ | ❌ |
| LEAN | ✅ | ✅ | ✅ | ✅ | ✅ map/factor files | ✅ |
| freqtrade | ✅ | partial | ✅ (futures) | n/a | n/a | ❌ |

**Only zipline-reloaded and LEAN model an asset's lifetime.** Everything else consumes whatever
frame you hand it, so survivorship is entirely your problem — see the parent SKILL.md §2.2.

## Adjacent

- `skfolio` — BSD-3, 2,341★, v1.0.3 (2026-08-31), very active. The live successor for portfolio
  optimization + purged cross-validation.
- `quantstats` — Apache-2.0, 7,612★, pushed 2026-07-20. Tearsheets only; see `portfolio-and-risk`
  for its measured metric bugs.
- `quantopian/zipline` (original) — not archived but pushed 2024-02-13; superseded.
- `enigmampc/catalyst` (crypto zipline) — **archived**, dead since 2022.
- 🚨 **`ib_fut` does not exist** — no PyPI package, no plausible repo `[V-API]`. If a doc names it,
  that is an error; the real adjacent IB tool is **`ibflex`** (v1.1, 2026-05-23), which parses IB
  Flex XML activity/trade reports for reconciliation and tax.
