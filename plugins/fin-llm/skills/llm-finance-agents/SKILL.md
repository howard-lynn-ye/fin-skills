---
name: llm-finance-agents
description: >-
  What the published evidence says about LLM trading agents, and the real status of the
  frameworks. TRIGGER - TradingAgents, FinGPT, FinRobot, FinMem, FinCON, FinAgent, AlphaAgent, RD-
  Agent, AI4Finance; building or evaluating an LLM-driven trading system, a multi-agent trader, or
  a news-sentiment-to-signal pipeline; "does AI trading work"; FinBERT and financial sentiment
  models; reproducing a Sharpe from an LLM-trading paper; whether a backtest window overlaps a
  model's training cutoff. No credible evidence exists that any of it produces alpha net of costs.
  SKIP for reinforcement learning and deep learning specifically (rl-and-ml-trading) and for MCP
  servers (finance-mcp-servers).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# LLM trading agents — the evidence

## 1. The executive answer

**No credible evidence exists that an LLM trading agent produces risk-adjusted alpha net of costs.**
The literature splits cleanly:

| Claim | Status |
|---|---|
| LLMs extract *sentiment* from news better than lexicon methods | ✅ well supported |
| LLM sentiment scores correlate with subsequent returns in-sample | ✅ supported, **but heavily contaminated by lookahead** |
| That correlation survives realistic transaction costs | ❌ **fails at ~20 bps round-trip** — by Lopez-Lira & Tang's own numbers |
| Multi-agent debate beats one well-prompted agent | ❌ **not established**; the general MAD literature finds the opposite |
| Reported Sharpe ratios of 5–8 are real | ❌ artifacts of 3-month windows, 3 tickers, zero costs, a bull regime |
| LLM agents beat buy-and-hold in a contamination-free setting | ❌ margin inside noise; **in the one real-money competition, 4 of 6 models lost money** |

**The one well-designed study kills its own strategy.** Lopez-Lira & Tang (arXiv 2304.07619) is
genuinely post-cutoff and reports a gross Sharpe of 2.97. Its cost curve:
**~700% at 0 bps → >300% at 5 bps → >100% at 10 bps → unprofitable at 20 bps round-trip.**
That curve, not a point estimate, is the format this field should have adopted.

**Alpha Arena S1** (Oct 18 – Nov 3 2025, **real capital**): 4 of 6 models lost money; GPT-5 finished
**−59%**; fees alone ate **$1,654 of Qwen3's $10k**. 🚨 The widely circulated **"+79%" is a
mid-competition snapshot, not a result** — the same model finished at **+22%**.

**StockBench contradicts itself:** its abstract says *"most models struggle to outperform"* while its
results section says *"most tested models outperform."* The real edge is ~2pp over 4 months **gross**,
Sortino ~0.03, index-level drawdowns — inside noise.

## 2. 🚨 Contamination is the first thing to check, not the last

An LLM has memorized the period you are backtesting. Before any result is meaningful:

1. **State the backbone model's training cutoff against the backtest window.** If they overlap, the
   result is uninterpretable until you show otherwise.
2. **Run a date probe.** Ask the model directly about outcomes inside the test window; measure how
   its accuracy changes at the cutoff boundary. A discontinuity at the cutoff is the signature.
3. Note that the one paper arguing lookahead is *not* a big deal (Glasserman & Lin) makes that claim
   **only for headline sentiment** — it is contradicted for general forecasting by Sarkar & Vafa,
   Levy (JAR 2026), and the cutoff-collapse result in Gao et al.

`scripts/contamination_probe.py` implements the date probe. **Nothing else in the skills ecosystem
does this** — it is the cheapest check that invalidates the largest class of published results.

## 3. The evidentiary bar

Any strategy result reported through this library must supply **all** of:

1. **Model training cutoff vs backtest window**, with a contamination check.
2. **A cost-sensitivity curve** — returns at 0 / 5 / 10 / 20 / 50 bps round-trip, **not a point
   estimate.** (Lopez-Lira & Tang's is the format the field should have adopted; almost nobody has.)
3. **Factor-adjusted alpha** — CAPM minimum, preferably FF5 + momentum, with t-stats.
4. **Trial count and Deflated Sharpe Ratio** — see `backtest-validation`.
5. **Regime coverage** — at least one drawdown regime. **A single bull quarter is not a backtest.**
6. **An equal-budget single-agent baseline**, if the design is multi-agent. No finance multi-agent
   paper has run this ablation.
7. **Turnover and capacity.**

## 4. Frameworks — what they are, and what to believe

**TradingAgents** (arXiv 2412.20138) has the most legible architecture — fundamental/sentiment/news/
technical analysts, bull-bear debate, trader, risk team, portfolio manager. ✅ **From its own tables:
the Sharpe 8.21 headline is 3 tickers over 60 trading days (2024-01-01 to 2024-03-29), backbones
`o1-preview`/`gpt-4o`, and transaction costs are never mentioned anywhere in the paper.** Its
102,425 stars measure virality, not validity. Use it as a design reference and a challenger, never
as evidence.

🚨 **Multi-agent debate is unsupported in general, not just in finance.** arXiv 2502.08788 (retitled
*"Stop Overvaluing Multi-Agent Debate"*) finds MAD fails to beat single-agent chain-of-thought across
5 methods × 9 benchmarks × 4 models; arXiv 2510.20963 shows competitive MAD degenerates into cheap
talk — **exactly the bull/bear structure these frameworks use.** No finance paper has run the
equal-budget ablation.

🚨 **Framework licence landmines:** **TradingAgents-CN has a proprietary core.** **AlphaAgent,
StockAgent, FinCon and both openbb-agents repos have NO licence = all rights reserved.** **OpenBB is
AGPL-3.0** (the GitHub API reports NOASSERTION — trust the LICENSE file).

⚠️ **Vaporware check:** **FinCon (NeurIPS 2024) has no code** — an 18 KB repo with one README and a
"3–4 month" release promise now 6+ months expired. **FinAgent has no official repo.** FinNLP and
FinMem have been dead ~2 years.

**Microsoft RD-Agent** (14,480★, MIT, very active) is the most defensible of these: an offline
hypothesis→implement→feedback loop for factor and model research. 🚨 **Every candidate it generates
is a trial** and must enter the ledger — an automated search inflates the multiple-testing problem
faster than any human can.

**FinGPT / FinRL / FinNLP / FinRobot** (AI4Finance): FinGPT is a language *component*, not a trading
system; FinRobot's perception/brain/action layering is useful for report generation, but wiring its
action layer to execution carries semantic uncertainty across the order boundary.

🚨 **FinRL does not install.** ✅ Verified: its 0.3.7 wheel (2024-04-12) declares
**`requires_dist: None`**, so `pip install finrl` pulls **zero dependencies** and `import finrl` dies
on `ModuleNotFoundError`. `elegantrl` and `finrl-meta` share the failure mode and have been frozen
since 2023-02-07. See **`../rl-and-ml-trading/SKILL.md`** for the verified state of the whole RL
stack and the evidence on whether any of it works.

**Do not install these into a production environment to "evaluate" them.** Each brings its own data
sources, caches, prompts, state and execution assumptions, which destroys the auditability of
whatever you already have. Read the architecture; port the one idea you want.

## 5. FinBERT and the licence problem

🚨 **The most-downloaded financial NLP models declare no licence at all.** Verified via the
HuggingFace API (downloads = last 30 days):

| Model | Downloads/30d | Licence |
|---|---:|---|
| `ProsusAI/finbert` | **4,926,116** | 🔴 **NONE** |
| `yiyanghkust/finbert-tone` | 792,989 | 🔴 **NONE** |
| `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` | 321,764 | apache-2.0 |
| `yiyanghkust/finbert-esg` / `finbert-pretrain` | 58k / 56k | 🔴 **NONE** |
| `StephanAkkerman/FinTwitBERT-sentiment` | 70,593 | 🟢 **MIT** |

No licence = all rights reserved. **The single most-used financial sentiment model in the world has
no grant of rights.** If the work is commercial, this matters; prefer the MIT/Apache options.

## 6. Where this library sits

✅ **`anthropics/skills` (the official repo, 173,666★) contains ZERO finance, trading or investing
skills.** There is no first-party anchor; the ecosystem is entirely community-built and quality
varies by two orders of magnitude.

**Do not duplicate — cite and move on:**
- Crypto/DeFi/Solana execution, MEV, prediction markets → `agiprolabs/claude-trading-skills` (68 skills, MIT)
- RIA compliance, KYC/AML, Reg BI, GIPS, practice ops → `JoelLewis/finance_skills` (91 skills, MIT)
- Personal/SMB bookkeeping and tax → `openaccountant/skills` (44 skills, MIT)
- Leakage-safe quant ML methodology → **`ml4t/skills`** (Stefan Jansen, 60 skills, Apache-2.0) is
  already excellent. **Build on it or differentiate deliberately; do not silently re-implement it worse.**
- ⚠️ `quantskills/*` is **GPL-3.0** with vendor lock-in. `ALAGENT-HKU/x2strategy` has **no licence**.

**The gap this library fills:** the ecosystem is saturated at both ends — API wrappers and knowledge
dumps — and nearly empty in the middle, at *the methodology that decides whether a result is real*.
That is what `research-integrity-guards` and `backtest-validation` are for.

## 7. Regulatory facts that changed recently

🚨 **The Pattern Day Trader rule was ELIMINATED effective 2026-06-04** (SEC Release 34-105226),
replaced by intraday margin. **Keep PDT in the backtester for historical periods; delete it from
live paths.** Rule state must be keyed to *date*.

🚨 **Reg SHO Rule 201**: after a 10% intraday drop, short sales may only be filled **above** the
national best bid. **This breaks exactly the mean-reversion backtests that claim the most edge** —
shorting into a crash is often not executable at the price the backtest assumed.

✅ **T+1 settlement** since 2024-05-28. ✅ **The SEC's predictive-data-analytics / AI proposal was
withdrawn 2025-06-17** — do not build to it.

**Also:** the SEC Marketing Rule (206(4)-1, with March 2025 and January 2026 FAQs) governs presenting
hypothetical/backtested performance; personalized investment advice is regulated under the Investment
Advisers Act — an LLM must not give it; and market-data redistribution terms differ per vendor on
caching, redistribution, derived works and non-display use. See `references/_regulatory.md`.

⚠️ **Financial NLP training data is legally contaminated too:** Financial PhraseBank is
**CC BY-NC-SA 3.0** and FiQA is non-commercial with no formal grant — so `ProsusAI/finbert`, itself
unlicensed, is fine-tuned on non-commercial data. **SEC EDGAR is the only clean text source.**

## 8. Reference files

`references/<framework>.md` — architecture, maintenance, licence, and an honest reading of its
published results. `references/_evidence-papers.md` — the citation list with arXiv IDs and what each
paper actually measured.
