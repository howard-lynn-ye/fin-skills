# The evidence base — what each paper actually measured

Verified 2026-09-04. ✅ = checked at the source · ⚠️ = secondhand · ❓ = unverified.
Read the "what it measured" column before citing anything here.

## Lookahead and memorization

| Paper | ID | What it actually measured |
|---|---|---|
| Sarkar & Vafa, *Lookahead Bias in Pretrained Language Models* | ICML 2025; SSRN **4754678** | LLMs recall post-cutoff outcomes for the periods they were trained on. Establishes the mechanism |
| Glasserman & Lin | arXiv **2309.17322** | ⚠️ Finds lookahead is **limited for headline sentiment specifically** — routinely over-generalized to "lookahead isn't a problem". It does not say that |
| Levy | *J. Accounting Research* 2026, DOI **10.1111/1475-679X.70058** | Direct rebuttal of Kim/Muhn/Nikolaev's "GPT-4 beats analysts" on memorization and numerical-reasoning grounds |
| Kim, Muhn & Nikolaev | arXiv **2407.17866** | The claim Levy contests: LLM financial-statement analysis beating analysts |
| Gao et al. | arXiv **2512.23847** | Defines **"Lookahead Propensity" (LAP)**, which **collapses to ~zero immediately after the training cutoff** — the cleanest available contamination diagnostic |
| *Profit Mirage* | arXiv **2510.07920** | *"dazzling back-tested returns evaporate once the model's knowledge window ends"* |

**The practical takeaway:** run a LAP-style date probe before believing any LLM backtest.
`../scripts/contamination_probe.py` implements it.

## Does the trading actually work

| Study | What it measured | Verdict |
|---|---|---|
| Lopez-Lira & Tang | arXiv **2304.07619** | Genuinely post-cutoff; gross Sharpe 2.97. ✅ Cost curve: **~700% @0bp → >300% @5bp → >100% @10bp → unprofitable @20bp round-trip.** The best-designed study in the space, and it kills its own strategy |
| TradingAgents | arXiv **2412.20138** | ✅ From its own tables: Sharpe up to **8.21 on 3 tickers over 60 trading days (2024-01-01→2024-03-29)**, backbones `o1-preview`/`gpt-4o`, and **transaction costs are never mentioned in the paper**. 102,425★ measures virality |
| StockBench | — | 🚨 **Abstract says "most models struggle to outperform"; results say "most tested models outperform."** Real edge ≈2pp over 4 months **gross**, Sortino ~0.03, index-level drawdowns |
| **Alpha Arena S1** | nof1.ai, 2025-10-18→11-03, **real capital** | ✅ **4 of 6 models lost money; GPT-5 −59%; fees alone ate $1,654 of Qwen3's $10k.** 🚨 The circulated **"+79%" is a mid-competition snapshot** — the same model finished **+22%** |

## Multi-agent debate

| Paper | ID | Finding |
|---|---|---|
| *Stop Overvaluing Multi-Agent Debate* | arXiv **2502.08788** | MAD fails to beat single-agent chain-of-thought across **5 methods × 9 benchmarks × 4 models** |
| Competitive MAD degeneration | arXiv **2510.20963** | Competitive debate degenerates into cheap talk — **exactly the bull/bear structure the finance frameworks use** |

🚨 **No finance multi-agent paper has run the equal-compute-budget single-agent ablation.** Until one
does, "the debate helped" is indistinguishable from "we spent more tokens".

## Backtest validity (general, not LLM-specific)

| Work | Where | Use |
|---|---|---|
| Bailey & López de Prado, *The Deflated Sharpe Ratio* | SSRN **2460551** | DSR / expected max Sharpe under N trials |
| Bailey, Borwein, López de Prado & Zhu, *Pseudo-Mathematics and Financial Charlatanism* | *Notices of the AMS* (⚠️ year unverified, cited as 2014) | The multiple-testing argument |
| *The Probability of Backtest Overfitting* | *J. Computational Finance* (⚠️ cited as 2016) | PBO via CSCV |
| López de Prado, *The 10 Reasons Most Machine Learning Funds Fail* | *JPM* **44(6), 120–133**; SSRN **3104816** | The canonical failure taxonomy |
| Arian, Norouzi & Seco | *Knowledge-Based Systems* **305** (2024), doi **10.1016/j.knosys.2024.112477** | ✅ **CPCV beats K-Fold, Purged K-Fold and especially Walk-Forward** on PBO and DSR in a synthetic controlled environment. ⚠️ Lead author also authors `RiskLabAI` — peer-reviewed but not disinterested |

## Time-series foundation models on prices

| Source | Finding |
|---|---|
| GIFT-Eval | Only **Chronos-2** and **TimesFM-2.5** beat classical AutoTheta at high frequency; benchmark contamination documented |
| Realized-volatility study | arXiv **2607.05291** — 9 TSFMs vs 8 econometric specs, 50 assets: **only Tiny Time Mixers beat Log-HAR, narrowly**, and Mincer-Zarnowitz recalibration showed most of the edge was **better scaling, not better dynamics** |
| M5 competition | *Int. J. Forecasting* — **Prophet among the worst performers**; still widely recommended |

**Treat TSFMs as a baseline to beat, not a source of alpha.** HAR on realized volatility remains hard
to beat.

## Data licensing traps in financial NLP

- `ProsusAI/finbert` — **4.93M downloads/month, NO licence declared**, and fine-tuned on Financial
  PhraseBank which is **CC BY-NC-SA 3.0**.
- FiQA is non-commercial with no formal grant.
- `Salesforce/moirai-*` weights are **`cc-by-nc-4.0`** (all variants, ✅ verified via the HF API).
- `NX-AI/TiRex` is under the **NXAI Community License**.
- ✅ **SEC EDGAR is the only unambiguously clean text source** in this domain.

## The scorecard

| Claim | Holds up? |
|---|---|
| LLMs extract sentiment better than lexicons | ✅ yes |
| LLM sentiment correlates with forward returns | ⚠️ yes in-sample, heavily contaminated |
| …and survives costs | ❌ dies at ~20 bps round-trip |
| Multi-agent debate > single agent | ❌ not established; general literature says no |
| Sharpe 5–8 from LLM agents | ❌ 3 tickers, 60 days, zero costs, bull regime |
| LLM agents beat buy-and-hold | ⚠️ inside noise where measured honestly |
| LLM agents make money with real capital | ❌ 4 of 6 lost in the one real-money test |
