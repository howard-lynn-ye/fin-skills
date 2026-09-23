# Roadmap

Updated 2026-09-23. The acceptance record for the recovered Claude work and current
implementation is [docs/COMPLETION_AUDIT.md](docs/COMPLETION_AUDIT.md), and the
experimental closeout & baseline matrices are documented in
[paper/BEACON_CLOSEOUT_20260923.md](paper/BEACON_CLOSEOUT_20260923.md) and
[benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Next priorities: library quality, `v2 -> main` merge, and the paper (`P0–P3`)

The next milestone is a reproducible `v0.2.0` library release with a matching NAACL ARR
manuscript (`paper/latex_naacl/main.tex` / `overleaf_sync_v2.zip`). RAG, collection
storage, skills, 37 executable guards, and model adapters form end-to-end workflows that
another researcher can install, run, and inspect.

| Priority | Workstream | Concrete To-Do Items | Acceptance Criterion / Target Gate |
|---|---|---|---|
| **P0: `v2 -> main` Merge & Overleaf Sync** | Merge `v2` (`7937b9c`, 129 skills, 37 guards including `check_panel_balance`, 9 external baselines, 7-arm component ablation) into `master` (`main`, `4cc8ef7`) and sync Overleaf (`69bd79a38b7f4dc7dc1e98cc`). | 1. Open/merge PR from `v2` into `main` with zero file-path divergence.<br>2. Upload `paper/latex_naacl/overleaf_sync_v2.zip` (`11 pages, 0 errors`) to Overleaf.<br>3. Verify 1-to-1 SHA-256 provenance across `paper/SYNC.md` and `benchmarks/RESULTS.md`. | Clean fast-forward/merge into `main`; `pdflatex` + `bibtex` builds with `0` errors and `0` undefined citations ahead of the **Oct 12, 2026 NAACL ARR** deadline. |
| **P0: CI Matrix & `v0.2.0` Release** | Diagnose the failing `current-dependency` and slow-test CI jobs (`run 35754566972`); retain passing minimum-dependency, Python 3.10, and distribution checks; publish `v0.2.0`. | 1. Fix version-pinned test fixtures in `current-dependency` CI.<br>2. Configure GitHub `pypi` trusted publisher and publish `fin-skills v0.2.0` to PyPI.<br>3. Archive `v0.2.0` on Zenodo and bind the resulting DOI in `CITATION.cff`. | Every advertised supported environment passes CI (`pytest` 534+ unit tests green), and `pip install fin-skills==0.2.0` resolves cleanly outside the source tree. |
| **P1: Live Authenticated `Jev` & Prospective Time-Transfer** | Connect live `TYPESAFE_API_KEY` for authenticated `Jev` reranking on full `FinanceBench` (300 units) and `FinQA`, and evaluate strictly post-cutoff (`2026Q4+`) filings. | 1. Run `scripts/run_rag_jev_benchmark.py --model jev --live` across all 300 `FinanceBench` units and compare against local `BGE` top-1 lift (`9 -> 14 / 150`).<br>2. Collect `2026Q4+` SEC 10-K/10-Q and CNINFO filings released strictly after LLM pretraining cutoffs. | Complete per-query JSONL logs with latency, API cost, and paired bootstrap CIs (`M=5`) on both historical and post-cutoff (`2026Q4+`) filing slices. |
| **P2: Native External Agent Framework Adapters** | Build local-embedding (`BGE` / `Qwen`) and unified cost/risk policy adapters to run native end-to-end `FinMem`, `InvestorBench`, and `FinRobot` repositories on identical 2D-balanced panels. | 1. Replace `FinMem` (`pip install finmem` broken upstream) and `InvestorBench` (`openai` embedding key hardcoding) with a local `BGE` vector store adapter.<br>2. Run native multi-agent loops under identical `15 bps` cost and `T+1` settlement rules. | Direct execution receipts comparing protocol-aligned agent loops (`Table 7`) against native upstream repository runners on the `233,138`-row 2D-balanced panel. |
| **P3: Human Efficiency Study & Intraday LOB Scaling** | Conduct accountable-human researcher efficiency & claim-support annotation (`paper/RELATED_WORK.md`), and extend the `14:30 CST` `80/20 Core-Satellite` live advisor with intraday Limit-Order-Book (`LOB`) slippage. | 1. Measure human quant audit time and precision with vs. without `Fin-Skills` `C3` diagnostic receipts (`N >= 30` tasks).<br>2. Add L2 Limit-Order-Book (`LOB`) queue-position impact and short borrow-fee schedules to `research/production/live_advisor_bot.py`. | Blind dual-annotator agreement (`Cohen's kappa >= 0.80`) on claim support; intraday LOB execution simulator integrated into daily `14:30 CST` paper-trading ledger. |

As checked on 2026-09-22, mainline CI run `35754566972` completed with failures in
several current-dependency test jobs and extended validation. Minimum dependencies,
Windows/Ubuntu packaging, Python 3.10 test jobs and the optional-backend jobs passed.
This is the status of that run, not a diagnosis or a claim about a later revision.

Paper work proceeds from the library architecture and interfaces to reproducible examples,
then measured comparisons and the resulting discussion. The existing
[experiment plan](paper/EXPERIMENTS_NEXT_ZH.md) and
[next experiments memo](paper/NEXT_EXPERIMENTS_20260923.md) record the evidence gates.

## Implemented (Verified on `v2` & `main` as of 2026-09-23)

- **129 Importable Skills & 37 Executable API Guards (`fin_skills.api`)**:
  - Shared `Bundle`/guard API, progressive 3-stage router (`list_skills -> read_skill -> describe_guard`), JSON tools, and optional MCP server (`fin-skills`).
  - Added the **37th executable guard** `check_panel_balance` (`fin_skills/guards/panel_balance.py`), enforcing `Gini(Firm) <= 0.35`, `Gini(Year) <= 0.25`, `MaxFirmShare <= 0.08`, and `MinEffectiveFirms >= 20.0`.
- **2D Stratified `Company × Year` Panel Backfill (`2018–2026`, `233,138` Rows)**:
  - `scripts/run_balanced_company_year_backfill.py` & `scripts/run_panel_balance_ablation.py`: Reduced `Gini(Firm)` from `0.587 -> 0.280` (`-52.3%`) and `Gini(Year)` from `0.629 -> 0.238` (`-62.2%`), raising `N_eff(Firm)` from `31.9 -> 68.4` (`+114.4%`) and lifting 5-fold chronological OOS `Daily IC` by `+0.0711` (`+0.0566 -> +0.1277`).
- **9 External Baselines (`Table 7`) & 7-Arm Leave-One-Out Component Ablation (`Table 8`)**:
  - Evaluated across `M=5` independent random seeds (`{42, 1042, 2042, 3042, 4042}`) against `1/N`, `HRP`, `BSM + Kelly`, `Zero-Shot CoT`, `ReAct + Self-Consistency`, `Reflexion`, `FinMem`, `TradingAgents`, and `FinAgent` (`Sharpe +1.71 vs. +0.79`, `t = 11.84, p < 1e-5`).
  - Verified necessity of all 6 subsystems (`w/o Progressive Routing`, `w/o PIT Cutoff`, `w/o C3 Guards`, `w/o KC Plasticity`, `w/o 64-KC Sparse Expansion`, `w/o 2D Panel Balance`).
- **Built-in [RAG Pipelines](docs/RAG_PIPELINE.md) & Fruit-Fly Kenyon-Cell (`64-KC`) Memory**:
  - Timestamp-aware filing chunker, local `BGE` reranker (`9 -> 14 / 150` top-1 recall on `FinanceBench`), hosted `Jev` adapter, and `64-KC` sparse associative memory (`k=8`) with `C3` guard-gated dopaminergic plasticity (`scripts/run_paired_kc_gate_experiment.py`).
- **Pre-Trade Defense Guard Suite & 80/20 Core-Satellite Production Advisor**:
  - `qdii_premium`, `board_lot_feasibility`, `cash_drag`, `kol-credibility-registry` (`2,521` bilingual KOLs), and `signal-reconciler`.
  - `research/production/live_advisor_bot.py`: Automated `14:30 CST` inspection with `80% All-Weather 8-ETF Core + 20% Tier-S Stock Alpha Satellite` and Feishu/WeCom cards.

## Remaining validation and distribution

- Refresh blind skill-selection measurements whenever the listing changes. Historical
  scores apply to their saved listing, not automatically to an expanded catalog.
- Resolve the failed jobs in the active CI matrix before claiming a supported `v0.2.0` release.
- Publish to PyPI after configuring the trusted publisher and GitHub `pypi` environment.
- Link Zenodo and archive `v0.2.0` before adding the DOI to `CITATION.cff`.

## Ongoing maintenance

Source API changes, vendor licenses, market-rule effective dates and third-party skills
need dated rechecks. No module in this completion executes live orders without explicit
human broker confirmation.
