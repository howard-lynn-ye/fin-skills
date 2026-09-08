# The mainstream finance agent systems — verified 2026-09-08

Everything below was read from the GitHub REST API, PyPI's JSON API, or the repository's own
source on 2026-09-08. Stars are a popularity signal, not a quality signal; they are reported
because people choose by them. ✅ = read at the source · ⚠️ = secondhand or not run · 🚨 = a
trap · 🔴 = dead, missing or misnamed.

## 1. Repository statistics, exactly as fetched

`gh api repos/<owner>/<repo>` → `stargazers_count, forks_count, pushed_at, archived,
license.spdx_id, created_at`.

| Repo | ★ | Forks | pushed_at | Archived | Licence (API) | Created |
|---|---:|---:|---|---|---|---|
| `TauricResearch/TradingAgents` | 103,315 | 19,898 | 2026-09-07 | false | Apache-2.0 | 2024-12-28 |
| `virattt/ai-hedge-fund` | 63,289 | 11,116 | 2026-09-03 | false | MIT | 2024-11-29 |
| `HKUDS/Vibe-Trading` | 33,045 | 5,389 | 2026-09-08 | false | MIT | 2026-04-01 |
| `hsliuping/TradingAgents-CN` | 31,650 | 6,649 | 2026-07-24 | false | NOASSERTION (mixed, see §3) | 2025-06-26 |
| `AI4Finance-Foundation/FinGPT` | 21,225 | 3,008 | 2026-09-08 | false | MIT | 2023-02-11 |
| `microsoft/RD-Agent` | 14,550 | 1,889 | 2026-09-04 | false | MIT | 2024-04-03 |
| `AI4Finance-Foundation/FinRobot` | 7,935 | 1,349 | 2026-09-07 | false | Apache-2.0 | 2024-02-27 |
| `The-Swarm-Corporation/AutoHedge` | 5,534 | 831 | 2026-05-11 | false | MIT | 2024-12-10 |
| `simonlin1212/TradingAgents-astock` | 3,229 | 834 | 2026-09-05 | false | Apache-2.0 | 2026-05-13 |
| `pipiku915/FinMem-LLM-StockTrading` | 956 | 196 | **2024-08-18** | false | MIT | 2023-12-01 |
| `MingyuJ666/Stockagent` | 699 | 167 | 2026-06-16 | false | **NONE** | 2024-02-02 |
| `TauricResearch/Trading-R1` | 480 | 31 | **2025-09-15** | false | **NONE** | 2025-09-15 |
| `RndmVariableQ/AlphaAgent` | 408 | 74 | 2026-07-03 | false | **NONE** | 2025-05-30 |
| `DVampire/FinAgent` | 75 | 27 | **2024-08-31** | false | MIT | 2024-04-05 |
| `The-FinAI/FinCon` | 68 | 6 | 2026-02-27 | false | **NONE** | 2024-10-28 |
| `OpenBB-finance/openbb-agents` → redirects to `experimental-openbb-platform-agent` | 1,344 | 171 | **2024-07-22** | false | **NONE** | 2023-11-25 |
| `AI4Finance-Foundation/FinNLP` | 1,482 | 276 | **2024-07-01** | false | MIT | 2023-02-07 |
| `OpenBB-finance/OpenBB` | 72,785 | 7,522 | 2026-07-30 | false | NOASSERTION (LICENSE file = AGPL-3.0 ✅) | 2020-12-20 |
| `microsoft/qlib` | 48,408 | 7,654 | 2026-09-02 | false | MIT | 2020-08-14 |

Also checked: `TheFinAI/FinBen` → 404 (the org is `The-FinAI`); `The-FinAI/Agent_Market_Arena`
7★, no licence; `felis33/INVESTOR-BENCH` 29★, MIT, last push 2025-09-13.

## 2. PyPI, exactly as fetched (`https://pypi.org/pypi/<name>/json`)

| Package | Version | Uploaded | requires_python | requires_dist | Note |
|---|---|---|---|---:|---|
| 🚨 `tradingagents` | 0.7.0 | 2026-05-21 | >=3.12 | 24 | **Repository = `Mai0313/tradingagents` (3★), NOT TauricResearch.** The real project has no PyPI release; its README says `git clone` + `pip install .` |
| ✅ `aihf` | 2.2.0 | 2026-08-07 | >=3.11,<4.0 | 15 | the real `virattt/ai-hedge-fund` (`pyproject.toml` name = `aihf`) |
| 🚨 `ai-hedge-fund` | 0.1.1 | 2026-04-10 | >=3.10 | 12 | a different author's package (`NamanVinayak/ai-hedge-fund`), "Claude Code slash commands" |
| ✅ `vibe-trading-ai` | 0.1.14 | 2026-08-20 | >=3.11 | 102 | `HKUDS/Vibe-Trading`; console scripts `vibe-trading`, `vibe-trading-mcp` |
| 🚨 `finrobot` | 0.1.5 | 2024-06-17 | >=3.10,<3.12 | **None** | `requires_dist: None` — `pip install finrobot` pulls zero dependencies, the same failure mode as `finrl` (see `../../rl-and-ml-trading/SKILL.md` §1) |
| 🔴 `fingpt` | 0.0.1 | 2023-10-20 | >=3.6 | 5 | stale placeholder; the repo is notebooks + modules |
| ⚠️ `rdagent` | 0.8.0 | 2025-11-03 | >=3.10 | 79 | latest GitHub release is also v0.8.0 (2025-11-03); `0.8.1.dev24` uploaded 2026-03-23; the repo itself is active (pushed 2026-09-04). README badge: Linux only |
| 🔴 `finmem` | — | — | — | — | 404 on PyPI |
| 🔴 `qlib` | 0.0.2.dev20 | 2018-11-20 | — | 0 | a squat; the real package is `pyqlib` 0.9.7 (2025-08-15) |
| `openbb` | 4.7.2 | 2026-05-26 | >=3.10,<4 | 50 | `license_expression: AGPL-3.0-only` |

## 3. Per-system notes (source-level)

### TradingAgents — `TauricResearch/TradingAgents` (v0.4.0, 2026-08-31; arXiv 2412.20138)

✅ Read from `tradingagents/graph/setup.py`, `conditional_logic.py`, `trading_graph.py`,
`agents/utils/memory.py`, `graph/reflection.py`, `graph/signal_processing.py`,
`default_config.py`, `pyproject.toml`, `CHANGELOG.md`, `README.md`.

- **Orchestration:** LangGraph `StateGraph(AgentState)`; `langgraph>=0.4.8`,
  `langgraph-checkpoint-sqlite` for opt-in checkpoint resume (`--checkpoint`).
- **Roles, in graph order:** analysts (`market`, `social`→"Sentiment Analyst", `news`,
  `fundamentals`), each with a `ToolNode` loop → `Bull Researcher` ↔ `Bear Researcher` for
  `2 * max_debate_rounds` turns → `Research Manager` → `Trader` → `Aggressive` / `Conservative` /
  `Neutral` risk debators for `3 * max_risk_discuss_rounds` turns → `Portfolio Manager`.
  Two model tiers: `quick_think_llm` for analysts/researchers/trader/risk, `deep_think_llm` for
  the two managers. Defaults in v0.4.0: `gpt-5.6` deep, `gpt-5.6-luna` quick.
- **Output:** the Portfolio Manager emits a structured `PortfolioDecision`; `SignalProcessor`
  parses a 5-tier rating (Buy / Overweight / Hold / Underweight / Sell) and returns `REVIEW`
  when it cannot (#1170 — previously coerced to a tradeable Hold).
- **Memory:** `TradingMemoryLog` — an append-only **markdown** decision log
  (`~/.tradingagents/memory/trading_memory.md`). Phase A stores the decision as `pending`;
  Phase B, on the next run, fetches the realised return and alpha vs SPY, asks the quick LLM
  for a 2–4 sentence reflection, and injects recent same-ticker and cross-ticker lessons into
  the PM prompt. `get_past_context(as_of=...)` filters lessons by resolution date (#1251).
- **Data:** Alpha Vantage, yfinance, FRED, Polymarket, Reddit, StockTwits (`dataflows/`).
- **What it does not have (✅ code search on 2026-09-08):** `slippage` 0 hits, `commission`
  0 hits, `Sharpe` 0 hits, `place_order` 0 hits, `class .*Exchange` 0 hits. "backtest" hits
  are date-window and look-ahead fixes plus their tests — there is no P&L simulator. The
  README's "sent to the simulated exchange and executed" has no module behind it.
  `backtrader>=1.9.78.123` is declared in `pyproject.toml` and the only hit for `backtrader`
  in the repo is `pyproject.toml` itself — an unused dependency on a library whose last commit
  was 2023-04-19 (`../../../../fin-core/skills/backtesting-engines/SKILL.md`).
- **Reproducibility, in its own words (README):** "Even at a fixed temperature, providers do
  not guarantee byte-identical output across calls", reasoning models "largely ignore
  temperature", and "News, StockTwits, and Reddit return different content as time passes".
- **The v0.4.0 changelog is a catalogue of agent-specific leaks, all fixed by code, not
  prompts:** FRED macro served from today's vintage (#1275); StockTwits/Reddit fetched with no
  date (#1220); memory returned lessons resolved after the trade date (#1251); reflection ran on
  a partial holding window (#1169); the newest NaN bar was silently dropped (#1201); the first
  debater rebutted an empty opponent, fabricating the other side (#1176); v0.3.1: the Alpha
  Vantage look-ahead filter never ran because the payload was a JSON string (#1115).
- **Cost per decision:** with the default 4 analysts and one round each, at least 12 LLM
  invocations per ticker-date before tool-call round trips (4 + 2 + 1 + 1 + 3 + 1); 17 with
  two rounds. Printed by `../scripts/agent_pipeline.py`.
- **Forks:** `hsliuping/TradingAgents-CN` (31,650★) — ✅ LICENSE file: Apache-2.0 by default,
  **proprietary for `app/` (FastAPI backend) and `frontend/`**.
  `simonlin1212/TradingAgents-astock` (3,229★, Apache-2.0). `KylinMountain/TradingAgents-AShare` (811★, NOASSERTION).
- **Trading-R1** (arXiv 2509.11420): repo is 480★, no licence, description "Terminal coming
  soon ...", pushed once on 2025-09-15. ⚠️ Vaporware until it lands.

### ai-hedge-fund — `virattt/ai-hedge-fund` (v2.2.0 / `aihf` 2.2.0, 2026-08-07)

✅ Read from `hedge_fund/README.md`, `VISION.md`, `ROADMAP.md`, `pyproject.toml`,
`hedge_fund/pipeline/run_cycle.py`, `risk/limits.py`, `brokers/sim.py`, `signals/llm_agent.py`,
`backtesting/engine.py`.

- **Shape:** `FUND = capital slices over STRATEGIES; STRATEGY = blend policy over MODELS;
  MODEL = alpha model → Signal (conviction in [-1, +1] + thesis)`. Discretionary pods are LLM
  investor personas (Buffett, Munger, Graham, Lynch, Druckenmiller — `signals/*.py`);
  systematic pods are quant models (`pead.py`). Both implement `AlphaModel.predict(ticker,
  date, data_client) -> Signal`.
- **Pipeline:** `run_cycle`: point-in-time data → analysts → blend → risk → execution →
  record. "A backtest is run_cycle in a loop over history with a SimBroker; paper trading is
  the same loop on a live clock with a PaperBroker; live is the same loop with a real broker."
  Stages other than `run_cycle` are pure functions; the record is byte-identical given the same
  inputs (a cold LLM cache is the stated exception; the prompt cache makes replays exact).
- **Where the LLM sits:** research time only. README principle: "The LLM never touches the
  trade. Agents form views and narrate; deterministic code sizes and places orders; risk limits
  are hard gates." `risk/limits.py`: per-ticker and gross-exposure clamps, "no clamp is ever
  negotiable", clamped exposure stays in cash. `llm_agent.py`: LLM/parse failures **abstain**
  (`Signal(0.0, abstained=True)`), data-layer errors propagate ("fail loud").
- **Orchestration:** no LangGraph in v2 — provider packages (`langchain-anthropic`,
  `langchain-openai`, `langchain-deepseek`, `langchain-google-genai`, `langchain-xai`) behind
  one `make_llm` factory; Textual TUI.
- **What it does not have:** `SimBroker` "fills every order completely, exactly at the
  order's reference price. Slippage/costs are a declared future addition"; "Margin is not
  modeled: cash may go negative". `validation/` (CPCV, PBO) is planned; point-in-time data
  correctness is marked in progress; paper/live brokers are planned (`brokers/protocol.py` +
  `sim.py` only);
  data is the paid Financial Datasets API only; no kill switch. Universe is caller-supplied
  (`--tickers`), so survivorship is the caller's problem (0 hits for `survivorship`).

### Vibe-Trading — `HKUDS/Vibe-Trading` (v0.1.14, 2026-08-20; created 2026-04-01)

✅ Read from `pyproject.toml`, `agent/SKILL.md`, `README.md` sections, `agent/src/` and
`agent/backtest/` listings, `tools/ci_grep_gates.sh`, `agent/backtest/validation.py`,
`run_card.py`, `regime.py`. ⚠️ Not installed or run here.

- **Shape:** a LangChain/LangGraph agent (`langgraph>=1.2.5,<1.3`) behind a FastAPI server
  and React front end, shipped three ways: CLI (`vibe-trading`), MCP server
  (`vibe-trading-mcp`, "74 MCP tools"), and an Agent Skill (`agent/SKILL.md` with `mcp:`
  frontmatter). `agent/src/` has `swarm/` (multi-agent "desk" presets: `investment_committee`
  bull/bear debate, `quant_strategy_desk`, `crypto_trading_desk`, `macro_rates_fx_desk`),
  `memory/` (hierarchy, lifecycle, compression, semantic links), `governance/` (`ledger.py`,
  `manifest.py`), `live/` (`order_guard.py`, `sdk_order_gate.py`, `halt.py`, `enforcement.py`,
  `daily_count.py`, `mandate/`), `trading/connectors/`, `shadow_account/`, `security/`.
- **Research workflow (README):** Plan → Ground → Execute → Validate → Deliver; the run halts
  with a recovery request after eight consecutive tool iterations without a new observation.
- **Backtest layer:** `agent/backtest/engines/` — `china_a`, `china_futures`, `crypto`,
  `forex`, `global_equity`, `global_futures`, `india_equity`, `korea_equity`, `vietnam_equity`,
  `options_portfolio`, `composite` (the "10 engines"); `validation.py` = Monte Carlo
  permutation, bootstrap Sharpe CI, walk-forward — **not** CPCV/DSR/SPA; `run_card.py` writes a
  JSON/Markdown run card with config and strategy hashes; `factor_costs.py`; `regime.py` is a
  correlation-regime timeline labelled "descriptive risk context — not a trading signal".
- **Where the LLM sits:** research and decision time, and — opt-in — execution: "autonomous
  trading through a broker you authorize yourself (e.g. Robinhood Agentic Trading)"; IBKR, MT5,
  Binance connectors. The `live/` gate modules exist; ⚠️ their behaviour is unverified here.
- **Gates as code:** `tools/ci_grep_gates.sh` enforces repo-wide floors in CI (no unsafe
  `yaml.load`, no per-stock data dumps in the wiki, no bare `datetime.now()`, no raw
  `os.environ` reads outside the config layer).
- **Caveats:** five months old; the README is 285 KB, mostly a dated changelog. Its own
  2026-09-08 entry reports that blanking one bar of inputs across the 462-alpha "Alpha Zoo" left
  **84 alphas returning a number anyway** (fixed at the registry) and **52 alphas still emitting
  inside their declared warm-up** (open). Treat the alpha library as unverified. Its own
  2026-09-08 fix (#1387): an announcement date carries no time of day, so on an intraday frame a
  filing was visible from the first bar of its own day — the same availability bug as
  `../../../../fin-core/skills/fundamental-and-macro-data/SKILL.md` §2.

### RD-Agent — `microsoft/RD-Agent` (R&D-Agent-Quant, NeurIPS 2025, arXiv 2505.15155)

✅ Read from `rdagent/app/qlib_rd_loop/quant.py`, `scenarios/qlib/proposal/quant_proposal.py`,
`scenarios/qlib/developer/feedback.py`, and
`scenarios/qlib/experiment/factor_template/conf_baseline.yaml`.

- **Loop (`QuantRDLoop`):** hypothesis generation (an LLM, or a bandit `EnvController` over the
  `factor` / `model` action) → `Hypothesis2Experiment` → `factor_coder` / `model_coder` (LLM
  writes the code) → `factor_runner` / `model_runner` (qlib backtest, Docker) →
  `Experiment2Feedback`
  (an LLM reads `IC`, `1day.excess_return_with_cost.annualized_return`,
  `1day.excess_return_with_cost.max_drawdown` for the candidate vs the SOTA and returns a JSON
  with `Replace Best Result: yes/no`) → trace.
- **Costs:** yes, by inheritance from qlib — `conf_baseline.yaml`: `TopkDropoutStrategy`
  (`topk: 50, n_drop: 5`), `open_cost: 0.0005`, `close_cost: 0.0015`, `min_cost: 5`,
  `limit_threshold: 0.095`, `deal_price: close`, CSI300 universe, benchmark SH000300, train
  2008–2014 / valid 2015–2016 / test 2017– with an LGBM baseline. 🚨 `deal_price: close` fills
  at the signal bar's close — the same-bar-fill issue of
  `../../../../fin-core/skills/backtesting-engines/SKILL.md` §2.1 — and the label is
  `Ref($close, -2)/Ref($close, -1) - 1`.
- **Where the LLM sits:** research time only. The judgement "keep this factor" is an LLM
  reading a backtest metric — every loop iteration is a trial and the ledger must count them
  (`../../llm-finance-agents/SKILL.md` §4).
- **No execution, no live path.** Also ships data-science, Kaggle, fine-tuning and RL scenarios;
  the quant scenario is one of six.

### FinRobot — `AI4Finance-Foundation/FinRobot` (arXiv 2405.14767)

✅ Read from `finrobot/agents/workflow.py`, `agent_library.py`, `finrobot_equity/README.md`,
`README.md`.

- **Legacy package (`finrobot/`):** AutoGen — `from autogen import ConversableAgent,
  AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, register_function`. An agent
  library of role profiles (`Market_Analyst`, `Financial_Analyst`, `Expert_Investor`, ...) plus
  toolkits and a RAG function. 🚨 The PyPI wheel declares no dependencies (§2).
- **New direction (README, "FinRobot Desktop v0.1.0"):** a macOS Apple-Silicon app built on
  **PydanticAI + FastAPI + React/Tauri**, "1 Lead Agent, 5 role-based sub-agents (data,
  analysis, modeling, synthesis, report), 3 debate agents (bull, bear, judge)". Its stated
  principle is the one design rule worth porting: **"Numbers are code-calculated. Narratives
  are LLM-assisted. Every output is provenance-tracked."** DCF/DDM/LBO/WACC/Monte Carlo are
  "pure-Python compute operators". ⚠️ The desktop source is not in this repo tree; the claim
  is from the README.
- **`finrobot_equity/`:** an equity-research report generator (FMP data → per-section LLM
  agents → HTML/PDF). Research-time LLM; no execution; a report pipeline, not a trading one.
- Also ships a commercial "FinRobot Pro" and a `TRADEMARK_POLICY.md`.

### FinGPT — `AI4Finance-Foundation/FinGPT`

✅ Tree: `fingpt/FinGPT_Forecaster`, `FinGPT_RAG`, `FinGPT_Sentiment_Analysis_v1/v3`,
`FinGPT_FinancialReportAnalysis`, `FinGPT_MultiAgentsRAG`, `FinGPT_Benchmark`,
`FinGPT_ForexIntelligence`, plus LoRA fine-tuning notebooks (ChatGLM2-6B, Llama-3-8B) and an
`ag2_financial_analysis_pipeline.ipynb`. **A model layer, not an agent architecture** —
"FinRobot vs FinGPT" is a category error: FinRobot is the agent platform, FinGPT is the
fine-tuned-model project of the same foundation.

### FinMem — `pipiku915/FinMem-LLM-StockTrading` (arXiv 2311.13743) 🔴 last push 2024-08-18

✅ Read from `puppy/memorydb.py`, `puppy/agent.py`, `puppy/reflection.py`, `puppy/environment.py`,
`puppy/memory_functions/`.

- **The canonical layered memory:** `BrainDB` = `short_term_memory`, `mid_term_memory`,
  `long_term_memory`, each a `MemoryDB` over a FAISS inner-product index of OpenAI embeddings
  (`text-embedding-ada-002` is hard-wired). Each memory carries an importance score, a recency
  score, an exponential decay, an access counter that feeds back on importance, and a compound
  score; `prepare_jump` / `accept_jump` promote or demote memories across layers when the
  importance crosses `jump_threshold_upper` / `jump_threshold_lower`; `_clean_up` evicts below
  thresholds. A separate reflection layer stores lessons.
- **Profiling ("character design"):** a `character_string` in the system prompt, a
  `look_back_window_size` (the "cognitive span").
- **Decision:** a Pydantic/`guardrails` structured output (`ValidChoices` over buy/sell/hold and
  the memory ids it cited), so the action is typed — the one part every later system copied.
- **Environment:** `MarketEnvironment.step()` yields `(date, price, filing_k, filing_q, news,
  record, terminated)` for one symbol; single-asset, no costs, no portfolio.
- Not on PyPI; frozen for two years. A design reference, not something to install.

### The rest

- **FinCon** (NeurIPS 2024): `The-FinAI/FinCon` is a 3 KB README-only repo, no licence, still
  promising code "within the next 3–4 months" (README dated 2026-02-27, repo created 2024-10-28).
  The README points instead to InvestorBench (ACL 2025, `felis33/INVESTOR-BENCH`, MIT, 29★) and
  Agent Market Arena (WWW 2026, arXiv 2510.11695, `The-FinAI/Agent_Market_Arena`, 7★, no
  licence), a live testbed that wraps agents as HTTP endpoints.
- **FinAgent** (arXiv 2402.18485): `DVampire/FinAgent` — the owner's display name is Wentao
  Zhang, the paper's first author, so ⚠️ this is plausibly the authors' code; MIT, 75★, last push
  2024-08-31, README is an install recipe (FMP/Polygon/Alpha Vantage keys, Playwright). Research
  code, unmaintained.
- **StockAgent** (`MingyuJ666/Stockagent`, 699★, no licence) is a market *simulation* of LLM
  traders, not a pipeline. **AlphaAgent** (`RndmVariableQ/AlphaAgent`, 408★, no licence) is an
  alpha-mining loop. **AutoHedge** (5,534★, MIT, pushed 2026-05-11; `swarms = "*"` in its
  pyproject) has Director / Quant / Risk / Execution agents and advertises "full autonomous
  trading on Solana" — the LLM sits at execution time and the README describes no gate.
- **`OpenBB-finance/openbb-agents`** now redirects to `experimental-openbb-platform-agent`
  (1,344★, no licence, last push 2024-07-22). OpenBB's live agent surface is the MCP server
  extension inside the AGPL-3.0 platform — see `../../finance-mcp-servers/SKILL.md`.
- **`microsoft/qlib`** is the backtest and data engine RD-Agent's quant scenario drives; it is
  not an agent framework (`../../../../fin-libraries/skills/lib-qlib/SKILL.md`).
