# Does access to fin-skills improve a trading agent's outcomes?

Research review and exploratory experiment protocol, 2026-09-21. This is a second
study strand alongside verified research-artifact execution in `STUDY_PLAN.md`.
No model was trained for this experiment. It tests an existing model with and without
library augmentation, which is the user's intended contribution.

This v1 intervention supplies predetermined tool outputs. The user's subsequent
clarification calls for model-selected tool use; see the separate
[autonomous-agent development study](AUTONOMOUS_LIBRARY_STUDY.md). V1 is retained as
an information-augmentation control and does not answer the autonomous-selection question.

## What the literature changes about our experiment

The defensible question is whether a reusable library improves a fixed agent under
matched market information, action constraints and inference limits. High return in
one selected backtest is insufficient evidence of either library value or method novelty.
The following primary sources were opened by the coding agent. They remain pending
accountable-human verification; this is not a systematic review or an exhaustive search.

| Source | Evidence locator and finding | Consequence for our study |
|---|---|---|
| [FinAgent, arXiv:2402.18485v1](https://arxiv.org/html/2402.18485v1) | Sections 7.1–7.2 and Table 5 compare tool augmentation and tools alone; the reported direction differs between equity and crypto cases. | Include the library's deterministic policy without an LLM. Do not assume adding tools always improves returns or claim tool augmentation as new. |
| [StockBench, arXiv:2510.02209v1](https://arxiv.org/html/2510.02209v1) | Sections 2–3 describe sequential decisions, common inputs and equal-weight buy-and-hold. The authors report that most tested agents did not beat the simple baseline. | Use chronological messages and buy-and-hold; preserve failed decisions. Its dataset and model-cutoff claims require independent checking before adopting them. |
| [InvestorBench, ACL 2025](https://aclanthology.org/2025.acl-long.126/) | Abstract and publication record describe an agent benchmark spanning stocks, crypto and ETFs, with multiple model backbones. | A reusable library needs testing beyond one backbone and one market. This pilot cannot support that breadth. |
| [Profit Mirage, arXiv:2510.07920v1](https://arxiv.org/html/2510.07920v1) | Sections 2–3 examine leakage and counterfactual evaluation, then describe FactFin. | Withhold future observations from prompts. Record model revision, dates and public-data exposure. Masking identifiers is only mitigation, not proof that pretraining contamination is absent. |
| [When Agents Trade / AMA, arXiv:2510.11695v2](https://arxiv.org/abs/2510.11695v2) | Abstract describes ongoing multi-market agent evaluation and differing framework behavior. | A later prospective paper-trading test would address weaknesses of retrospective examples. This run does not submit real trades or establish live performance. |
| [FinRobot, arXiv:2405.14767](https://arxiv.org/abs/2405.14767) | Abstract describes open financial agent toolchains and platform layers. | An open financial tool library alone is not an unoccupied research category; emphasize what our measured intervention adds. |
| [Bailey and Lopez de Prado, Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) | Abstract, Selection Bias and Deflated Sharpe Ratio sections discuss selection over trials and non-normal returns. | Freeze the protocol before scoring, keep all attempts and report uncertainty. Do not search the same evaluation period until a positive result appears. |
| [ECB reference-rate documentation](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html) | Introductory reference-rate publication and usage statement. | Reference observations are not executable quotes. This pilot measures a price-only proxy, excluding interest/carry; it cannot substantiate investment returns. |

### A useful contribution if the evidence supports it

The proposed empirical contribution is **downstream economic utility of verified domain
library access**, measured by a paired intervention on a fixed financial agent. It should
report both gains and losses, costs, drawdowns, turnover, exposure and task completion.
It complements the existing accepted-output correctness experiment. Neither strand's
outcome should be substituted for the other's: correct research can find no profitable
strategy, and profitable-looking output can have invalid accounting.

## Frozen feasibility experiment

Implementation: `benchmarks/library_utility/run.py`; independent accounting:
`benchmarks/library_utility/evaluate.py`. The accounting module does not import fin-skills.

- Existing Qwen2.5-Coder 7B and 14B snapshots already stored under RADFM; exact revisions
  copied from the earlier preparation manifest. No new weights or external datasets.
- Three conditions: `no_library`, `skills_text`, `full_library`. All receive the same
  anonymous historical price arrays and basic return, volatility and correlation summaries.
- The text condition additionally receives deterministic, bounded excerpts of
  portfolio-and-risk, backtest-validation and trend-following-models. Excerpts and hashes
  are stored. The full condition also receives actual `run` and `recommend_strategy`
  outputs, with execution receipts. This is orchestration-supplied tool augmentation;
  it does not test whether an autonomous agent discovers the correct tools.
- Only past values enter these APIs. The model has no filesystem, network or arbitrary
  Python execution tool. Common output validation applies to every condition.
- Existing ECB fixture, transformed from foreign currency per EUR to foreign-currency
  value in EUR via reciprocal. Four anonymous series; zero-interest EUR cash. Dataset
  identity and calendar dates are omitted from model messages, retained in receipts.
- First 84 reference observations of 2025; rebalance every 21 observations. Each decision
  sees 126 historical observations. These are declared feasibility choices, not parameters
  selected for observed returns. This is a public retrospective convenience sample,
  already present in the repository, not a new private holdout.
- Two inference seeds, 11 and 23; paired seed per decision and randomized condition order.
  One provider call per decision, 256 generated-token cap, temperature 0.1. Report actual
  input/output tokens and time; matching limits does not match actual consumption.
- No leverage or shorting. Decision after reference observation t is modeled as executing
  at reference t+1, so it cannot earn the t-to-t+1 movement. This conservative time lag
  still does not turn a reference observation into an executable fill.
- Fee sensitivity at 0, 5 and 20 basis points, including entry and final liquidation.
  Model prompts state 5 bps; the other schedules rescore the frozen actions without
  rerunning the model. Target weights use after-fee wealth through a self-financing solve.
- Invalid JSON or infeasible allocations cause no rebalance, remain in the record and
  are counted. Provider failures stop the run instead of silently manufacturing cash
  decisions. Partial runs cannot be reported as completed-period model outcomes.
- Cash, equal-weight buy-and-hold and the library's rule policy are separate controls.
  Model responses/actions are saved before scoring. No test-return feedback reaches
  the model or triggers policy tuning.

Primary exploratory comparison: full-library minus no-library cumulative proxy return
at 5 bps, shown separately for every model and seed. Also show risk, exposure and costs.
Paired date-block intervals use common dates and retain cross-asset dependence. The
short pilot does not have a confirmatory significance claim; seeds are not new markets.

## Conditions for a paper-level trading claim

1. Obtain a documented, permissible dataset with executable-price conventions, corporate
   actions, historical membership, trading constraints and point-in-time news if news is used.
   The current ECB fixture cannot replace it. Verify any institutional data-use requirements
   before transferring new datasets to Beacon.
2. Freeze the public development and untouched evaluation partitions and a practical effect
   size before inspecting confirmatory outcomes. Account for overlapping labels and serial
   dependence, all model/strategy searches and finite market coverage.
3. Expand to independent market periods and markets, with more than one backbone and an
   adequately justified sample size. Do not count repeated inference on one tape as many
   independent financial observations.
4. Add risk-matched controls, generic alternative-tool access and a relevant published agent
   baseline. This pilot isolates useful library context, but not against every competing
   library or toolchain and not at identical consumed compute.
5. Compare receipt-backed library use, knowledge-only guidance and suitable rule-only
   policies; audit action timing and accounting independently. Preserve negative results.
6. Freeze a prospective paper-trading protocol if moving to live data. Do not equate that
   with permission to submit real-money orders.

## Execution record

Remote root: `/beacon-projects/radfm/wy891/fin-skills-utility-20260921-v1`.
Initial source archive SHA256:
`b0519d22f764d76c646af7bd562229e4e104e705dc81eefb40cedb5c8bb56a04`.
Slurm job `1614147` completed on an L40S GPU with exit code `0:0` in `00:04:11`.
All 12 trials completed and all 48 decisions passed allocation validation. This establishes
execution feasibility, not profitable performance. See [the complete pilot results](LIBRARY_UTILITY_RESULTS.md).
The earlier audit study and pending 32B job `1613767` were left intact.
