# Remaining framework comparisons: bounded review

Reviewed official sources on 2026-09-23 after the submitted reuse batch finished.
Completed results remain immutable. This review distinguishes required inputs from
task mismatch and from work that is merely unimplemented.

## Native financial systems

| System | Verified native requirements | Fit to the current task and decision |
|---|---|---|
| FinRobot | Current README documents FMP and OpenAI configuration for equity reports; legacy tutorials also list Finnhub/SEC access. Official SingleAssistant accepts custom configuration/toolkits. | Native report generation is not the same endpoint as scalar FinQA or fixed method-selection contracts. API-free module adaptation is possible; missing a native key does not prove all code unusable. Do not label a replaced workflow as a native full-system reproduction. |
| FinMem | README supports local TGI generation but requires OpenAI embeddings. Actual `puppy/embedding.py` constructs OpenAIEmbeddings and its dimension resolver supports ada-002. | Native task is chronological trading, not cross-company financial QA. A local embedding replacement is a new adaptation needing fixed data, reward and execution/cost protocol. Not a drop-in baseline for the completed QA matrix. |
| InvestorBench | README specifies vLLM, Qdrant, OpenAI embeddings, warmup/test intervals and model configuration. | Local generation is supported; embeddings remain a distinct native dependency. Trading performance also needs a matched FinSkills trading policy and common risk/cost rules, which this QA protocol does not define. |

Sources: [FinRobot README](https://github.com/AI4Finance-Foundation/FinRobot),
[official workflow implementation](https://raw.githubusercontent.com/AI4Finance-Foundation/FinRobot/master/finrobot/agents/workflow.py),
[FinMem README](https://github.com/pipiku915/FinMem-LLM-StockTrading),
[FinMem embedding implementation](https://raw.githubusercontent.com/pipiku915/FinMem-LLM-StockTrading/main/puppy/embedding.py),
[InvestorBench README](https://github.com/felis33/INVESTOR-BENCH).
These are live reviewed sources, not frozen experimental software revisions. No claims
are made about credentials outside the existing authorized experiment setup; none were
searched for, printed or requested. In particular, Jev's known missing key must not be
used as evidence that other providers' credentials are absent.

The earlier broad statement that FinRobot V2 is unavailable is not carried forward as a
current blanket claim: the reviewed README now describes additional Desktop/Pro code and
deployment paths. Any reproduction must choose a specific accessible revision and task.
Full-system trading/report comparisons remain uncompleted. Their mismatch is not an
environment failure, and changing embeddings/backbones would need an explicitly labeled
adaptation rather than a claim to reproduce the paper's reported score.

## One directly executable missing control

Inspection of the completed `finqa_reuse.py` shows one FunctionTool (the official
calculator). Its memory experiment therefore cannot test whether FinSkills tool naming
and organization helps selection among methods inside an actual mature framework.
The original multi-method comparison used a hand-written loop. This is a concrete gap
that can be addressed with existing inputs and no new credentials.

Freeze [FRAMEWORK_TOOLS_PROTOCOL_20260923.md](FRAMEWORK_TOOLS_PROTOCOL_20260923.md):
actual LlamaIndex ReActAgent with three tool-organization conditions on the same original
32 authored contracts, two fixed models, 192 cells. Reuse all original algorithms and
rules; expose the same financial information. Same-task inputs avoid inventing a new
benchmark, but also prevent claiming independent external evidence. The old and new
framework scores must stay separate because interaction budgets and prompt formats differ.
No additional seeds or result-dependent tuning are authorized by this bounded control.

CPU qualification **1655240** was accepted at 07:41 UTC. It checks the actual framework
tool path and all numerical references before production. A read-only comparison found
all 29 copied algorithm/model/task source files identical to the previous decision-loop
snapshot. Production sources must match the qualification manifest. Acceptance is not
completion; use deployment/completion receipts for subsequent status. Subsequent check:
qualification completed and passed audit; production Qwen 1655292 and Mistral 1655293
were accepted at 07:43 UTC and are running, with matching source hashes and 96 cells each.

Final 08:21 UTC check: both jobs completed and all 192 cells passed audit. See
[FRAMEWORK_TOOLS_RESULTS_20260923.md](FRAMEWORK_TOOLS_RESULTS_20260923.md). This resolves
the one identified input-ready control; the broader system-task limitations above remain.

Jev, human efficiency and new independent support annotation remain separate gaps.
Public FinQA and FinanceBench annotations already exist and were used, so the plan must
not classify all external evidence as unavailable. No completed model output is repaired
or rerun because of this review.
