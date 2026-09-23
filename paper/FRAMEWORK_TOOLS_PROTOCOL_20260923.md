# Fixed LlamaIndex tool-organization comparison

Frozen before inference on 2026-09-23. This closes the specific missing framework/tool
interface comparison; it is not another seed sweep or an external-task benchmark.
Reuse the original 32 `decision_tasks.cases(101)` contracts (four mechanisms, eight
clients, two acquisition-designated and two later-designated numerical blocks each).
There is no memory or learning here; preserve phase labels only to align with prior work.

Actual LlamaIndex ReActAgent and FunctionTool use the already qualified HF chat adapter,
unchanged numerical dispatcher, reference and grade predicate. Three arms per model:

1. Generic: anonymous function names with explicit implemented method and parameter schema.
2. Guidance: the same generic tools plus a flat repeated copy of the domain rules.
3. Organized: descriptive method names with those same rules attached to each tool.

All arms receive the current contract/data and the entire identical method-rule manual.
Thus knowledge and numerical capabilities are shared. Guidance and organized differ in
placement/naming, not access to extra financial algorithms or labels. Generic differs in
redundancy as well as presentation. This is a narrow catalog-interface treatment, not a
whole-library advantage or the learned automatic router. No hidden calculator upgrade.

Two unchanged models: Qwen2.5-Coder-14B revision
`aedcc2d42b622764e023cf882b6652e646b95671` and Mistral-Nemo-Instruct-2407 revision
`04d8a90549d23fc6bd7f642064003592df51e9b3`. Each has 96 planned cells, total 192.
Use one GPU allocation/model; random arm-task order seed 20260923, task-paired inference
seed 11 plus the first eight hex digits of decision_tasks.digest(task ID), modulo
100,000,000. No extra seeds. All arms have at most six responses, 512 output tokens each,
eight framework iterations and the existing 32,768 context cap. No silent truncation.
Actual calls, tokens and wall time differ and are reported. These maxima differ from the
old hand-written loop, so do not pool its scores or claim a controlled framework winner.

Native ReAct parser retries are retained. Adjacent same-role messages are joined without
deleting text using the previously qualified adapter. Final JSON must contain exactly
values and receipt_sha256. No fence removal or parameter imputation. The numerical grade
uses the action corresponding to the cited successful receipt; uncited tool success is
not completion. Record all dispatch failures and first method/parameter choices as well.
Receipt freezing precedes scoring in a separate process. Code/data separation is not
an adversarial sandbox; tasks are author-constructed, previously exposed development cases.

Before production, CPU qualification checks all 32 numerical references, each arm's
actual framework/tool execution using scripted responses, wrong-receipt rejection and
strict final JSON. Scripted fixtures are engineering checks, not model quality results.
Source hashes must equal qualification hashes. OOM stops the job and retains unfinished
denominators; model format/selection errors remain outcomes. Only evidenced implementation
defects justify a new frozen repair directory. No prompt/budget/model tuning after scores.

After completion audit every source/input/receipt/score, framework message mapping,
actual tool receipt and budget. Report each model/arm/phase separately, all failures and
actual costs. End this comparison after the fixed 192 planned cells; no expansion to
FinRobot/FinMem trading episodes under the same label.
