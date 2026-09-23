# Autonomous library selection: development experiment v2

## Research question and correction to v1

The user intends to evaluate an agent that decides which library tools to use, inspects
their results, and then chooses its strategy and allocation. The previous experiment
supplied predetermined tool outputs. It tested context augmentation, not autonomous
tool choice. Its mixed results cannot establish whether autonomous use is effective.

This experiment adds a real bounded action loop. It is not model training. Source files:
`benchmarks/library_utility/autonomous.py` and `agent_tools.py`. The v1 runner and its
completed remote artifacts remain unchanged.

## Frozen design

- Four conditions: no library, preselected skill excerpts, preselected excerpts plus
  fixed tool outputs, and autonomous library access. The autonomous condition starts
  with tool descriptions and no pre-executed tool results or forced skill excerpts.
- The model may discover algorithms, read a related skill in pages, execute an algorithm
  with chosen parameters, or ask the optional heuristic strategy adviser about an asset.
  It chooses call order, parameters and whether to follow results. It may call no tool.
- An actual dispatcher executes only requested calls and returns results before the
  model's next action. A model-written claim that it used a tool does not count as a call.
- Tools are restricted to the subset compatible with the available historical prices.
  Optional arbitrary Python, filesystem access, future-data queries and live news are
  not offered. Historical news is a separate data requirement, not silently fabricated.
- All four conditions use the same system protocol and maximum of five model responses
  per decision, each capped at 256 generated tokens. Each response may finalize, reflect,
  or request an available tool. Only the autonomous condition has callable tools.
  Finalization ends the decision early; no minimum tool or reflection count is imposed.
- The last response must finalize. Tool/format errors consume turns and are returned for
  possible model correction. Exhausting the budget without a valid allocation leaves
  existing holdings unchanged and is counted as a failure. Provider errors abort the run.
- Tool requests/results, provider responses, tokens, errors, final allocations and hashes
  are saved. Raw responses are written before dispatch; allocations are frozen before
  scoring. A JSON action protocol is used, not provider-native function-call syntax.
- Existing Qwen2.5-Coder 7B and 14B snapshots, seeds 11 and 23. The four conditions are
  rerun under this common budget and randomized order. V1's one-call controls are not
  presented as budget-matched v2 controls.
- Same data window, history length, rebalance spacing, execution lag, costs and independent
  ledger as v1: 84 ECB reference observations ending 2025-05-02, 126 historical observations
  per decision, four decisions per trial, costs 0/5/20 bps. Primary descriptive comparisons
  are autonomous minus no-library and autonomous minus fixed-tools at 5 bps.
- Cash, equal-weight buy-and-hold and deterministic library rules remain separate controls.
  All trials and failures are retained; no outcome-based reruns or window selection.

## Claim limits fixed before this run

V1 outcomes on this same public market path have already been inspected. V2 is a development
experiment prompted by a mismatch between the intended agent and the v1 intervention. It is
not an untouched confirmatory test. No gain here can be described as independent validation.

The model chooses among offered tools; the protocol does not claim unrestricted discovery
of the entire library. The strategy adviser itself contains deterministic routing, but it
is optional and distinct from the LLM choosing which API or algorithm to invoke.

Identical maximum budgets do not imply identical actual token consumption or latency. Those
must be reported alongside call success, tool-selection diversity and valid decisions. A
successful invocation does not prove that the chosen tool was appropriate or caused a gain.

ECB reference prices remain non-executable proxies without interest/carry. There is one
short market path, one model family, no historical news archive and unresolved pretraining
exposure. Neither this dataset nor this experiment supports a high-return investment claim.

## Execution record

All project execution is on Beacon under RADFM:
`/beacon-projects/radfm/wy891/fin-skills-autonomous-20260921-v2`.

Source archive SHA256:
`fc5aeb27b49f65e2436f1f837563acd3f548c58767f2365095a1ee405dee3698`.
Submitted Slurm job: `1614319`, L40S, medium QOS, one-hour wall limit. Submission alone
is not completion; observed results and test outcomes will be recorded below.

### Interface correction before interpreting financial outcomes

Job `1614319` passed 25 instrument tests, then exposed a parser inconsistency: the models
returned JSON inside a single Markdown fence, accepted by v1 but rejected by the v2 loop
before the common final-action parser ran. Tool calls, reflection and final allocations
were all affected. The run was cancelled and retained as an instrument-failure attempt.
No financial outcomes were used to choose the correction.

The corrected parser accepts one complete JSON fence for all action types, while still
rejecting surrounding prose and unsupported fences. An integration regression test checks
reflection, actual tool dispatch and final allocation in a fenced conversation. Prompt,
models, window, seeds, budgets and tool definitions are unchanged. The corrected source is
synced into a separate `fin-skills-autonomous-20260921-v2b` RADFM directory; the failed
attempt is not overwritten or counted as a completed trading experiment.

Corrected run: Slurm `1614326`; source archive SHA256
`2a47c7697adb1326a456b7b4f9c812091b35390f0e03fd14005f99e7658422b0`.
The expanded instrument suite passed: `26 passed in 5.83s` on Beacon.

## Completed corrected run

Job `1614326` completed with exit code `0:0` in `00:09:16`. All 16 trials completed;
all 64 final decisions were valid. Results were inspected without changing the protocol.

Protocol digest: `60b3ca78c4c07b5c30f2cef0d4dd3c6a6ffa9654a97df690e8cce1d98ef249ce`.
Completed `results/summary.json` SHA256:
`1c2319779a18d78e5da2b549b99bc2544a524054ac02afdcd71ae2fc75eefa6f`.
Raw artifacts remain under the corrected RADFM root; this is a synthesized report.

### Did autonomous selection actually happen?

Yes. The autonomous conditions made 43 successful requests: 13 for 7B and 30 for 14B,
with no tool or format errors in those conditions. Selection was optional. The model
chose algorithms and parameters, received their actual outputs, and then chose its next
action. The 7B seed-23 first decision finalized without a tool call.

| Successful request | 7B | 14B |
|---|---:|---:|
| Discover algorithm catalog | 2 | 8 |
| Read trend-following skill | 0 | 7 |
| Request heuristic strategy advice | 5 | 0 |
| Execute cross-sectional momentum | 2 | 1 |
| Execute Bollinger reversion | 3 | 14 |
| Execute volatility-targeted momentum | 1 | 0 |

These are invocation counts, not independent examples or evidence of optimal selection.
There were 18 advertised runnable algorithms. The model was not required to try all of them.
Example 7B seed-23 trajectories:

- Decision 2: discover catalog, select `cross_sectional_momentum` with lookback 60 and
  top-k 3, receive weights, then finalize.
- Decision 3: request strategy advice for asset_0, select `vol_target_momentum` for that
  asset, receive the lagged signal output, then finalize.

Indices above are zero-based, matching saved decision directories. Reflection steps are
omitted from this summary; no private reasoning transcript is reproduced.

### Did autonomous selection improve results?

Not on average against either the no-library or fixed-tool condition in this development
run. The table shows means over two seeds at 5 bps, on the same 84-observation path.
These are cumulative reference-price proxy returns, not annual returns or live P&L.

| Model | No library | Skill text | Fixed tool outputs | Autonomous library |
|---|---:|---:|---:|---:|
| 7B | +0.099720% | -2.930339% | -3.074957% | -3.984067% |
| 14B | -3.324386% | -4.566714% | -3.030291% | -3.783412% |

Autonomous minus no-library: -4.083787 percentage points for 7B and -0.459026 points
for 14B. Autonomous minus fixed-tools: -0.909110 and -0.753121 points, respectively.
Autonomous 14B did outperform its text-only condition by 0.783302 points. This exception
does not change the direction of either predeclared primary comparison.

All individual seeds are retained:

| Model | Seed | No library | Skill text | Fixed tools | Autonomous |
|---|---:|---:|---:|---:|---:|
| 7B | 11 | 0.000000% | -2.767109% | -3.498518% | -5.931914% |
| 7B | 23 | +0.199440% | -3.093570% | -2.651396% | -2.036219% |
| 14B | 11 | -2.398143% | -4.845107% | -2.820003% | -4.108767% |
| 14B | 23 | -4.250628% | -4.288321% | -3.240580% | -3.458058% |

### Risk and compute differences

Maximum drawdowns and exposures are averaged over seeds. Token counts and model calls
are totals for both seeds (eight decisions per row), not per-decision means.

| Model / condition | Mean max drawdown | Mean risky exposure | Model calls | Prompt tokens | Output tokens |
|---|---:|---:|---:|---:|---:|
| 7B / no library | -1.529924% | 0.500000 | 8 | 46,546 | 431 |
| 7B / skill text | -3.753745% | 1.000000 | 28 | 218,357 | 2,084 |
| 7B / fixed tools | -3.285201% | 0.964270 | 8 | 69,459 | 575 |
| 7B / autonomous | -5.027982% | 0.968680 | 38 | 249,893 | 2,363 |
| 14B / no library | -3.701537% | 1.000000 | 12 | 70,353 | 943 |
| 14B / skill text | -4.803647% | 1.000000 | 16 | 123,005 | 1,377 |
| 14B / fixed tools | -3.788480% | 0.999975 | 8 | 69,438 | 484 |
| 14B / autonomous | -4.538332% | 0.500000 | 38 | 308,731 | 1,908 |

Exposure differs materially. These are whole-policy comparisons, not risk-matched alpha
estimates. The entire corrected run consumed 156 model calls, 1,155,782 prompt tokens and
10,165 output tokens. One unavailable-tool request in the 7B text-only condition was denied;
that condition also had three format errors. The model recovered within its existing
budget. These events are retained, so valid final decisions do not imply error-free traces.

Cost sensitivity, rescoring the same frozen allocations:

| Model / condition | 0 bps | 5 bps | 20 bps |
|---|---:|---:|---:|
| 7B / no library | +0.149845% | +0.099720% | -0.050354% |
| 7B / skill text | -2.820412% | -2.930339% | -3.259374% |
| 7B / fixed tools | -2.779410% | -3.074957% | -3.956152% |
| 7B / autonomous | -3.887253% | -3.984067% | -4.273923% |
| 14B / no library | -3.193487% | -3.324386% | -3.716116% |
| 14B / skill text | -4.416084% | -4.566714% | -5.017348% |
| 14B / fixed tools | -2.926005% | -3.030291% | -3.342472% |
| 14B / autonomous | -3.591064% | -3.783412% | -4.357584% |

The saved paired date-block intervals are descriptive mean-daily-return comparisons.
For autonomous versus no-library, the 7B seed-11 interval lies below zero; the other
three model/seed intervals span zero. This is not confirmatory significance evidence:
the path was previously inspected, the market sample is short, and inference seeds
are not independent market samples. No favorable seed is selected as the headline.

The corrected experiment now measures the intended autonomous behavior. It does not yet
show that these models choose appropriate tools well enough to improve economic outcomes.
Subsequent selector or prompt development must be separated from an untouched evaluation
set; this negative result must not be erased by rerunning the same path until it improves.
