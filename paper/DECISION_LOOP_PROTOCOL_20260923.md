# Prospective tools / experience / HiSTrim financial loop

User authorization (2026-09-23): implement and run all three groups, including writing
HiSTrim ourselves. This supersedes the assumption that an external HiSTrim checkpoint
must be supplied. No existing recommendation-table score will be transferred to finance.

## Scope and fixed design

This first execution is an authored synthetic development study, not an external holdout.
Four financial task mechanisms (volatility, tail risk, capital allocation, forecasting)
each have two client contracts. Each contract has two acquisition episodes followed by
two disjoint, later synthetic time blocks. The input generator is fixed at seed 101;
inference seed 11 is paired across conditions. Eight contracts are not eight independent
datasets, and repeated seeds will not be counted as additional task mechanisms.

Every current contract is supplied in every condition, including no-memory. Thus memory
does not receive a secret rule unavailable to the baseline. After an episode, all arms
receive the same reference feedback and one reflection response opportunity. Only storage
and later readout differ. The LLM receives no future feedback. No investment return,
market outperformance, human time saving or external generalization is measured here.

Each episode permits one selected numerical tool call, one final numerical response,
and one post-outcome reflection: three model responses capped at 384 output tokens each.
Invalid JSON, missing calls, numerical errors and missing receipts remain failures in
the planned denominator. There are no extra response retries. Record actual input/output
tokens, calls, tool time, wall time, peak allocated/reserved GPU bytes and physical KV bytes.

The same FinSkills numerical implementations execute behind every API interface.
Independent NumPy/Python formula references must agree before inference. Report method
choice, parameter correctness, numerical correctness, receipt-linked output use and their
conjunction. The final answer's explanation is a trace, not proof of a causal reasoning
process; choice changes and memory items actually supplied are logged separately.

## Eleven conditions, two fixed model families

| Group | Conditions | Planned episodes per model |
|---|---|---:|
| Tools | Generic function interface; same interface + domain guidance; FinSkills named APIs with the same guidance organized by tool | 96 |
| Memory, FinSkills + full context | Frozen acquisition memory; ordinary text reflection; structured episodic retrieval; causal fly readout + the same episodic records | 128 |
| Context, FinSkills | Recency and HiSTrim, each crossed with no memory and fly memory | 128 |

The full-context/no-memory cell is shared from Tools. Full-context/fly is shared from
Memory, completing the 3 context policies x 2 long-term-memory conditions. Total: 352
episodes per model, 704 across Qwen2.5-Coder-14B and Mistral-Nemo-12B. Acquisition and later
evaluation are reported separately (352 later-evaluation episodes across both models).
Model revisions are inherited unchanged from the matched-repair campaigns. Different
tokenizers/model sizes are not equal actual compute. Condition order is seeded; each arm
starts with an independent memory state. A single inference seed is an execution pilot,
not an adequate uncertainty estimate for a top-conference generalization claim.

## What the memory intervention implements

The actual causal circuit in `benchmarks/fly_paper/online.py` uses the upstream central
fitted parameter vector and its existing native-equation reproduction receipt. It is
coupled here to LLM-facing experiences and two candidate-method cue scores per client.
Correct task completion drives +1 and incorrect completion drives -1 reinforcement on
the chosen valid method cue. This is a new correctness-to-circuit mapping, explicitly
not a realized financial return and not a fabricated NAV receipt. It must be evaluated,
not presumed beneficial. Frozen memory acquires the first two episodes and then stops
both record and circuit updates. Retrieval and fly see the same structured experience
records; fly adds actual circuit readout. Text reflection retains the LLM's lesson alone.
All read items, IDs, availability times and circuit-state hashes are recorded. Reading
does not update state; duplicate experience IDs are rejected.

## What this HiSTrim implementation implements

`benchmarks/agent_study/histrim.py` is new paper-equation code, not an original released
checkpoint. It implements sigmoid-times-utility scores, normalized ridge initialization,
forward paired tangent-direction refinement, separately fitted thresholds, protected
tokens, nested whole-history-item selection at one-third and two-thirds decoder depth,
physical shorter hidden/KV buffers, original RoPE positions, and sink compensation
with fixed beta=0.1. Calibration uses four manual-relevance prompts with no task output
labels; its objective is item relevance squared error. Refinement has 24 paired steps.
All settings are frozen before task evaluation, with no outcome-based tuning.

Historical token caps are 50% then 25% of initial history tokens. Recency uses the same
caps and whole items. Thresholding and whole-item packing can underfill the budget;
actual retained tokens are reported rather than called identical token counts. System,
current contract, current API result and answer schema are protected. Memory records are
history items and may be removed, permitting the joint experiment to reveal interaction.
Record retained positions and item IDs so useful-evidence retention can be audited.

Batch size is one, with packed offsets `[0, length]`; this implementation does not claim
an optimized multi-request VarLen kernel. All three context arms use the same manual
decoder and backend. Full-context final logits must match the official forward pass
(same top token, max absolute error <0.15 in BF16); protected positions, nested subsets
and actual smaller KV storage must pass before task runs. Calibration overhead is
reported separately; inference measurements include routing, gathers and decoding.

## Execution and remaining evidence

First submit one GPU qualification job running the two models sequentially. It runs numerical/unit checks,
decoder equivalence, physical cache checks, and two closed-loop smoke episodes. A model's
wrong answer is retained and does not block expansion; a proven execution defect does.
Fix execution defects only in a new frozen campaign. Once qualified, submit six jobs
(three groups x two models) using dynamic Slurm placement, no node or exclusion lists.
All execution, outputs, caches and logs are under RADFM. Never resubmit an uncertain job.

This closes an executable development loop. Independently authored tasks, actual market
time-period transfer, more task mechanisms, multiple inference seeds, and independent
evidence-support labels remain necessary before broad paper claims. Do not inflate the
pilot by presenting its correlated episodes as independent research tasks.
