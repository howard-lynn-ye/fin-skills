# External FinQA memory transfer: prospective protocol

This protocol extends the verified LlamaIndex runtime to the task IDs frozen by
`fin-skills-campaign-memory-reuse-prepare-20260923-v1`. It is not a reproduction of
Reflexion's ALFWorld scores, nor a claim that the FinQA calculator is a FinSkills tool.
The original eight development failures remain separate and unchanged.

Each fixed model (Qwen and Mistral, previous revisions) runs one dynamic GPU allocation.
It first attempts the same 16 training questions from eight companies, without memory.
Each immutable episode is scored by a separate process against training labels only.
Only correctness and error categories return to the learner; gold numbers/programs are
not placed in its context. Failed attempts receive one reflection response, capped at
512 tokens, through the author's unchanged Reflexion `update_memory` function. Successful
attempts skip reflection, matching that function's behavior. All 16 records remain.

The original ALFWorld examples and prompt construction are preserved. A fixed completion
wrapper tells the model to produce reusable financial procedure advice without copying
report-specific answers. Original prompts, wrapper text, responses and hashes are logged.
Using these reflections on different companies is explicitly a cross-task adaptation of
Reflexion, whose original prompt refers to retrying the same task.

The resulting common acquisition bank is cloned conceptually across five evaluation arms:

| Arm | Visible experience | Circuit state |
|---|---|---|
| none | Empty | Absent |
| reflexion_recent | Most recent three acquisition records and their reflections | Absent |
| reflexion_retrieval | Top three positive lexical matches using actual FinSkills RAGIndex | Absent |
| fly_frozen | Exactly the retrieval arm's records | Unconditioned circuit readout |
| fly_learned | Exactly the retrieval arm's records | Circuit conditioned on acquisition feedback |

No evaluation label or reward is available until all evaluation outputs are frozen. Each
arm's bank and circuit remain unchanged throughout evaluation. Thus the frozen/learned
comparison isolates acquisition conditioning given identical text retrieval; it does not
test online adaptation during the test phase. Shared acquisition is executed once per model
and charged once, not reported as five independent training runs.

There are 32 fixed test questions from 16 companies disjoint from acquisition and dev
calibration, each from a different company/year report. Five arms yield 160 evaluation
episodes per model plus 16 acquisition episodes: 352 total episodes across the two models,
of which 320 are evaluation. The 32 evaluation questions, grouped by report/company, are
the task units; arms, models and seeds do not multiply independent task counts.

Every task has the verified six-response cap, 512 output tokens per response, eight native
framework iterations and 32,768 prompt-plus-output capacity. Over-capacity inputs fail
explicitly. No parser relaxation, retry-until-success, threshold tuning or new model is
introduced. Reflection has one extra bounded response per failed acquisition task; this
cost is reported separately. Five evaluation arms are interleaved in SHA-seeded order,
with the same per-question inference seed across arms and fresh ReAct state each time.

Memory is supplied as a structured user-input field, including IDs, source questions,
attempted programs, correctness/error categories and reflection text. All arms receive
the same instruction to treat memory as fallible procedure guidance and answer numerically
from the current report. Record exactly which experiences were read and their bank hash.
Different memory payloads have different actual input-token costs; caps are not equal FLOPs.

The existing causal fly equations and previously qualified parameters are reused. Two
engineering cues represent attempted program families: table operations versus scalar
arithmetic. The final program, or last returned tool program if no final program exists,
determines the cue using a frozen lexical rule. Unknown programs do not update a cue.
Execution correctness maps to +1 and all incorrect/ungradable attempts to -1, with the
existing 30-unit conditioning and 135-unit rest schedule. This is a new task-reward mapping,
not a return/NAV signal or biological-to-market time conversion. Scores are uncalibrated
method-family readouts, not probabilities. A reference program is not the only valid method;
family-choice counts are descriptive rather than an oracle of method-selection correctness.

Before any model inference, qualify actual Reflexion update behavior, memory visibility
through the real ReAct interface, disjoint task IDs, nonmutating reads and frozen versus
conditioned circuit state on explicit software fixtures. Production uses the same source.
Keep all planned failures and missing outputs. Score immutable evaluation receipts using
the unchanged official FinQA numerical and symbolic functions; report format errors,
scorer errors, correctness, program families, response/token/tool-receipt counts, wall time
and peak allocated/reserved GPU bytes. Reserved memory reflects allocator history.

This is a bounded public-benchmark study with one inference seed and two fixed models.
Public pretraining exposure, recurring question templates, ALFWorld prompt-domain mismatch,
the engineering cue/reward mapping and previously poor development performance remain
limitations. Do not select the best arm/seed, rewrite labels, or claim overall library
superiority, independent human efficiency, or the entire research plan is complete.
