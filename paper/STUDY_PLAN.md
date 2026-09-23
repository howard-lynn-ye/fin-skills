# Verified execution for financial research agents

Status: revised study protocol, 2026-09-21. This supersedes the unified blueprints and
overnight delivery claims. It is an amendment made after seeing the historical pilot,
not a retrospective claim of preregistration. One paper remains the intended outcome.

## Question and contribution

Does attaching verifiable execution evidence to a financial agent's final research
artifact reduce incorrect accepted conclusions, beyond text guidance and optional tools?
The library supplies domain knowledge and executable checks. Independent perturbation
and accounting tests assess outputs. Neither the number of skills nor a high backtest
return is the scientific contribution by itself.

## Conditions and budgets

Use four conditions: no library excerpt, excerpt only, excerpt plus optional guards,
and excerpt plus mandatory final audit. Ordinary accounting inspection is available
in all conditions. Keep market, model snapshot, inference seed, turn cap, output-token
cap, temperature and available data paired. Actual tokens and elapsed time must also
be reported: equal limits do not imply equal consumption. Randomize execution order.

The initial Beacon feasibility run uses three public development seeds (11, 23, 37),
Qwen2.5-Coder 7B/14B/32B, four conditions and one repetition. It tests operation and
informs power/cost planning. It cannot validate size-monotonic claims or a general
compliance paradox. Exact model revisions are frozen before weight download; each
model's protocol is frozen before inference. Full scaling and independent holdout
collection follow only after instrument checks, with amendments recorded explicitly.

Transport amendment, 2026-09-21: the first Beacon preflight was stopped after the runner
rejected valid tool-call JSON solely because it appeared in Markdown code fences. The
raw runs and frozen source were archived separately; their source hashes match the saved
protocol. The revised parser accepts one JSON object or one fenced JSON block, preserves
the original response, rejects ambiguous/malformed calls, and does not repair arguments.
All conditions use the same parser. New runs use new result directories and frozen hashes.
This fixes an interface failure; the aborted preflight is not evidence that the models
could not solve the financial task.

## Outcomes and analysis

Primary: incorrect accepted submissions / all attempted submissions, paired by market.
Also report correctness among accepted outputs and the acceptance rate: blocking every
submission is not useful completion. A rejection, timeout, provider error, missing output,
ungradable artifact and failed task are separate outcomes, never silently removed.

Secondary: independent future/same-session dependence, absolute reporting discrepancy,
post-delisting exposure, unexecuted citations, stale citations, repair success, tool calls,
tokens, wall time and GPU allocation time. A guard may have executed and failed; an
execution citation alone does not claim it passed. Preserve that distinction.

Current correctness criteria for development summaries: finite independent grade,
absolute Sharpe gap <= 0.05, zero future/same-session dependence and zero post-delisting
mass. These are operational criteria, not proof of universal financial validity. Publish
raw metrics and sensitivity to thresholds. Scores use a fixed-cost oracle while public
accounting feedback uses stated costs; report both to identify convention mismatches.

For repeated studies, bootstrap paired differences by market, preserving all repetitions
and conditions within each sampled market. Do not treat seeds, perturbations, dates and
agent runs as interchangeable independent observations. Use paired binary comparisons
where appropriate. Small feasibility cells are descriptive only. Freeze a practically
meaningful effect and power calculation before choosing the confirmatory sample size.
Failure to reject a difference in returns does not establish equivalence.

## Instrument and data separation

Public intervention adapters call the actual library guards; final checks also test
accounting and sampled same-session dependence. Evidence includes hashes of final code,
data and numerical claims. Unknown names, stale receipts and missing checks cannot
establish completion. The independent oracle is never called for agent repair feedback.

Add separately authored defect mechanisms and realistic clean cases before a general
detection claim. Known planted defects used while developing guards are regression tests.
The public benchmark and pilot have been inspected during development. Do not relabel
them as unseen. A container/process alone does not establish that labels were inaccessible:
document mounted paths, network access and scorer separation. Local subprocess mode is
only a fault boundary. Beacon workers additionally require Landlock and seccomp, tested
against outside reads/writes and sockets. These host restrictions do not establish an
in-process evaluator's resistance to malicious Python monkeypatching.

## Real-world evidence

The separate [library-utility study](LIBRARY_UTILITY_RESEARCH.md) asks whether the same
trading agent benefits from library access. Its first run uses existing ECB reference
data as an explicitly non-executable price proxy. Keep its return outcomes separate
from the artifact-correctness outcomes here; neither implies the other.

The KOL fixture permits recomputing profile aggregates. It does not contain predictions,
outcomes or point-in-time credibility states needed to reproduce predictive results.
`benchmarks/prediction_audit.py` requires row-level variant/date/asset predictions and
outcomes, prediction time, feature availability, fitting-label availability, and label
end time. Credibility-update timestamps are checked when supplied. Export those data
from the original experiment; do not infer missing timestamps from narrative summaries.

Report both pooled and average daily cross-sectional Rank IC. Use paired date blocks
at least as long as the target horizon for uncertainty; document longer dependence and
block-length sensitivity. KOL tiers chosen with evaluation-period outcomes cannot be
used as an independent demonstration that a forecast strategy works.

## Related work and publication

CheckList motivates behavioral testing; Profit Mirage already studies financial-agent
leakage and counterfactual interventions. Finance Agent Benchmark evaluates financial
research tasks. Compare the contribution in execution provenance, accepted-output risk,
and text-versus-enforcement experiments, not merely the financial application domain.

The FinVault arXiv record was withdrawn on 2026-07-30. It can be recorded in the search
history, but its earlier abstract must not be treated as validated current evidence.
Sources and machine-derived numerical evidence are in `evidence.json` and `RELATED_WORK.md`.
Human verification, author details and final submission approval remain outstanding.
