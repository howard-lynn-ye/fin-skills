# Beacon bounded campaign closeout

Status: 2026-09-23, 08:21 UTC verification. Every submitted job belonging to this
bounded campaign is resolved and its intended result or recorded failure is accounted
for. The final 192-cell LlamaIndex tool comparison is complete and audited. No campaign
inference remains running or queued. Other projects' jobs were not modified. Automatic
follow-up is paused under the user's closeout instruction (app status confirmed PAUSED);
this does not declare
the entire research plan complete.

## What the evidence supports

| Question | Measured evidence | Claim boundary |
|---|---|---|
| Can the library and agent actually execute the proposed loop? | Numerical calibration, true tool receipts, visible prior experiences, frozen data and independent scoring were checked. | Engineering validity is not generalization or performance superiority. |
| Does organizing tools improve decisions? | Actual LlamaIndex comparison: generic/flat/organized correct totals are Qwen 25/28/26 and Mistral 10/13/3, each out of 32. | No stable organized-catalog advantage; naming, repeated text and placement are bundled. Same authored contracts, not new external tasks. |
| Does experience help a later external task? | FinQA Qwen none/recent/retrieval/frozen/learned scores are 3/7/5/6/6 out of 32; Mistral scores are all zero. | Some Qwen experience conditions improve observed counts; learned conditioning does not exceed frozen. Evaluation memory is frozen, not online learning. |
| Does HiSTrim complement long-term memory? | Physical KV reduction and retention are measured; DocFinQA correctness shows no HiSTrim gain for either model. | Lower KV is not equal-quality speed or total reserved-memory superiority. Full means retrieved pack, not an entire report. |
| Does the library outperform mature retrieval components? | FinSkills and BM25S have equal aggregate page-hit counts; BM25S queries are much faster. BGE improves top-1 page hits from 9 to 14 out of 150. | No whole-library speed advantage; BGE is not Jev, and page hits are not answer correctness. |

Detailed failure denominators, costs, immutable hashes and protocols:

- [Framework tools](FRAMEWORK_TOOLS_RESULTS_20260923.md).
- [External FinQA memory](FINQA_MEMORY_RESULTS_20260923.md).
- [DocFinQA context](DOCFINQA_CONTEXT_RESULTS_20260923.md).
- [FinanceBench retrieval](FINANCEBENCH_REUSE_RESULTS_20260923.md).
- [Original campaign history](BEACON_CAMPAIGN_20260923.md) and
  [reuse execution history](REUSE_EXECUTION_20260923.md).

The older 2,112 decision-loop episodes and earlier negative/failed attempts remain
unchanged. Repeated seeds, arms and overlapping FinQA/DocFinQA questions are not new
independent tasks. Public author labels were genuinely reused but do not establish that
the pretrained models never saw them. Do not populate the manuscript with stronger or
nonexistent scores, and do not overwrite the user's subsequent manuscript edits.

## Remaining research, separated by reason

- **External access:** Jev requires a genuine TYPESAFE_API_KEY. No authenticated Jev
  result exists; BGE does not fill that provider-specific gap.
- **Human participation or new data:** measured human efficiency, new independent
  annotation of generated claims, and prospective/private time-transfer evidence remain
  outstanding. Existing public answer/evidence labels have already been used and are
  not classified as missing.
- **Different system-level task:** native FinRobot equity reporting and FinMem/
  InvestorBench trading comparisons remain uncompleted. They require a matched
  task/policy, data and cost/risk protocol rather than renaming the present QA adapter.
  Their native embedding/data-service dependencies and possible local adaptations are
  documented in [the scope review](FRAMEWORK_SCOPE_REVIEW_20260923.md). We have not
  established that all such adaptations are impossible or that all provider keys are
  absent. They are outside this fixed QA/method-selection closeout, not failed jobs.

Within the agreed bounded main-line comparison, no necessary input-ready batch remains.
Do not create another seed sweep, tune on observed errors, repeat accepted jobs or start
an unrelated trading program to keep the queue busy. Resume this follow-up when the
specific external inputs arrive or the user selects a new system-level protocol.
