# Beacon execution checkpoint

Observed remotely at 2026-09-23 05:19:27 UTC. This checkpoint supplements the campaign
record; it does not replace the frozen protocols or report new model-quality results.

The earlier September 22 handoff is stale: the campaign has since submitted and
completed multiple batches. Read `BEACON_CAMPAIGN_20260923.md` and the decision result
documents before reporting that no jobs were submitted.

## Direct scheduler observations

| Jobs | Study | Observed state |
|---|---|---|
| 1652208, 1652209, 1652210, 1652211 | Decision replication | COMPLETED, exit 0:0 |
| 1649355 | Same-gate learned/frozen fly comparison | COMPLETED, exit 0:0 |
| 1649462 | Retrieval/downstream development pilot | COMPLETED, exit 0:0 |
| 1649796 | Native-component/database comparison | COMPLETED, exit 0:0 |
| 1653290 | Public benchmark and framework preparation | RUNNING |
| 1613767_[2] | Earlier single-GPU 32B recovery | PENDING, Priority |

For job 1653290, logs report acquisition of FinQA, FinanceBench and Reflexion at
fixed revisions. Neither `completion.json` nor `qualification.json` existed at the
check. Download progress is not successful dependency qualification or model inference.
The scheduler reports no requested or excluded nodes; placement is dynamic.

Remote preparation root:
`/beacon-projects/radfm/wy891/fin-skills-campaign-reuse-bootstrap-20260923-v1`.
Its working directory, scheduler logs, temporary files and configured caches are in RADFM.
Do not resubmit this accepted job or alter its frozen source while it runs.

## What remains

The campaign documents record 704 pilot and 1,408 replication episodes audited, along
with completed development pipeline, retrieval, historical-data and repair comparisons.
This is not completion of the entire research program. External benchmark evaluation
and the mature-framework comparison still require qualification and execution. Jev needs
its actual API credential; the current process has none configured. Human efficiency
requires participants. Independently authored defect cases and prospective time-transfer
evidence remain separate requirements.

Mixed and negative outcomes remain in the results. Do not add seeds merely to obtain a
preferred outcome. Scheduler completion and integrity audits do not establish superiority.
