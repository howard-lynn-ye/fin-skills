# Fixed-seed replication and within-allocation cost comparison

Frozen after the complete seed-11 pilot audit on 2026-09-23 UTC. The pilot remains a
separate immutable result. This follow-up addresses two previously declared limitations:
one inference seed and comparisons of latency across separate dynamic GPU allocations.
It does not provide new independent tasks or external generalization evidence.

Run the same eleven conditions with inference seeds 23 and 37 for both pinned models.
Each model/seed job runs all eleven conditions in a seeded random order within one dynamic
GPU allocation, with one model load and one router calibration. Each condition starts
with a fresh memory state. Thus comparisons within that model/seed use the same allocated
GPU. Allocation remains generic; no nodes, exclusions or GPU models are pinned. This is
not a claim that both model families have identical compute or tokenization.

The fixed input generator remains seed 101: four authored financial mechanisms, eight
client contracts and two acquisition plus two later synthetic blocks per contract.
There are 352 episodes per job, four jobs, and 1,408 planned episodes, of which 704 are
later evaluation. The sampling seeds are replicates of the same tasks, not independent
tasks. Report each seed and task mechanism; do not pool correlated episodes into an
independent-sample confidence interval or pick the best seed.

The method implementations, strict exact-JSON parser, scoring predicates, reference
formulas, memory updates, router calibration, 50%/25% history budgets, beta=0.1 and three
384-token response opportunities per episode are unchanged. Do not strip Markdown
fences, fill parameters, tune thresholds or increase budgets in response to pilot errors.
The sole study-code addition is a `joint` group concatenating the eleven existing unique
conditions. The joint group is excluded from its own concatenation. Each job repeats the
existing numerical/decoder/physical-cache qualification before inference.

Primary quality comparisons remain complete task success, method choice, parameters,
numeric answer and receipt use, with acquisition and later outcomes separate. Include
all missing, failed or rejected attempts. Record supplied and retained experience IDs,
state hashes, choices and actual API receipts. Report all ordinary-memory controls; a
tie between frozen, retrieval and fly cannot establish a benefit of fly's online updates.

For cost, compare policies within a model/seed allocation and report actual output length,
input tokens, retained tokens, prefill and complete generation latency, actual GPU
allocated/reserved peaks, physical KV bytes, calibration time and job wall time. Total
runtime remains affected by different generated lengths; shorter KV is not automatically
lower end-to-end latency. Allocator reserved-memory peaks depend on execution history,
so report them without treating them as isolated per-condition requirements. The recorder
continues to count missing cost traces from failed calls explicitly.

Stop after these two prespecified seeds and audit all four jobs. Do not keep adding seeds
until a preferred method wins. Independent externally authored tasks, real time-period
transfer and human/content-support evaluations still require their own data and labels.
This bounded replication cannot close those evidence gaps or establish conference readiness.
