# Frozen launch evidence, not completed model scores

Each folder contains the exact remote source manifest, Slurm command and submission receipt.
The qualification folder also contains its completed software checks and dependency hashes.
The first separate Qwen rerank-answer job was cancelled before execution; the separate Mistral
submission was rejected by the association limit. The combined answer job replaces scheduling
only and verifies both original manifests before running the two models sequentially.

No inference outputs or final quality results are represented by these submission receipts.
All raw responses remain in the RADFM roots recorded in the manifests. Apply the frozen
completion auditor before adding final score summaries. Public FinQA questions are shared
with the memory study and cannot be counted as additional independent samples.
