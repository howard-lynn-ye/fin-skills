# Local and Overleaf manuscript merge

Status: merged working draft; not submission-ready or a jointly validated system.

The user identified the local manuscript and Shwai He's shared Overleaf project as the
same article and requested a merge retaining the strengths of both. The destination is
https://www.overleaf.com/project/6aad9a03f27c3c07a182965d.

## Sources and preservation

- `local_before/`: complete copy of the local `paper/latex_naacl/` before this merge,
  including the user's uncommitted manuscript changes.
- `overleaf_before/`: full project downloaded through the authenticated Overleaf UI.
  Downloaded ZIP SHA-256:
  `55ae7aac5ad93ea697cf7bfc5eea3281ade2becc6e6312eaeee724e2fd5bdd9f`.
- `overleaf_finskills/`: the FinSkills ZIP retained inside that project, extracted for
  comparison with both the current HiSTrim entry point and the local financial draft.
- `source_hashes.json`: file hashes for all three snapshots. These backups are local;
  the merge does not imply public release of the shared manuscript or source archives.

Overleaf history showed Shwai's FinSkills upload on September 22 at 09:35 and a later
HiSTrim upload at 19:05, using the timestamps displayed by the interface. The original
online entry point is preserved in the upload bundle as `main_histrim_before_merge.tex`.
Its `sections/`, figures and bibliography are not overwritten by the merged entry point.

## What the merged manuscript retains

| Source | Retained contribution | Location in merged manuscript |
| --- | --- | --- |
| Local financial draft | Current library/model/RAG interfaces, artifact-bound audit, independent evaluator and financial results | Library design through autonomous-use study |
| Local memory draft | Complete-episode feedback, availability/lineage checks and matched learning controls | Fruit-fly-inspired learning |
| Online FinSkills draft | Concrete mechanical versus interpretation failure examples and explicit evaluation scope | Framework overview and audit motivation |
| Online HiSTrim draft | Protected history selection, sigmoid sign property, normalized router fitting, tangent updates and physical compaction | Selective context with HiSTrim |
| Online HiSTrim results | All generative MRR baselines/tasks and selected contrasting rating/ranking results | Recommendation evidence appendix |
| Online writing notes | Overview before equations, explicit task comparisons, task metrics separated from teacher metrics | Abstract, framework table and evidence appendix |

The merged title is **Verified Execution and Selective Memory for Financial Research
Agents**. It is a working title, not a claim of final author approval. The existing
`OUTLINE.md` contains a separate FAST naming proposal and was not overwritten.

The common argument is that context selection, tool execution and learning feedback
need distinct checks. The manuscript explicitly says their integration has not been
jointly evaluated. Recommendation gains are not treated as financial return gains.

## Scientific corrections and evidence limits

1. The sigmoid gate preserves the utility logit's sign and is monotone in that logit
   with the gate fixed. This does not guarantee correct item selection after
   normalization and thresholding.
2. Normalized direction averaging fixes the norm only when its inputs and average are
   nonzero. Unconditional variance reduction and convergence claims were not retained.
3. The count-based sink offset is a heuristic. Exact denominator replacement requires
   the removed exponential mass; even that does not replace its value-weighted output.
4. Keeping protected tokens does not establish unchanged mutual information or hidden
   states. The merged notation also separates item indices from token positions.
5. The five reported seed rows yield Pass@32 mean 18.78 and sample SD 0.7224956747;
   the main source text's 18.79 was not copied. MRR times 100 has mean 3.808 and sample
   SD 0.1883348083. These are arithmetic checks on rounded manuscript rows.
6. HiSTrim run IDs appear in the source, but the original jobs, code and predictions
   were not accessed. The draft labels its retained tables author-reported, omits
   unsupported significance markers and does not combine incompatible caption maxima.
7. Online financial claims about FinGuardBench-180, FinGuardBench-60 compliance, broad
   KOL profitability, exact parity and asymptotic scaling were not promoted over the
   locally recorded evidence. Their original text remains in the backup.
8. No new experiment, joint-system result, energy saving, author list, data permission
   or submission declaration was invented. Human scientific review remains pending.

## Reproduction

Run `python paper/merge_20260922/build_merge_evidence.py` with the preserved source
snapshot in place, and `python scripts/build_paper_evidence.py` for the financial macros.
The first command extracts every generative result row and all five seed rows from the
source TeX, checks algebraic examples and writes `merge_evidence.json`, `merge_numbers.tex`
and `histrim_source_table.tex`. These checks are not independent empirical replication.

Compile the manuscript with pdfLaTeX, BibTeX, then pdfLaTeX until references stabilize.
Compilation uses a clean staging directory to avoid incompatible old auxiliary files.
The final PDF and source remain under `paper/latex_naacl/`.

## Delivery verification

Local build, rendering, repository checks and remote synchronization are recorded in
`delivery.json` when completed. A prepared upload is not evidence of remote delivery.
