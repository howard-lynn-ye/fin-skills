# Historical routing check, 2026-09-14 (answer isolation unverified)

The evaluator's supplied prompt contained the saved 127-skill listing and numbered queries.
It omitted skill bodies, implementation code, expected labels and historical answers.
However, no enforced filesystem/tool isolation or independent access audit was recorded.
The earlier description of an "isolated evaluator" overstated the available evidence.
Selections were saved before scoring and have not been revised to match the labels, but
these scores cannot establish that answers were inaccessible. Both question sets and their
labels are now public regression fixtures, not private holdouts. Use the separate
[sealed evaluation protocol](../../docs/EVALUATION_SECURITY.md) for new runs.

| Set | Result | Scope |
|---|---|---|
| Existing 108 queries and unchanged historical labels | 92/108 exact matches (85.2%) | Many labels still name broad routes superseded by new specialist skills. These are raw exact-label results, not adjusted post hoc. |
| 16 new capability questions | 16/16 exact matches | Small implementation-focused smoke set, including four collection questions; not a comprehensive performance estimate. |

Examples of historical mismatches include Korean short-ban rules routed to
`korea-taiwan-markets`, crypto annualization to `crypto-market-structure`, CIK lookup to
`finding-and-searching-data`, and funding/liquidation calculations to `perpetuals-and-funding`.
Other overlaps, such as American option pricing and broker order selection, remain genuine
routing ambiguity. No descriptions were tuned to hide these results.

Reproduce the scores:

```bash
python scripts/eval_blind.py score --output-dir evals/2026-09-14
python scripts/eval_blind.py score --output-dir evals/2026-09-14/new-capabilities --queries evals/2026-09-14/new-capabilities-queries.jsonl
```

The original answers under `evals/` remain a historical snapshot. New capability labels were
written before the evaluator received those queries. Both sets use the same saved listing;
future catalog changes require a new evaluation snapshot.
