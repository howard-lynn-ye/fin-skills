# Tools and memory framing revision

Date: 2026-09-22. User-authorized manuscript revision following the initial merge.

The user selected professional tool use and experience-based decision improvement as
the central research direction. Error reduction is a supporting outcome, not the main
contribution. The revised title is **From Tool Access to Research Capability: Tools and
Memory for Financial Agents**. Capability development is the research question; no
integrated learning gain is asserted.

## Changes

- Rewrote the title, abstract, introduction, related-work organization, discussion and
  conclusion around method selection, information retention and adaptation.
- Moved persistent memory before the empirical study and clarified its difference from
  within-task KV selection. The numerical adapter is not described as an implemented
  LLM tool-selection learning policy.
- Added a capability-evaluation section with explicitly proposed comparisons of tools,
  context and memory. No planned ablation is presented as completed.
- Reduced execution/feedback checking to a supporting section. Preserved the complete
  prior audit and validation material in `supporting_evidence.tex`, with the study's
  original outcomes and limitations.
- Kept professional tool interfaces in the main text and moved the full interface
  inventory and implementation scope to `library_details.tex`.
- Retained autonomous-use numbers, comparison arms, exposure and resource information;
  retained the recommendation tables and their author-reported evidence status.
- Added the Chinese framing document and pointers from the historical outline and
  superseded unified plan. Their existing bodies and pre-existing edits were retained.

## Interpretation

The observed lack of a mean return advantage is limited to the tested development
setting. It does not establish a general law about tools or identify a context/memory
failure. HiSTrim and persistent memory are hypotheses for further comparisons, not
proven remedies. Check results establish only their stated scope; valid losses remain
eligible feedback. No claim is made that audit or software-resource papers are less
publishable than other contribution types.

The pre-revision files are retained in `reframe_before/`, and commit `758c618` preserves
the initial merge. The latest PDF and sources are in `paper/latex_naacl/`. This note
records framing changes, not a new empirical result or submission approval. Delivery
verification is recorded separately in `reframe_delivery.json` after completion.
