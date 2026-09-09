# Where this repo has been submitted, and what is still owed to whom

A record so a submission is never re-derived, re-sent, or sent somewhere that forbids it.
Verified against each list's own contributing rules on the date shown.

## Open pull requests

| List | PR | Opened | State |
|---|---|---|---|
| ComposioHQ/awesome-claude-skills | [#1863](https://github.com/ComposioHQ/awesome-claude-skills/pull/1863) | 2026-09-08 | open |
| BehiSecc/awesome-claude-skills | [#692](https://github.com/BehiSecc/awesome-claude-skills/pull/692) | 2026-09-08 | open |
| ccplugins/awesome-claude-code-plugins | [#456](https://github.com/ccplugins/awesome-claude-code-plugins/pull/456) | 2026-09-08 | open |

Each is a one-line entry with authorship disclosed.

## hesreallyhim/awesome-claude-code - eligible 2026-09-18, and a human must send it

Its `CONTRIBUTING.md` (read 2026-09-09) sets two rules that matter here.

**Eligibility is either-or, not a wait:**

> (i) Be at least 14 days old (14 days since first commit on default branch) AND show signs of
> active development (I expect there to be also additional commits after the first day);
>
> (ii) Have at least 100 stars.
>
> Resources that fail these criteria will be closed automatically.

This repository's first commit is 2026-09-04, so (i) is satisfied on **2026-09-18**; the
active-development half has been satisfied since the first day. (ii) would allow an earlier
submission at 100 stars.

**Two procedural rules, both binding:**

> **ALL RECOMMENDATIONS MUST BE MADE USING THE WEB UI ISSUE FORM TEMPLATE, OR YOU RISK BEING
> RESTRICTED FROM INTERACTING WITH THIS REPOSITORY TEMPORARILY.**

> Although resources themselves may be partially or entirely written by a coding agent,
> resource recommendations must be created by human beings.

So this one cannot be opened by an agent or by the API, and the text below is a draft for a
person to paste into the form at
<https://github.com/hesreallyhim/awesome-claude-code/issues/new/choose> (template
`recommend-resource.yml`). Check the form's current fields before pasting; they may have
changed.

### Draft

**Resource name:** fin-skills

**URL:** https://github.com/howard-lynn-ye/fin-skills

**Category:** Skills (or Plugins - the marketplace ships both)

**Author:** howard-lynn-ye

**License:** MIT

**Description** (trim to the form's limit; the first sentence stands alone):

> Agent Skills for Python quantitative finance that tell an LLM which library to use, what
> each one silently gets wrong, and whether a backtest result is real. Every claim carries a
> verification date and a marker for how it was checked - verified at a primary source,
> secondhand, or could not verify - and the numbers in each skill are produced by that
> skill's own runnable script rather than quoted. Ships as a plugin marketplace and as an
> importable Python package, with the same checks available as a unified API, as JSON-callable
> tools, and as an MCP server. A benchmark plants known defects in synthetic data and reports
> which guard catches each one.

**Why it belongs on the list:** it is the research-integrity half of quant work, which the
existing finance entries do not cover; the claims are dated and falsifiable rather than
asserted; and it is disclosed as agent-written per the list's own rule.

**Disclosure to include in the form:** the skills were written with Claude Code. The
recommendation itself must be written and submitted by a person - that is the list's rule, not
a formality.

## Lists deliberately not submitted to

| List | Why |
|---|---|
| VoltAgent/awesome-agent-skills | Its policy requires demonstrated adoption before a new skill repository is listed. Revisit once there is usage to point at. |

## Before submitting anywhere new

Read that list's contributing rules first and record them here. Two things have already
differed between lists: whether a pull request is accepted at all, and whether a submission
may be machine-generated.
