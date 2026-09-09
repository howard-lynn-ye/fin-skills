# Security policy

## Reporting a vulnerability

This repository does not yet have a private reporting channel. As of 2026-09-09 GitHub's
private vulnerability reporting is disabled for it (checked with
`gh api repos/howard-lynn-ye/fin-skills/private-vulnerability-reporting`, which returned
`{"enabled": false}`); secret scanning, push protection and Dependabot security updates are
disabled as well.

Until that changes, report a vulnerability in this repository's own code by opening a GitHub
issue at https://github.com/howard-lynn-ye/fin-skills/issues with `[security]` at the start of
the title. Reporters cannot apply labels; the maintainer applies the `security` label on
triage. Include the affected file, the commit or version, a reproduction, and what an attacker
gains. Expect an acknowledgement within seven days. There is no bug bounty.

Because nothing in this repository holds credentials, opens network connections or places
orders (see Scope), a public issue is acceptable for the defects it can realistically have.
If you believe a report is sensitive enough that a public issue would cause harm, open the
issue naming only the affected file, with no details, and say so; the maintainer will reply
with a private channel.

<!-- MAINTAINER: enabling private vulnerability reporting is one switch under
     Settings > Code security and analysis. Once it is on, create the `security` label and
     replace the three paragraphs above with "use the Report a vulnerability button on the
     Security tab". -->

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x (the latest `v0.1.*` tag, and `master`) | yes |
| anything older | no - there is no earlier release |

Security fixes go into `master` and the next patch release. There are no backport branches.

## Scope

In scope: this repository's own code and text - the skill scripts under
`plugins/*/skills/*/scripts/`, their generated copies under `fin_skills/`, the hand-written
`fin_skills/api/` layer, `scripts/`, `benchmarks/`, and the marketplace manifest
`.claude-plugin/marketplace.json`. A skill text that would lead an agent to do something
unsafe - for example an instruction that would send live orders from a client that believes
it is in a sandbox - is in scope too; if it is not a vulnerability in the narrow sense, report
it through the claim-correction issue form instead.

What the code does and does not do:

- The scripts depend only on numpy, pandas and scipy. None reads, stores or transmits an API
  key, password or token, and none connects to a broker, an exchange or a data vendor. The
  test suite refuses every socket connection (`tests/conftest.py`), so a script that gained a
  network dependency would fail its own tests.
- The library ships guidance on broker execution (`broker-execution-apis`, `lib-alpaca-py`,
  `lib-ib-async`) and one guard, `paper_account_guard`, that fails closed unless a
  server-returned account id, host or header proves a session is paper. The guard checks facts
  you hand it; it never logs in. No skill in this repo places an order.
- The skills document hazards in third-party libraries - for example that tushare sends its
  token over plaintext HTTP. Those are properties of the upstream library, not vulnerabilities
  in this repo: report them upstream, and open a claim-correction issue here only if the
  documentation of them is wrong or stale.

Out of scope:

- **Federated third-party packs.** `.claude-plugin/marketplace.json` lists eight skill packs
  by their own GitHub source. They install disabled, and this repo does not audit them. In the
  words of `catalog/federation-notes.md`, what was checked for each pack is "Licence (from the
  repository), `pushed_at` and stars (GitHub API), that real `SKILL.md` files with `name` and
  `description` frontmatter exist at the declared path, and the skill count", and "Nothing
  inside the skills was verified". A vulnerability in a federated pack belongs to that pack's
  maintainers; what you can report here is that an entry should be removed or its description
  changed.
- The libraries the skills describe (yfinance, vectorbt, QuantLib, ccxt and the rest) and the
  data vendors and brokers they connect to.
- The runtimes that load the skills (Claude Code, claude.ai, the Claude API). Treat any skill
  from any source as software you install; the Agent Skills documentation says so, and it
  applies to this repo as much as to the federated packs.

## Dependencies

Runtime dependencies are `numpy>=1.24`, `pandas>=2.0` and `scipy>=1.10` - floors only, no
lockfile. Dependabot alerts are disabled, so advisories in those packages are not tracked here
automatically; keep them current in your own environment.
