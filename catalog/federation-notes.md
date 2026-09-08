# Federation notes

How third-party skill packs get into this marketplace without being copied, what was checked, and
what was left out and why. Verified 2026-09-08.

## The mechanism

A marketplace plugin entry's `source` may be an external repository (from the Claude Code plugin
marketplace reference, code.claude.com/docs/en/plugin-marketplaces, read 2026-09-08):

- `{"source": "github", "repo": "owner/repo"}` with optional `ref` (branch or tag) and `sha`
  (full 40-hex commit)
- `{"source": "git-subdir", "url": "owner/repo", "path": "sub/dir"}` for a plugin inside a monorepo,
  fetched as a sparse clone
- `url`, `npm`, `archive`, `command` forms also exist and are not used here

`skills` paths on an entry *add* to the default `skills/` scan and must start with `./`. `metadata`
is free-form and **Claude Code does not read it** - which is why `scripts/validate.py` requires the
provenance fields on every external entry itself. `defaultEnabled: false` installs a plugin disabled
until the user turns it on. Version resolution is `plugin.json` version, then the marketplace entry's
`version`, then the commit SHA - so an upstream that pins its own `version` only updates when it bumps.

`claude plugin validate .` (Claude Code 2.1.257) accepts external-source entries; it checks shape,
duplicate names and path traversal, not existence. The repository's `$schema` pointed at a URL that
now 404s; the catalogued schema is `https://json.schemastore.org/claude-code-marketplace.json`, and
it lags the docs (no `archive` or `command` branch).

**Federation is per plugin.** Nothing inside a flat `skills/` directory can be excluded, so a pack is
taken whole or not at all. Packs of 68 and 74 skills therefore cost three to four times the default
skill-listing budget by themselves; the descriptions say so.

## What this repo verified for each federated pack

Licence (from the repository), `pushed_at` and stars (GitHub API), that real `SKILL.md` files with
`name` and `description` frontmatter exist at the declared path, and the skill count - all on the
`verified_on` date in the entry's `metadata`. **Nothing inside the skills was verified**, and several
packs contradict findings this repo has measured (noted in their descriptions).

## Considered and left out

| Repository | Why not |
|---|---|
| JoelLewis/finance_skills (compliance sub-plugin) | Its `plugin.json` declares `"dependencies": ["core"]`, which resolves *in the federating marketplace*; whether enable succeeds without also listing its `core` under that exact name is unverified. Revisit once tested. |
| ajeeshworkspace/indian-trading-skills | Hard dependency on a "Groww MCP" whose provenance could not be established; 4 of 10 skills are derived copies of tradermonty's. |
| ml4t/skills | 61 skills in 8 category directories with non-spec frontmatter keys (`when_to_use`, `dependencies`, `paths`); a `strict: false` entry would work in Claude Code but the files hard-fail claude.ai upload. Scope overlaps research-integrity-guards and backtest-validation. |
| gauss314/skills | At least eight skills self-labelled as scrapers of Finviz, Macrotrends, MarketWatch, TradingView and others - terms-of-service exposure that cannot be excluded from a flat `skills/`. |
| openaccountant/skills | `plugin.json` lists skills without the `./` prefix; the validator reports it as a load failure and `strict: false` cannot override a manifest that declares components. Blocked until upstream fixes it. |
| alirezarezvani/claude-skills `finance/` | Corporate ratio, DCF and SaaS-metrics content; out of this library's scope. |
| prof-little-bear/cc-equity-research | `SKILL.md` files carry no frontmatter at all. |
| BaggaT236/AI-Trading-Skills | Sample `SKILL.md` byte-identical to tradermonty's. |
| Superior-Trade, GMGNAI, komako-workshop/digital-oracle, cmoralesm/P123 | Platform-locked, single root `SKILL.md`, or a marketplace `source` of `"."` that violates the `^\./` pattern. |

## Runtime behaviour not verified

Whether an entry `name` that differs from the upstream `plugin.json` `name` loads cleanly (the
validator accepts it; the docs are silent - the two entries here whose upstream has a manifest use
the upstream name); whether string-form `dependencies` resolve at install; a bare `"./"` as a
`skills` path. Installation itself was not exercised, since it writes to `~/.claude` and clones
third-party code.
