# Roadmap

Updated 2026-09-14. The acceptance record for the recovered Claude work and current
implementation is [docs/COMPLETION_AUDIT.md](docs/COMPLETION_AUDIT.md).

## Implemented

- Importable skill texts and executable modules, a shared `Bundle`/guard API, JSON tools
  and an optional MCP server.
- Market data and discovery adapters, a reference research engine, synthesis, pre-trade
  checks and read-only bridges to external research outputs.
- Model and strategy families, fixed income, credit, macro, tax and alternative data.
- Deeper Japan, Hong Kong, India, Korea/Taiwan and ASEAN market skills; token events,
  perpetual funding, AMM mechanics and crypto market structure.
- Trading sessions, corporate-action processing and data-quality validation, including
  the `data_quality` API guard.
- [Public information collection](docs/COLLECTION.md): RSS/Atom, bounded static crawling,
  SEC Form 4/13F, House PTR and Bluesky. SQLite checkpoints, revisions, retry scheduling
  and an alert outbox support repeated polling. Operators choose and start their watches.
- Worked examples, an API documentation site, the v0.1.0 release and active CI on Ubuntu
  with Python 3.11–3.13. The original release predates this completion work.

## Remaining validation and distribution

- Refresh blind skill-selection measurements whenever the listing changes. Historical
  scores apply to their saved listing, not automatically to an expanded catalog.
- Extend active CI to Windows, Python 3.10 and minimum dependency versions using the
  reviewed `ci/validate.yml` template. The current token lacks `workflow` scope; existing
  CI remains active. See [ci/README.md](ci/README.md).
- Publish to PyPI after the maintainer configures a trusted publisher and GitHub `pypi`
  environment. A checked wheel is not a published package.
- Link Zenodo and archive a release before adding a DOI to `CITATION.cff`.
- Directory submissions already opened await third-party review. The human-only
  awesome-claude-code form has its own eligibility requirements; do not duplicate PRs.

## Ongoing maintenance

Source API changes, vendor licenses, market-rule effective dates and third-party skills
need dated rechecks. New exchanges, sources and distribution channels such as conda-forge
can be added as concrete requirements arise. Source-specific credentials and operating a
continuous collector are deployment choices, not capabilities enabled by importing Python.

No module in this completion executes live orders. Public-disclosure tracking observes
published reports and statements, with their delays and scope, not private trading accounts.
