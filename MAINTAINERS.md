# Maintainers

| GitHub | Role | Verified |
|---|---|---|
| @howard-lynn-ye | Maintainer and repository owner: writes, reviews, merges, releases | repository owner; `admin` on the collaborators API, 2026-09-09 |
| @Shwai-He | Admin collaborator: can merge and release if the maintainer is unavailable | `admin` on `repos/howard-lynn-ye/fin-skills/collaborators`, 2026-09-09 |

## Bus factor

The bus factor is one. Every commit to date was made by the maintainer (61 of 63 carry the
maintainer's git identity), and the maintainer alone holds the GitHub Pages, PyPI and Zenodo
logins. A second admin exists so the repository can be kept alive without him, but nothing
has yet been merged or released by anyone else.

## How decisions are made

- **Through issues.** Anything that changes what the repository claims starts as an issue.
  The claim-correction form (`.github/ISSUE_TEMPLATE/claim-correction.yml`) is the main one:
  a file and line, what it says, what you observed, and the versions and date. A reproduction
  settles a claim; an opinion does not, including the maintainer's.
- **Numbers are produced, not typed.** A number in a skill is printed by a script in that
  skill or by a command whose output the author saw. A pull request that states a number it
  did not produce is not merged, whoever opened it (`CONTRIBUTING.md`, `AGENTS.md`).
- **Scope is set by the README.** "Scope" in `README.md` and the "Out of scope today" section
  of `ROADMAP.md` say what the repo does not cover; a proposal to change that is an issue,
  argued from what the neighbouring repositories already own.
- **The regeneration chain is the gate.** `build_index.py`, then `build_package.py`, then
  `validate.py` must leave the tree clean and print OK, and `pytest` and `check_scripts.py`
  must pass, before anything is merged. Generated files are never edited by hand.
- **Releases** are semantic versions tagged `vX.Y.Z` by the maintainer, with the entry in
  `CHANGELOG.md` written before the tag.

## Contact

GitHub issues only; there is no mailing list, chat, or published e-mail address. Security
reports follow `SECURITY.md`; conduct reports follow `CODE_OF_CONDUCT.md`.
