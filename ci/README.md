# Workflows waiting on one account step

`ci/validate.yml` and `ci/publish.yml` are finished GitHub Actions workflows. They are here
rather than under `.github/workflows/` because GitHub refuses any push that touches that
directory unless the pushing token carries the `workflow` OAuth scope, and the token used to
build this repository has `gist, read:org, repo` only. Keeping them in git makes them
reviewable and means they cannot be lost; activating them is a `git mv`.

## Activating them

```bash
gh auth refresh -h github.com -s workflow     # opens a browser, one device code
mkdir -p .github/workflows
git mv ci/validate.yml .github/workflows/validate.yml
git mv ci/publish.yml  .github/workflows/publish.yml
git rm ci/README.md
git commit -m "Activate CI: validate on push, publish to PyPI on release"
git push
```

The first push runs `validate` immediately. Add the badge to the README afterwards:

```markdown
[![validate](https://github.com/howard-lynn-ye/fin-skills/actions/workflows/validate.yml/badge.svg)](https://github.com/howard-lynn-ye/fin-skills/actions/workflows/validate.yml)
```

## What `validate.yml` runs

Four jobs, all of which pass locally as of 2026-09-09.

- **test** - a matrix of Python 3.10 to 3.13 on Ubuntu and Windows: the skills validate
  against the Agent Skills spec, the catalog and the README table and its counts are
  regenerated and `git diff` must stay clean, the importable package must be in sync with
  the skills (`build_package.py --check`), the package imports, and the fast test suite
  passes. Windows is in the matrix because this repo is developed there and the
  console-encoding rule only means something on it.
- **scripts-and-benchmark** - runs every skill script on a default console encoding, runs
  the slow test set, runs the leak benchmark in `--quick` mode, and fails if
  `benchmarks/RESULTS.md` is not what the benchmark actually produces.
- **minimum-dependencies** - installs exactly the floors `pyproject.toml` declares
  (numpy 1.24, pandas 2.0, scipy 1.10) and runs the suite against them. An untested floor is
  a wish, not a claim.
- **lint** - `ruff check`. Deliberately not `ruff format`: formatting the skill scripts would
  force a package regeneration on every run.

## What `publish.yml` needs

It publishes to PyPI on a published GitHub release using trusted publishing (OIDC), so no
API token is stored in this repository or anywhere else. One-time setup, on pypi.org, before
the first release:

1. Log in to pypi.org, go to **Your projects -> Publishing -> Add a new pending publisher**.
2. Enter: PyPI project name `fin-skills`, owner `howard-lynn-ye`, repository `fin-skills`,
   workflow filename `publish.yml`, environment name `pypi`.
3. In this repository's settings, create an environment named `pypi`.

The name `fin-skills` was still unclaimed on PyPI as of 2026-09-09. The distribution builds
and passes `twine check` locally.
