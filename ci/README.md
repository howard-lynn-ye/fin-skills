# CI and publishing

Updated 2026-09-14. `.github/workflows/ci.yml` is active: it validates the package, runs API
and full default tests, and runs the leak benchmark on Ubuntu/Python 3.11–3.13. The v2 run
[completed successfully on September 13](https://github.com/howard-lynn-ye/fin-skills/actions/runs/34773998923).
The earlier statement that the repository had no active CI was outdated.

`ci/validate.yml` is a broader, inactive template: Windows plus Ubuntu/Python 3.10–3.13,
minimum dependencies, scripts, slow tests and lint. Its presence does not prove those jobs
have run. Extending the active workflow requires a GitHub credential with `workflow` scope;
the current local token has `gist`, `read:org`, and `repo` only.

`ci/publish.yml` is also inactive. Before moving it into `.github/workflows/`, the maintainer
must configure the PyPI pending trusted publisher for owner `howard-lynn-ye`, repository
`fin-skills`, workflow `publish.yml`, environment `pypi`, and create that GitHub environment.
Then a published GitHub release can trigger the workflow. It uses OIDC rather than a stored
PyPI token. The PyPI project endpoint returned 404 on 2026-09-14; GitHub installs work
independently of PyPI publication.
