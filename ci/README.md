# CI and publishing

Updated 2026-09-14. The working-tree `.github/workflows/ci.yml` now defines Linux/Windows
Python 3.10-3.13, macOS 3.12, minimum dependencies, isolated optional extras, coverage,
slow tests, scripts, benchmarks and clean distribution installation. These expanded jobs
have not yet been observed running on GitHub. The earlier Ubuntu-only v2 run
[completed successfully on September 13](https://github.com/howard-lynn-ye/fin-skills/actions/runs/34773998923).
The earlier statement that the repository had no active CI was outdated.

`ci/validate.yml` is a historical inactive template, now superseded by the active workflow
definition. Its presence does not prove those jobs have run. Pushing workflow changes may
require GitHub credential scope; a local file edit alone does not establish remote access.

`.github/workflows/release.yml` builds and checks release-candidate artifacts. It does not
upload to PyPI. See `docs/API_COMPATIBILITY.md` for release acceptance and migration rules.

`ci/publish.yml` is also inactive. Before moving it into `.github/workflows/`, the maintainer
must configure the PyPI pending trusted publisher for owner `howard-lynn-ye`, repository
`fin-skills`, workflow `publish.yml`, environment `pypi`, and create that GitHub environment.
Then a published GitHub release can trigger the workflow. It uses OIDC rather than a stored
PyPI token. The PyPI project endpoint returned 404 on 2026-09-14; GitHub installs work
independently of PyPI publication.
