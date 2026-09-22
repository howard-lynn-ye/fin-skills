# API compatibility and release policy

The current package remains Alpha. This change does not claim a stable 1.0 API or PyOD-level
adoption. Existing `run`, `auto_run`, `recommend(Request)`, JSON tool names and `GuardResult`
interfaces remain available. The additions are `research`, `profile_data`, `fit`, `load_model`,
`doctor`, `ResearchResult`, and `FittedModel`.

## Compatibility contract

### Audit verdict tightening (2026-09-21)

`RunReport.passed` is now false when no checks ran or an input was rejected. Skipped
checks remain distinct from failed checks. For a release decision, call
`report.audit(["assert_causal", ...])` with an explicit non-empty task policy. It returns
`PASS`, `FAIL`, or `INCOMPLETE`, required-check coverage and missing/rejected guards.
An ordinary passing subset is not a certificate of full research integrity. Existing
callers relying on `all([]) == True` must handle incomplete input explicitly.

The agent-study intervention now uses four conditions and real artifact adapters. The
old three-condition scaffold receipt is historical test data, not a model experiment.

- Preserve published names and required arguments within a minor release. Add optional
  keyword arguments or fields; document stricter validation when it rejects formerly wrong input.
- Announce a removal with a changelog entry and `DeprecationWarning` before removing it in
  a later minor Alpha release. Stable 1.x removals require a major release.
- Data-only research records carry a schema version. Readers must reject unsupported major
  schema versions. New optional fields do not reinterpret existing values or score units.
- Serialized models record exact Python and backend versions. Cross-version pickle loading
  is not promised. Keep the original environment or refit from recorded data and parameters.
- Model environments also hash the hand-written `algorithms/` Python files, detecting
  revisions to this layer even before a version bump. Generated modules and other package
  layers still rely on the recorded package version; publish a new version when they change.
- A library being discoverable means its adapter can be attempted. `doctor` tests imports;
  numerical parity tests separately verify the specific supported configuration.

## Migration notes for the algorithm workflow

The supported base dependencies are NumPy >=1.24, pandas >=2.2.2 and SciPy >=1.11.4.
The former pandas >=2.0 / SciPy >=1.10 declarations admitted environments that failed
existing code: quarter/month-end aliases, `future_stack`, and the multivariate-t CDF.
The floor now uses pandas' NumPy-2-compatible 2.2.2 release and the final SciPy 1.11 patch.
CI pins the exact lower bounds on Python 3.10; its development extra includes `tomli`
there for TOML checks. Upgrade these dependencies when installing this revision.
See the [pandas release notes](https://pandas.pydata.org/docs/whatsnew/v2.2.2.html)
and [SciPy CDF addition](https://scipy.github.io/devdocs/release/1.11.0-notes.html).

Automatic selection now checks actual data and supplied parameters. Constant assets can route
to equal weight; HRP is excluded until `linkage` is explicit. Metadata-only recommendations
retain their previous purpose. Unknown adapter parameters fail during preflight.

The former catalog-only algorithms now have explicit, bounded adapters. Qlib uses its linear
ridge model on caller-supplied features; Heston requires calibrated parameters and explicit
dates. Neither adapter adds data download, feature generation or parameter calibration.
See [the workflow guide](RESEARCH_WORKFLOW.md) for timing and evaluation contracts.

## Release acceptance

Run the ordered repository generators and validator, default and slow suites, optional backend
parity tests, the frozen benchmark, and `python scripts/check_distribution.py`. The distribution
check builds both formats, runs `twine check --strict`, installs each into a new environment
outside the checkout, and exercises packaged skills, research and model persistence.

The active CI definition now includes Linux/Windows Python 3.10-3.13, macOS Python 3.12,
minimum base dependencies, isolated optional extras, coverage reports, slow tests, benchmarks
and packaging. Local success is not evidence those hosted jobs ran; record their run URLs
when the change is pushed. Optional backend matrix support is initially Python 3.11.

`Build release candidate` produces checked artifacts on manual dispatch or a `v*` tag.
It does not publish a PyPI project. Before public publication, select the new version,
update both version constants/changelog, ensure the tag matches, configure the repository's
PyPI trusted publisher and protected environment, and publish the exact tested artifacts.
The existing `ci/publish.yml` documents that separately configured publishing path.

## External acceptance

Use the Research workflow feedback issue template to collect independent reproduction on
a clean machine, a task attempted without maintainer intervention, unsupported data or
dependencies, and the expected/observed result. Only include data the reporter can share.
External adoption and support history must be earned through actual use; they are not
asserted by internal tests or a version number.
