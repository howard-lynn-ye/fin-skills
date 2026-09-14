# Answer-separated evaluation

`fin_skills.api.sealed_eval` and `scripts/eval_sealed.py` implement an input boundary for
remote model evaluations. The evaluated model receives a single question, the approved
listing/context and allowed answers through a fresh API request. It has no tool executor,
workspace mount, browsing tool, conversation history or callback into the grading process.
The host controller and grader remain trusted. This is not a sandbox for arbitrary local
Python code or a coding agent running with host filesystem access.

## What is enforced

- The routing exporter copies only `q`, skill names and descriptions into the public packet.
  Labels, source notes and arbitrary context are stripped. Random question IDs do not encode
  the expected answer. The private answer key must be outside any Git checkout and the public
  export. A salted commitment binds the answer key before inference without publishing a
  guessable hash of the labels alone.
- The runner does not open the answer key. It builds API bodies from an allowlist, sends
  each question independently, and never executes model output. The endpoint and reviewed
  plain-text model snapshots are fixed. Request overrides, implicit search models, proxy
  environment variables, redirects and automatic retries are unavailable.
- Every request and original response is retained, excluding authorization headers. Missing
  infrastructure responses and tool requests invalidate the run. Malformed, refused,
  truncated or out-of-vocabulary answers count as misses, preserving the complete denominator.
- Exclusive file creation prevents overwriting outputs or retrying the same packet under
  another output directory. Failed attempts remain consumed. The receipt binds the manifest,
  requests, responses and parsed answers. The scorer requires the receipt hash saved by the
  controller **before scoring**, reconstructs answers from original responses and verifies the
  precommitted key. A partial batch or changed artifact cannot produce a valid score.
- Scoring reports aggregates without returning expected answers. Repeated scoring in the same
  output directory is refused. Every report explicitly says training contamination and
  independent answer custody were **not** verified.

The API request format follows the official [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
The initial model allowlist uses the plain-text GPT-4.1 and GPT-4.1 mini snapshots dated
2025-04-14; see the official [GPT-4.1 mini reference](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
No model/API access is required for the offline security tests. Live runs use `OPENAI_API_KEY`
from the environment and incur the provider's normal API charges.

## Routing: prepare, run, score

Example for PowerShell, run from the repository root. Supply a new run name each time and
record all attempted runs, including failures. These commands evaluate the existing public
questions as a regression test; they do **not** turn them into private questions.

```powershell
$evalPublic = Join-Path $env:TEMP 'fin-eval-public-run1'
$evalPrivate = Join-Path $env:USERPROFILE 'fin-eval-private/run1-key.json'
$evalOutput = Join-Path $env:TEMP 'fin-eval-output-run1'

python scripts/eval_sealed.py prepare --source evals/queries.jsonl --public-dir $evalPublic --private-key $evalPrivate --exposure public-fixture
python scripts/eval_sealed.py run --public-dir $evalPublic --output-dir $evalOutput --model gpt-4.1-mini-2025-04-14

# Retain the printed hash separately, then paste it here; do not recompute from receipt.json.
python scripts/eval_sealed.py score --public-dir $evalPublic --output-dir $evalOutput --private-key $evalPrivate --receipt-sha256 <saved-hash>
```

Each question is a separate API call with the full listing. A full routing run can therefore
use substantial input tokens; start with a small fixture to verify connectivity. Do not
selectively report the best of repeated runs or reset/delete attempts to hide failures.

For a real private holdout, an independent author should supply fresh JSONL containing
`{"q":"...","expect":"skill-name"}` outside the repository, use `--exposure private-holdout`,
and retain label custody. That flag records the author's declaration; it cannot prove the
questions were never published or memorized. Once scores guide development, retire that set
as validation data and use a fresh final holdout. Existing public benchmark answers cannot
be made private by renaming files or deleting them from the latest commit.

## Time-separated financial predictions

`prepare-pit` accepts JSON with `feature_names`, `allowed`, `selection_end`,
`selection_labels_end`, `holdout_start`, `observations` and `decisions`:

```json
{
  "feature_names": ["close"],
  "allowed": ["up", "down", "flat", "none"],
  "selection_end": "2026-09-01T00:00:00Z",
  "selection_labels_end": "2026-09-02T00:00:00Z",
  "holdout_start": "2026-09-03T00:00:00Z",
  "observations": [
    {"observed_at": "2026-09-03T20:00:00Z", "available_at": "2026-09-03T20:01:00Z",
     "features": {"close": 100.0}}
  ],
  "decisions": [
    {"at": "2026-09-03T20:02:00Z", "label_end_at": "2026-09-04T20:00:00Z",
     "q": "Predict the next close direction.", "expect": "up"}
  ]
}
```

This is illustrative synthetic data. Prepare with the same public/private arguments as above:

```powershell
python scripts/eval_sealed.py prepare-pit --source <trusted-source.json> --public-dir $evalPublic --private-key $evalPrivate --exposure public-fixture
```

At each decision time, the exporter admits only observations whose event **and availability**
times are no later than that decision. Only explicitly allowed numeric features are copied;
outcome/target columns are rejected. Selection and all its labels must finish strictly before
the final holdout. Labels must resolve after their decision. Timezones are mandatory.
Separate API calls ensure a later decision's history cannot leak into an earlier prediction.
This is an offline classification evaluation, not a trading engine or profitability estimate.

## Limits and custody

Hashes detect changes relative to an independently retained receipt; they are not signatures
from an independent service. A host administrator who controls the runner, all files and the
saved hash can fabricate a run, recreate a packet or read the key. Protect the controller and
store receipts/attempt records with a separate reviewer or append-only service for that threat.
Windows file creation modes do not establish ACL separation between local processes. Do not
give an untrusted local agent the private directory and call this an OS security boundary.

The trusted data author must audit free-text questions, listing descriptions, feature
construction and timestamp provenance. The exporter cannot detect an answer paraphrased in
a question, future outcomes encoded in an apparently harmless feature, a fitted model trained
on holdout data, fabricated availability times or facts already memorized in pretraining.
Continue using causal perturbation, point-in-time data, fold independence and contamination
checks. Prospective predictions frozen before outcomes occur provide stronger evidence than
historical LLM backtests. This implementation does not certify those independent requirements.

## Historical results and verification

The saved 2026-09-14 results, 92/108 and 16/16, are historical prompt-only routing scores with
unverified access isolation. They remain unchanged for traceability. `eval_blind.py` now
prints that limitation and refuses missing/invalid batches instead of reporting a partial
subset as a complete score. New protocol tests are under `tests/test_sealed_eval.py` and
`tests/test_eval_blind_protocol.py`; run them with `python -m pytest -q` and the two paths.
These adversarial fixtures test protocol enforcement, not general model honesty or skill
selection accuracy. A genuine private capability score requires a separately authored test set.

On 2026-09-14, the live protocol smoke attempt returned HTTP 429 on its first request. The
failure was retained and was not retried; no live model score is claimed. Offline tests use
explicit synthetic responses to exercise leakage and tampering attempts. API access/quota
must be available before a new, separately recorded live run can verify the provider path.
