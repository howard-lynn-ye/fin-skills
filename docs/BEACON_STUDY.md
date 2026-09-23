# Running the agent study on Beacon

## Connection route verified on 2026-09-23 UTC

Use the existing connection guide at `D:/ct_agent_vqa/BEACON_CONNECTION_GUIDE.md`.
Its credential-isolated helper is `D:/ct_agent_vqa/scripts/beacon_ssh_session.py`;
it verifies the pinned Beacon host key and uses the existing local password file.
The SSH key `id_ed25519_beacon_gnwt` was rejected in this session, while the documented
password-file route authenticated and executed remote commands successfully. A failed
key-only attempt does not establish that Beacon is unreachable or that the password
must be reset. Do not print credential contents or copy them into this repository.

The helper initially enters the medical project's directory. For fin-skills commands,
explicitly change to `/beacon-projects/radfm/wy891` and use a fin-skills run directory.
Do not change the helper's project defaults or touch another project's jobs.
The current parallel batch is recorded in
[the campaign note](../paper/BEACON_CAMPAIGN_20260923.md).

The current workspace is `/beacon-projects/radfm/wy891/fin-skills-audit-20260921`.
Source, the virtual environment, temporary files, model caches, Slurm logs and results
are all stored there. Do not stage this experiment under the Beacon home directory.
The existing RADFM `rfj-gpu` runtime supplies read-only PyTorch dependencies.

The three scripts have separate responsibilities:

1. `benchmarks/agent_study/beacon_prepare.sh` creates the environment, freezes its
   installed packages and downloads model snapshots at recorded Hugging Face revisions.
2. `beacon_validate.sh` probes Linux confinement, runs the targeted worker tests under
   confinement, and then runs the default test suite. Only success writes its sentinel.
3. `beacon_run.sh` requires both completion sentinels. Each Slurm array task loads one
   model and runs the four conditions on paired public development seeds.

Submit preparation and validation as CPU jobs. Submit inference as a dependent GPU
array using `--dependency=afterok:<prepare-job>:<validation-job>`. Preparation and validation
have now completed. The current allocation uses one RTX 6000 Ada for each of 7B and 14B,
and two for 32B; job-specific hardware metadata is saved in `logs/hardware-*.json`.
These jobs use the account's `medium` QoS (at most two GPUs and 64 GiB per job),
with four CPU cores and 64 GiB host memory. Inspect permitted QoS limits before submitting.
Do not modify the remote source while inference is running.
Each model's `protocol.json` freezes the model revision, budgets, randomized cell order
and Python source hashes before inference. Keep the environment manifest with results.

`scripts/beacon_workspace.py` provides pinned-host source sync and command transport.
Its `--credential-file` argument refers to a local file; the credential is never copied
to Beacon or printed. Supply its path locally rather than putting a password in commands.
Sync includes selected repository source paths and excludes local backups and secrets.

Each cell retains the provider responses, tool feedback, execution receipts, usage,
submission hash and independent grade or grading error. Grades are generated after
submission freezing and never returned as repair feedback. Failed or missing cells stay
in the analysis denominator; do not delete them to improve a summary. A partially
created cell is deliberately not retried automatically.

The initial two-GPU 32B attempt failed at the inference runtime. Single-H200 recovery
job `1613767` is queued separately; its records must remain distinct from that attempt.
After that job finishes, run:

```bash
env/bin/python source/benchmarks/agent_study/summarize_matrix.py \
  results/model-0 results/model-1 results-single-gpu/model-2 --output results/summary.json
```

Review accepted-but-ungradable cells and grading errors before interpreting rates.
The first study uses public generator code and public development seeds. It is a
feasibility experiment, not an unseen benchmark. Landlock/seccomp constrain host access;
the in-process checks are not proven resistant to adversarial Python monkeypatching.
See `paper/STUDY_PLAN.md` for the confirmatory study and real-data requirements.
