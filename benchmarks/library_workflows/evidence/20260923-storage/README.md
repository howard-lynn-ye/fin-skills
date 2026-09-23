# Storage crash-recovery evidence

`manifest.json` binds the code snapshot and Slurm plan; `submission-receipt.json` and
`completion.json` distinguish launch from completion. `cases/` records the exact process
barrier, SIGKILL exit status and recovered database state. `recovery-verified.json` is a
separate read-only SQLite replay of all twelve databases still stored in the RADFM root
named in the manifest. `revisions.json` retains arrival-order behavior, including the
older final revision. `scales.json` contains all timing repetitions, not just medians.

`SHA256.json` hashes the retained machine-readable artifacts. These fixtures are synthetic
engineering tests; they do not measure financial prediction, power-loss safety, competing
database performance or human efficiency. Source: `benchmarks/library_workflows/storage_recovery.py`.
