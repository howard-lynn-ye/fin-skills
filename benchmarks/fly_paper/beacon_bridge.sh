#!/usr/bin/env bash
set -euo pipefail
: "${FLY_PAPER_ROOT:?}" "${SLURM_JOB_ID:?}"
case "$FLY_PAPER_ROOT" in /beacon-projects/radfm/wy891/fin-skills-memory-paper-*) ;; *) exit 64;; esac
test "$(realpath "$FLY_PAPER_ROOT")" = "$FLY_PAPER_ROOT"
cd "$FLY_PAPER_ROOT"
STAGE=${FLY_PAPER_STAGE:-bridge}
REPRODUCTION_STAGE=${FLY_REPRODUCTION_STAGE:-port}
export TMPDIR="$FLY_PAPER_ROOT/tmp" XDG_CACHE_HOME="$FLY_PAPER_ROOT/cache/xdg"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$FLY_PAPER_ROOT/$STAGE/source"
PY=/beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python
"$PY" "$STAGE/source/benchmarks/fly_paper/run_bridge.py" \
  --parameters "$FLY_PAPER_ROOT/upstream/data_and_parameters/Dx_steady_state_nonlinear_3_27-Mar-2023_3modules.mat" \
  --reproduction "$FLY_PAPER_ROOT/$REPRODUCTION_STAGE/results/results.json" --output "$FLY_PAPER_ROOT/$STAGE/results"
