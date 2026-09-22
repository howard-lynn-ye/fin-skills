#!/usr/bin/env bash
set -euo pipefail
: "${FLY_REUSE_ROOT:?}" "${FLY_REUSE_SOURCE:?}" "${SLURM_JOB_ID:?}"
case "$FLY_REUSE_ROOT" in /beacon-projects/radfm/wy891/fin-skills-fly-reuse-*) ;; *) exit 64;; esac
test "$(realpath "$FLY_REUSE_ROOT")" = "$FLY_REUSE_ROOT"
test -f "$FLY_REUSE_ROOT/PREPARED"
cd "$FLY_REUSE_ROOT"
export TMPDIR="$FLY_REUSE_ROOT/tmp" XDG_CACHE_HOME="$FLY_REUSE_ROOT/cache"
export XDG_CONFIG_HOME="$FLY_REUSE_ROOT/cache/config" MPLCONFIGDIR="$FLY_REUSE_ROOT/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export FLY_REUSE_UPSTREAM="$FLY_REUSE_ROOT/upstream"
export PYTHONPATH="$FLY_REUSE_SOURCE:$FLY_REUSE_UPSTREAM/fruit-fly-fund:$FLY_REUSE_UPSTREAM/fruit-fly-fund/vendor:$FLY_REUSE_UPSTREAM/stonkfly-lab"
export STONKFLY_DATA="$FLY_REUSE_ROOT/data/connectome"
PY="$FLY_REUSE_ROOT/env/bin/python"
MODE="${1:?compact or full}"
"$PY" -m pytest "$FLY_REUSE_SOURCE/benchmarks/fly_reuse/test_core.py" -q
"$PY" "$FLY_REUSE_SOURCE/benchmarks/fly_reuse/run.py" \
  --upstream "$FLY_REUSE_UPSTREAM" --data "$FLY_REUSE_ROOT/data/kraken-daily.json" \
  --output "$FLY_REUSE_ROOT/results-$MODE-v1" --mode "$MODE" \
  --connectome "$STONKFLY_DATA" --pilot-bars 180
