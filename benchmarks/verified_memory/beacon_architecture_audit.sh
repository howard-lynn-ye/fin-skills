#!/usr/bin/env bash
set -euo pipefail
: "${FIN_MEMORY_ROOT:?}" "${SLURM_JOB_ID:?}"
case "$FIN_MEMORY_ROOT" in /beacon-projects/radfm/wy891/fin-skills-memory-*) ;; *) exit 64;; esac
test "$(realpath "$FIN_MEMORY_ROOT")" = "$FIN_MEMORY_ROOT"
cd "$FIN_MEMORY_ROOT"
export TMPDIR="$FIN_MEMORY_ROOT/tmp" XDG_CACHE_HOME="$FIN_MEMORY_ROOT/cache/xdg"
export HF_HOME="$FIN_MEMORY_ROOT/cache/hf" TORCH_HOME="$FIN_MEMORY_ROOT/cache/torch"
export MPLCONFIGDIR="$FIN_MEMORY_ROOT/cache/matplotlib" NUMBA_CACHE_DIR="$FIN_MEMORY_ROOT/cache/numba"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$FIN_MEMORY_ROOT/source"
PY=/beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python
exec "$PY" source/benchmarks/verified_memory/audit_architecture.py --output "$FIN_MEMORY_ROOT/results"
