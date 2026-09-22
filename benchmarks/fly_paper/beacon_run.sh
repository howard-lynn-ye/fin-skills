#!/usr/bin/env bash
set -euo pipefail
: "${FLY_PAPER_ROOT:?}" "${SLURM_JOB_ID:?}"
case "$FLY_PAPER_ROOT" in /beacon-projects/radfm/wy891/fin-skills-memory-paper-*) ;; *) exit 64;; esac
test "$(realpath "$FLY_PAPER_ROOT")" = "$FLY_PAPER_ROOT"
cd "$FLY_PAPER_ROOT"
export TMPDIR="$FLY_PAPER_ROOT/tmp" XDG_CACHE_HOME="$FLY_PAPER_ROOT/cache/xdg"
export XDG_CONFIG_HOME="$FLY_PAPER_ROOT/cache/config"
export XDG_DATA_HOME="$FLY_PAPER_ROOT/cache/data"
export OCTAVE_HISTFILE="$FLY_PAPER_ROOT/cache/octave-history"
export OCTAVE_HOME="$FLY_PAPER_ROOT/runtime/octave"
export MPLCONFIGDIR="$FLY_PAPER_ROOT/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
sha256sum -c octave.sha256
runtime/octave/bin/octave --no-init-file --no-site-file --no-history --quiet --eval \
  "addpath('$FLY_PAPER_ROOT/source/benchmarks/fly_paper'); run_native('$FLY_PAPER_ROOT/upstream','$FLY_PAPER_ROOT/native')"
# Native execution ends here. The separate port phase installs isolated plotting
# and spreadsheet dependencies, compares arrays, and preserves its own receipt.
