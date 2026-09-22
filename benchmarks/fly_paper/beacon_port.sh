#!/usr/bin/env bash
set -euo pipefail
: "${FLY_PAPER_ROOT:?}" "${SLURM_JOB_ID:?}"
case "$FLY_PAPER_ROOT" in /beacon-projects/radfm/wy891/fin-skills-memory-paper-*) ;; *) exit 64;; esac
test "$(realpath "$FLY_PAPER_ROOT")" = "$FLY_PAPER_ROOT"
cd "$FLY_PAPER_ROOT"
STAGE=${FLY_PAPER_STAGE:-port}
export TMPDIR="$FLY_PAPER_ROOT/tmp" XDG_CACHE_HOME="$FLY_PAPER_ROOT/cache/xdg"
export MPLCONFIGDIR="$FLY_PAPER_ROOT/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PIP_CACHE_DIR="$FLY_PAPER_ROOT/cache/pip"
export PYTHONPATH="$FLY_PAPER_ROOT/$STAGE/source:$FLY_PAPER_ROOT/$STAGE/python-deps"
PY=/beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python
"$PY" -m pip install --disable-pip-version-check --no-input \
  --target "$FLY_PAPER_ROOT/$STAGE/python-deps" --no-deps \
  openpyxl et-xmlfile matplotlib contourpy cycler fonttools kiwisolver pillow pyparsing packaging
"$PY" -c 'import matplotlib, scipy, numpy, openpyxl; print(matplotlib.__version__, scipy.__version__, numpy.__version__, openpyxl.__version__)'
"$PY" "$STAGE/source/benchmarks/fly_paper/run_port.py" --upstream "$FLY_PAPER_ROOT/upstream" \
  --native "$FLY_PAPER_ROOT/native" --output "$FLY_PAPER_ROOT/$STAGE/results"
