#!/usr/bin/env bash
set -euo pipefail
root="${1:?absolute new RADFM output root}"
prior="${2:?absolute prior RADFM fly-reuse root with pinned upstream and snapshot}"
for path in "$root" "$prior"; do
  case "$path" in /beacon-projects/radfm/wy891/fin-skills-fly-*) ;; *) exit 64;; esac
  test "$(dirname "$path")" = /beacon-projects/radfm/wy891
  test "$(realpath -e "$path")" = "$path"
done
test "$root" != "$prior"
test "$(realpath -e "$root/source")" = "$root/source"
: "${SLURM_JOB_ID:?submit with absolute RADFM scheduler paths}"
for part in tmp cache results; do
  test ! -L "$root/$part"
  mkdir -p "$root/$part"
done
export TMPDIR="$root/tmp" TMP="$root/tmp" TEMP="$root/tmp"
export XDG_CACHE_HOME="$root/cache" XDG_CONFIG_HOME="$root/cache/config"
export XDG_DATA_HOME="$root/cache/data" XDG_STATE_HOME="$root/cache/state"
export MPLCONFIGDIR="$root/cache/matplotlib" HF_HOME="$root/cache/hf"
export TORCH_HOME="$root/cache/torch" TRITON_CACHE_DIR="$root/cache/triton"
export CUDA_CACHE_PATH="$root/cache/cuda" NUMBA_CACHE_DIR="$root/cache/numba"
export PIP_CACHE_DIR="$root/cache/pip" PYTHONPYCACHEPREFIX="$root/cache/pycache"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export FLY_REUSE_UPSTREAM="$prior/upstream"
py="$prior/env/bin/python"
test -x "$py"
test -f "$prior/data/kraken-daily.json"
test "$(git -C "$prior/upstream/fruit-fly-fund" rev-parse HEAD)" = 56f01f6426a4d6d6293a4e06afcf4a035133a968
test "$(git -C "$prior/upstream/stonkfly-lab" rev-parse HEAD)" = 09e4529e2e1a135083838abd00ccacfd95c63a29
cd "$root/source"
"$py" -m pytest tests/test_fly_gate_control.py benchmarks/fly_reuse/test_core.py -q \
  -o "cache_dir=$root/cache/pytest" --basetemp="$root/tmp/pytest-$SLURM_JOB_ID"
"$py" -m benchmarks.fly_reuse.run --upstream "$prior/upstream" \
  --data "$prior/data/kraken-daily.json" --output "$root/results/paired-$SLURM_JOB_ID" \
  --mode compact --paired-gate
