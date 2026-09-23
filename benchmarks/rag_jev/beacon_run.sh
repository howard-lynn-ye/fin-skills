#!/usr/bin/env bash
set -euo pipefail
root="${1:?absolute new RADFM experiment root}"
case "$root" in /beacon-projects/radfm/wy891/fin-skills-rag-jev-*) ;; *) exit 64;; esac
test "$(dirname "$root")" = /beacon-projects/radfm/wy891
test "$(realpath -e "$root")" = "$root"
test "$(realpath -e "$root/source")" = "$root/source"
: "${SLURM_JOB_ID:?submit with absolute RADFM --chdir/--output/--error}"
for part in tmp cache results; do
  test ! -L "$root/$part"
  mkdir -p "$root/$part"
done
export TMPDIR="$root/tmp" TMP="$root/tmp" TEMP="$root/tmp"
export XDG_CACHE_HOME="$root/cache" XDG_CONFIG_HOME="$root/cache/config"
export XDG_DATA_HOME="$root/cache/data" XDG_STATE_HOME="$root/cache/state"
export HF_HOME="$root/cache/hf" TORCH_HOME="$root/cache/torch"
export TRITON_CACHE_DIR="$root/cache/triton" CUDA_CACHE_PATH="$root/cache/cuda"
export NUMBA_CACHE_DIR="$root/cache/numba" MPLCONFIGDIR="$root/cache/matplotlib"
export PIP_CACHE_DIR="$root/cache/pip" PYTHONPYCACHEPREFIX="$root/cache/pycache"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$root/source"
py=/beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python
test -x "$py"
prepared="$root/results/prepared-$SLURM_JOB_ID"
result="$root/results/inference-$SLURM_JOB_ID"
"$py" -m pytest tests/test_rag_jev_evaluation.py -q \
  -o "cache_dir=$root/cache/pytest" --basetemp="$root/tmp/pytest-$SLURM_JOB_ID"
"$py" -m benchmarks.rag_jev.run prepare --dataset benchmarks/rag_jev/development.json \
  --output "$prepared" --top-k 3 --fetch-k 8 --max-context-chars 4000
status=0
"$py" -m benchmarks.rag_jev.run infer --prepared "$prepared" --output "$result" \
  --allow-network --model "${JEV_MODEL:-jev-1.13.0}" --max-requests 6 || status=$?
if test -f "$result/receipt.json"; then
  "$py" -m benchmarks.rag_jev.run score --prepared "$prepared" --run "$result" \
    --qrels benchmarks/rag_jev/development-qrels.json \
    --output "$root/results/score-$SLURM_JOB_ID.json"
fi
exit "$status"
