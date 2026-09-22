#!/usr/bin/env bash
set -euo pipefail
: "${FIN_UTILITY_ROOT:?}" "${SLURM_JOB_ID:?}"
case "$FIN_UTILITY_ROOT" in /beacon-projects/radfm/wy891/fin-skills-*) ;; *) exit 64;; esac
cd "$FIN_UTILITY_ROOT"
OLD=/beacon-projects/radfm/wy891/fin-skills-audit-20260921
export TMPDIR="$FIN_UTILITY_ROOT/tmp" XDG_CACHE_HOME="$FIN_UTILITY_ROOT/cache/xdg"
export HF_HOME="$OLD/cache/hf" TORCH_HOME="$FIN_UTILITY_ROOT/cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$FIN_UTILITY_ROOT/source"
PY="$OLD/env/bin/python"
"$PY" -m pytest -q source/tests/test_library_utility.py > "logs/tests-$SLURM_JOB_ID.txt"
"$PY" source/benchmarks/library_utility/run.py --output "$FIN_UTILITY_ROOT/controls" --controls-only
"$PY" -c 'import torch,json; print(json.dumps({"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0)}))'
exec "$PY" source/benchmarks/library_utility/run.py --output "$FIN_UTILITY_ROOT/results" \
    --models "$OLD/models/models.json" --model-indices 0,1 --seeds 11,23 --sessions 84
