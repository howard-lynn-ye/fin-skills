#!/usr/bin/env bash
# Called by the generated submit.sh after staging an immutable source bundle.
set -euo pipefail
: "${SLURM_JOB_ID:?Run this script in a Slurm GPU allocation}"
FIN_STUDY_ROOT="${1:?Pass the absolute RADFM experiment root}"
FIN_RUNTIME_ROOT=/beacon-projects/radfm/wy891/fin-skills-audit-20260921
case "$FIN_STUDY_ROOT" in
  /beacon-projects/radfm/wy891/fin-skills-followup-*) ;;
  *) echo "Refusing an output root outside the RADFM followup directory" >&2; exit 64 ;;
esac
test "$(dirname "$FIN_STUDY_ROOT")" = /beacon-projects/radfm/wy891
test "$(realpath -e "$FIN_STUDY_ROOT")" = "$FIN_STUDY_ROOT"
cd "$FIN_STUDY_ROOT"

# Reject pre-existing symlinks before creating any application output directories.
for item in logs tmp cache results; do
  test ! -L "$FIN_STUDY_ROOT/$item"
done
mkdir -p logs tmp cache results
if find cache logs tmp results -type l -print -quit | grep -q .; then
  echo "Refusing symlinks in experiment output directories" >&2
  exit 64
fi
unset HISTFILE
export FIN_STUDY_ROOT
export TMPDIR="$FIN_STUDY_ROOT/tmp" TMP="$FIN_STUDY_ROOT/tmp" TEMP="$FIN_STUDY_ROOT/tmp"
export XDG_CACHE_HOME="$FIN_STUDY_ROOT/cache/xdg"
export XDG_CONFIG_HOME="$FIN_STUDY_ROOT/cache/config" XDG_DATA_HOME="$FIN_STUDY_ROOT/cache/data"
export HF_HOME="$FIN_STUDY_ROOT/cache/hf"
# Existing weights are reused in offline mode; new application caches use this run's root.
export HF_HUB_CACHE="$FIN_RUNTIME_ROOT/cache/hf/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$FIN_STUDY_ROOT/cache/datasets"
export HF_MODULES_CACHE="$FIN_STUDY_ROOT/cache/hf-modules"
export TORCH_HOME="$FIN_STUDY_ROOT/cache/torch" TORCHINDUCTOR_CACHE_DIR="$FIN_STUDY_ROOT/cache/inductor"
export TRITON_CACHE_DIR="$FIN_STUDY_ROOT/cache/triton" CUDA_CACHE_PATH="$FIN_STUDY_ROOT/cache/cuda"
export NUMBA_CACHE_DIR="$FIN_STUDY_ROOT/cache/numba" MPLCONFIGDIR="$FIN_STUDY_ROOT/cache/matplotlib"
export PIP_CACHE_DIR="$FIN_STUDY_ROOT/cache/pip"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export PYTHONPATH="$FIN_STUDY_ROOT/source"
PY="$FIN_RUNTIME_ROOT/env/bin/python"
test -x "$PY"
test -d "$HF_HUB_CACHE"
test ! -e "$FIN_STUDY_ROOT/results/model-14b"

"$PY" - "$FIN_STUDY_ROOT" > "logs/preflight-$SLURM_JOB_ID.json" <<'PY'
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import sys
import torch

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / "bundle-manifest.json").read_text())
if str(root) != manifest["remote_root"]:
    raise RuntimeError("bundle intended for a different experiment directory")
for name, expected in manifest["source_sha256"].items():
    path = root / "source" / name
    if not path.resolve().is_relative_to(root / "source"):
        raise RuntimeError("source resolves outside the experiment")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"source hash mismatch: {name}")
devices = [{"name": torch.cuda.get_device_name(i),
            "bytes": torch.cuda.get_device_properties(i).total_memory}
           for i in range(torch.cuda.device_count())]
if not devices or max(d["bytes"] for d in devices) < 40 * 1024 ** 3:
    raise RuntimeError("14B feasibility requires an allocated GPU with at least 40 GiB")
print(json.dumps({"job_id": os.environ["SLURM_JOB_ID"], "remote_root": str(root),
    "devices": devices, "python": sys.version,
    "packages": {p: importlib.metadata.version(p) for p in
                 ("torch", "transformers", "numpy", "pandas", "scipy")},
    "paths": {k: os.environ[k] for k in ("TMPDIR", "HF_HOME", "HF_HUB_CACHE",
              "TORCH_HOME", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH")}}, indent=2))
PY

cd "$FIN_STUDY_ROOT/source"
"$PY" -m benchmarks.agent_study.sandbox_probe > "$FIN_STUDY_ROOT/logs/sandbox-$SLURM_JOB_ID.json"
export FIN_STUDY_REQUIRE_SANDBOX=1
"$PY" -m pytest -q -o cache_dir="$FIN_STUDY_ROOT/cache/pytest" \
  tests/test_agent_study_interface.py tests/test_agent_study_execution.py \
  tests/test_agent_study_summary.py > "$FIN_STUDY_ROOT/logs/tests-$SLURM_JOB_ID.txt" 2>&1

set +e
"$PY" -m benchmarks.agent_study.run_matrix \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --revision aedcc2d42b622764e023cf882b6652e646b95671 \
  --output "$FIN_STUDY_ROOT/results/model-14b" \
  --seeds 11,23,37 --repetitions 1 --max-turns 16 --max-tokens 2048
run_status=$?
set -e
if test -f "$FIN_STUDY_ROOT/results/model-14b/protocol.json"; then
  "$PY" -m benchmarks.agent_study.summarize_matrix \
    "$FIN_STUDY_ROOT/results/model-14b" --output "$FIN_STUDY_ROOT/results/summary.json"
fi
printf '%s\n' "$run_status" > "$FIN_STUDY_ROOT/logs/runner-exit-$SLURM_JOB_ID.txt"
exit "$run_status"
