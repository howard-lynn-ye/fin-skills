#!/usr/bin/env bash
set -euo pipefail
: "${FIN_STUDY_ROOT:?}" "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_TASK_ID:?}"
case "$FIN_STUDY_ROOT" in /beacon-projects/radfm/wy891/fin-skills-*) ;; *) exit 64;; esac
cd "$FIN_STUDY_ROOT"
test -f preparation.complete
test -f validation.complete
export TMPDIR="$FIN_STUDY_ROOT/tmp" HF_HOME="$FIN_STUDY_ROOT/cache/hf"
export XDG_CACHE_HOME="$FIN_STUDY_ROOT/cache/xdg" TORCH_HOME="$FIN_STUDY_ROOT/cache/torch"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export FIN_STUDY_REQUIRE_SANDBOX=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
env/bin/python -c 'import json,os,torch; print(json.dumps({"job":os.environ["SLURM_JOB_ID"],"torch":torch.__version__,"cuda":torch.version.cuda,"devices":[{"name":torch.cuda.get_device_name(i),"bytes":torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]}))' > "logs/hardware-$SLURM_JOB_ID.json"
readarray -t model_fields < <(env/bin/python -c 'import json,sys; m=json.load(open("models/models.json"))[int(sys.argv[1])]; print(m["model"]); print(m["revision"])' "$SLURM_ARRAY_TASK_ID")
exec env/bin/python source/benchmarks/agent_study/run_matrix.py \
  --model "${model_fields[0]}" --revision "${model_fields[1]}" \
  --output "$FIN_STUDY_ROOT/results/model-$SLURM_ARRAY_TASK_ID" \
  --seeds 11,23,37 --repetitions 1 --max-turns 16 --max-tokens 2048
