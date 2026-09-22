#!/usr/bin/env bash
set -euo pipefail
: "${FIN_STUDY_ROOT:?RADFM workspace required}"
case "$FIN_STUDY_ROOT" in /beacon-projects/radfm/wy891/fin-skills-*) ;; *) exit 64;; esac
cd "$FIN_STUDY_ROOT"
mkdir -p tmp cache logs models
export TMPDIR="$FIN_STUDY_ROOT/tmp" PIP_CACHE_DIR="$FIN_STUDY_ROOT/cache/pip"
export HF_HOME="$FIN_STUDY_ROOT/cache/hf" XDG_CACHE_HOME="$FIN_STUDY_ROOT/cache/xdg"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_XET=1
/beacon-projects/radfm/envs/rfj-gpu/bin/python -m venv "$FIN_STUDY_ROOT/env"
printf '%s\n' /beacon-projects/radfm/envs/rfj-gpu/lib/python3.12/site-packages > env/lib/python3.12/site-packages/shared-torch.pth
env/bin/python -m pip install 'numpy==1.26.4' 'pandas==2.2.3' 'scipy==1.15.2' 'transformers==4.57.6' 'accelerate==1.12.0' 'pytest>=8,<9' 'statsmodels>=0.14,<0.15' 'scikit-learn>=1.3,<1.7'
env/bin/python -m pip install --no-deps -e source
env/bin/python -m pip freeze > environment.txt
env/bin/python source/benchmarks/agent_study/prepare_models.py --output models
touch preparation.complete
