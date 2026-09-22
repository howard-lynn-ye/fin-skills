#!/usr/bin/env bash
set -euo pipefail
: "${FLY_REUSE_ROOT:?}" "${SLURM_JOB_ID:?}"
case "$FLY_REUSE_ROOT" in /beacon-projects/radfm/wy891/fin-skills-fly-reuse-*) ;; *) exit 64;; esac
test "$(realpath "$FLY_REUSE_ROOT")" = "$FLY_REUSE_ROOT"
cd "$FLY_REUSE_ROOT"
mkdir -p upstream logs tmp cache data
export TMPDIR="$FLY_REUSE_ROOT/tmp" XDG_CACHE_HOME="$FLY_REUSE_ROOT/cache"
export PIP_CACHE_DIR="$FLY_REUSE_ROOT/cache/pip" XDG_CONFIG_HOME="$FLY_REUSE_ROOT/cache/config"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export STONKFLY_DATA="$FLY_REUSE_ROOT/data/connectome"
if [ ! -d upstream/fruit-fly-fund ]; then
  git clone https://github.com/armanbabazadeh6/fruit-fly-fund.git upstream/fruit-fly-fund
  git -C upstream/fruit-fly-fund checkout --detach 56f01f6426a4d6d6293a4e06afcf4a035133a968
fi
if [ ! -d upstream/stonkfly-lab ]; then
  git clone https://github.com/paappraiser/stonkfly-lab.git upstream/stonkfly-lab
  git -C upstream/stonkfly-lab checkout --detach 09e4529e2e1a135083838abd00ccacfd95c63a29
fi
# Existing archive-based checkouts must be verified against their deployment manifest.
for repo in fruit-fly-fund stonkfly-lab; do
  test -f "upstream/$repo/LICENSE"
done
BASE_PY=/beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python
test -x env/bin/python || "$BASE_PY" -m venv "$FLY_REUSE_ROOT/env"
env/bin/python -m pip install numpy==2.2.6 pandas==2.2.3 scipy==1.15.3 pyarrow==19.0.1 Pillow==11.1.0 pytest==8.3.5
env/bin/python -m pip freeze > environment.txt
export PYTHONPATH="$FLY_REUSE_ROOT/upstream/fruit-fly-fund:$FLY_REUSE_ROOT/upstream/fruit-fly-fund/vendor:$FLY_REUSE_ROOT/upstream/stonkfly-lab"
env/bin/python -c 'from stonkfly.data import prepare,verify; prepare(); print(verify())'
cd upstream/fruit-fly-fund
STONKFLY_FULL_TEST=1 "$FLY_REUSE_ROOT/env/bin/python" -m pytest vendor/tests/test_neural.py -q
printf 'prepared\n' > "$FLY_REUSE_ROOT/PREPARED"
