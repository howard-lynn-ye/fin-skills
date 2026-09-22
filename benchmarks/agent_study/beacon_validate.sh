#!/usr/bin/env bash
set -euo pipefail
: "${FIN_STUDY_ROOT:?}"
case "$FIN_STUDY_ROOT" in /beacon-projects/radfm/wy891/fin-skills-*) ;; *) exit 64;; esac
export TMPDIR="$FIN_STUDY_ROOT/tmp" XDG_CACHE_HOME="$FIN_STUDY_ROOT/cache/xdg"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$FIN_STUDY_ROOT/source"
"$FIN_STUDY_ROOT/env/bin/python" -m benchmarks.agent_study.sandbox_probe > "$FIN_STUDY_ROOT/sandbox-validation.json"
export FIN_STUDY_REQUIRE_SANDBOX=1 FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
"$FIN_STUDY_ROOT/env/bin/python" -m pytest -q tests/test_agent_study_execution.py tests/test_prediction_audit.py tests/test_api_bundle.py tests/test_api.py > "$FIN_STUDY_ROOT/logs/targeted-tests.log" 2>&1
unset FIN_STUDY_REQUIRE_SANDBOX
"$FIN_STUDY_ROOT/env/bin/python" -m pytest -q > "$FIN_STUDY_ROOT/logs/default-tests.log" 2>&1
touch "$FIN_STUDY_ROOT/validation.complete"
