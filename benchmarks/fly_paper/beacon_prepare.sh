#!/usr/bin/env bash
set -euo pipefail
: "${FLY_PAPER_ROOT:?}" "${SLURM_JOB_ID:?}"
case "$FLY_PAPER_ROOT" in /beacon-projects/radfm/wy891/fin-skills-memory-paper-*) ;; *) exit 64;; esac
test "$(realpath "$FLY_PAPER_ROOT")" = "$FLY_PAPER_ROOT"
cd "$FLY_PAPER_ROOT"
export TMPDIR="$FLY_PAPER_ROOT/tmp"
export XDG_CACHE_HOME="$FLY_PAPER_ROOT/cache/xdg"
export MAMBA_ROOT_PREFIX="$FLY_PAPER_ROOT/runtime/mamba"
export CONDA_PKGS_DIRS="$FLY_PAPER_ROOT/cache/conda-pkgs"
export XDG_CONFIG_HOME="$FLY_PAPER_ROOT/cache/config"
export XDG_DATA_HOME="$FLY_PAPER_ROOT/cache/data"
export OCTAVE_HOME="$FLY_PAPER_ROOT/runtime/octave"
mkdir -p "$XDG_CACHE_HOME" "$MAMBA_ROOT_PREFIX" "$CONDA_PKGS_DIRS" runtime/bootstrap
curl --fail --location --retry 2 https://micro.mamba.pm/api/micromamba/linux-64/latest \
  --output runtime/micromamba.tar.bz2
tar -xjf runtime/micromamba.tar.bz2 -C runtime/bootstrap bin/micromamba
runtime/bootstrap/bin/micromamba --no-rc create --yes \
  --root-prefix "$MAMBA_ROOT_PREFIX" --prefix "$FLY_PAPER_ROOT/runtime/octave" \
  --channel conda-forge octave
runtime/bootstrap/bin/micromamba --no-rc list --explicit \
  --prefix "$FLY_PAPER_ROOT/runtime/octave" > octave-explicit.txt
runtime/octave/bin/octave --no-init-file --no-site-file --no-history --quiet --eval 'disp(version)'
sha256sum runtime/micromamba.tar.bz2 octave-explicit.txt > octave.sha256
