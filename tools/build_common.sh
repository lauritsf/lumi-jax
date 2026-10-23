#!/bin/bash
# Source after resolving REPO_DIR, WORK, and IMAGES_DIR.
# The systools proot: /appl/local/containers/proot is group-restricted.
export PATH=/appl/lumi/SW/CrayEnv/EB/systools/25.03/bin:$PATH
export SINGULARITY_CACHEDIR="$WORK/sing-cache"
export SINGULARITY_TMPDIR="${BUILD_SCRATCH:-/tmp/lumi-jax-build-${SLURM_JOB_ID:-$$}}/sing-tmp"
export TMPDIR="$SINGULARITY_TMPDIR"

mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" "$IMAGES_DIR"

command -v proot
singularity --version

cd "$REPO_DIR"
