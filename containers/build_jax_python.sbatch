#!/bin/bash
# Python-only update: lean Python adapters.
# Native kernel hashes are checked by the recipe; no compiled dependency is changed.
# Set SBATCH_ACCOUNT and submit from the repo root.
#SBATCH --job-name=jax-python-build
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=build-python-%j.out
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO_DIR/tools/image_name.sh"
WORK="${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}"
VARIANT=lean
JAXVER="${LUMI_JAX_LINE:-${JAXVER:-0.11.2}}"
BASE="$(jax_image_resolve "$VARIANT" "$JAXVER")"
OUT="$IMAGES_DIR/$(jax_image_name "$VARIANT" 227 "$JAXVER")"
export PATH=/appl/lumi/SW/CrayEnv/EB/systools/25.03/bin:$PATH
export SINGULARITY_CACHEDIR="$WORK/sing-cache"
export SINGULARITY_TMPDIR="/tmp/lumi-jax-build-${SLURM_JOB_ID:-$$}/sing-tmp"
export TMPDIR="$SINGULARITY_TMPDIR"
mkdir -p "$SINGULARITY_TMPDIR" "$IMAGES_DIR"
cd "$REPO_DIR"
singularity build --notest --force --build-arg BASE="$BASE" "$OUT" containers/jax-python.def

# The recipe changes only Python files and verifies identical native kernel bytes.
# Runtime packages are inherited intact, so the parent's tuning results remain valid.
CACHE_ROOT="${LUMI_JAX_AUTOTUNE_CACHE:-$WORK/xla-autotune}"
PARENT_CACHE="$CACHE_ROOT/$(basename "$BASE" .sif)"
mkdir -p "$PARENT_CACHE"
ln -s "$(basename "$PARENT_CACHE")" "$CACHE_ROOT/$(basename "$OUT" .sif)"
jax_image_publish "$OUT" "$VARIANT" "$JAXVER"
