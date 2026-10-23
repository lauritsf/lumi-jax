#!/bin/bash
# Build the lean JAX image (stage-mpich). No GPU needed; use a CPU partition.
# --notest, then containers/test_jax.sbatch.
#
#   sbatch containers/build_jax.sbatch          # submit from the repo root
# Set SBATCH_ACCOUNT before submitting.
#SBATCH --job-name=jax-image-build
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=build-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

source "$REPO_DIR/tools/image_name.sh"
WORK="${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}"

# Build the initial JAX line; retarget this stage before compiling kernel shims.
BASE="$LAIF_DIR/lumi-multitorch-mpich-${LAIF_PINS}-${LAIF_STAMP}.sif"
OUT="$IMAGES_DIR/$(jax_image_name stage-mpich 227 0.9.1)"

test -f "$BASE" || { echo "FATAL: LAIF base missing: $BASE"; exit 1; }
echo "base  $BASE"
echo "out   $OUT"

source "$REPO_DIR/tools/build_common.sh"
time singularity build --notest --force --build-arg BASE="$BASE" "$OUT" containers/jax.def

ls -la "$OUT"
jax_image_publish "$OUT" stage-mpich 0.9.1
