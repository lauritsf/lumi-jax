#!/bin/bash
# Validate the native RCCL/OFI dependencies on the lean root. No GPU needed.
# --notest, then containers/test_jax.sbatch.
#
#   sbatch containers/build_jax_rccl.sbatch          # submit from the repo root
# Set SBATCH_ACCOUNT before submitting.
#SBATCH --job-name=jax-rccl-build
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

# Native RCCL validation serves both JAX lines.
JAXVER="${LUMI_JAX_LINE:-${JAXVER:-0.9.1}}"   # JAXVER is an accepted alias
BASE="$(jax_image_resolve stage-mpich "$JAXVER")"
OUT="$IMAGES_DIR/$(jax_image_name stage-net 227 "$JAXVER")"

echo "jax   $JAXVER"
echo "base  $BASE"
echo "out   $OUT"

source "$REPO_DIR/tools/build_common.sh"
time singularity build --notest --force --build-arg BASE="$BASE" "$OUT" containers/jax-rccl.def

ls -la "$OUT"
jax_image_publish "$OUT" stage-net "$JAXVER"
