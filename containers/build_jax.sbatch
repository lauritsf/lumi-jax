#!/bin/bash
# Build the lean JAX image (stage-mpich). No GPU needed; dev-g schedules soonest.
# --notest, then containers/test_jax.sbatch.
#
#   sbatch containers/build_jax.sbatch          # submit from the repo root
# Set SBATCH_ACCOUNT before submitting.
#SBATCH --job-name=jax-image-build
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=build-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

source "$REPO_DIR/tools/image_name.sh"
WORK="${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}"

# The 0.9.1 entry point: the one version needing no hipBLASLt stub.
BASE="$LAIF_DIR/lumi-multitorch-mpich-${LAIF_PINS}-${LAIF_STAMP}.sif"
OUT="$WORK/images/$(jax_image_name stage-mpich 226 0.9.1)"

test -f "$BASE" || { echo "FATAL: LAIF base missing: $BASE"; exit 1; }
echo "base  $BASE"
echo "out   $OUT"

# The systools proot: /appl/local/containers/proot is group-restricted.
export PATH=/appl/lumi/SW/CrayEnv/EB/systools/25.03/bin:$PATH
export SINGULARITY_CACHEDIR="$WORK/sing-cache"
export SINGULARITY_TMPDIR="$WORK/sing-tmp"
export TMPDIR="$WORK/sing-tmp"

mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" "$WORK/images"

command -v proot
singularity --version

cd "$REPO_DIR"   # %files paths are relative to the build's working directory
time singularity build --notest --force --build-arg BASE="$BASE" "$OUT" containers/jax.def

ls -la "$OUT"
jax_image_publish "$OUT" stage-mpich 0.9.1
