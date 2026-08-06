#!/bin/bash
# Build the standalone-CK-FMHA image on the AOTriton one; needs tools/build_ck_fmha.sh
# output. No GPU needed; dev-g schedules soonest. --notest, then containers/test_jax.sbatch.
#
#   sbatch containers/build_jax_ck_fmha.sbatch          # submit from the repo root
# Set SBATCH_ACCOUNT before submitting.
#SBATCH --job-name=jax-ckfmha-build
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

# CK FMHA is prebuilt and only copied in here, so the same def serves both jax lines.
JAXVER="${LUMI_JAX_LINE:-${JAXVER:-0.9.1}}"   # JAXVER is an accepted alias
BASE="$(jax_image_resolve stage-aotriton "$JAXVER")"
OUT="$WORK/images/$(jax_image_name stage-fmha 227 "$JAXVER")"

echo "jax   $JAXVER"
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
time singularity build --notest --force --build-arg BASE="$BASE" --build-arg CK_FMHA_SRC="$WORK/ck-fmha" \
    "$OUT" containers/jax-ck-fmha.def

ls -la "$OUT"
jax_image_publish "$OUT" stage-fmha "$JAXVER"
