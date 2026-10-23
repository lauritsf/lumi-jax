#!/bin/bash
#SBATCH --job-name=jax-attention-build
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=build-attention-%j.out
# Set BASE to rebuild from a compatible immutable product; otherwise use stage-net.
# Output is not published until its acceptance and application checks pass.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO_DIR/tools/image_name.sh"
WORK="${LUMI_JAX_WORK:?set LUMI_JAX_WORK}"
JAXVER="${LUMI_JAX_LINE:-${JAXVER:-0.11.2}}"
BASE="${BASE:-$(jax_image_resolve stage-net "$JAXVER")}"
OUT="${OUT:-$IMAGES_DIR/$(jax_image_name lean 227 "$JAXVER")}"
CK_FMHA_SRC="${CK_FMHA_SRC:-$BUILD_ARTIFACT_DIR/ck-fmha}"
test -f "$CK_FMHA_SRC/lib/libck_fmha.so"
test ! -e "$OUT"
source "$REPO_DIR/tools/build_common.sh"
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT
singularity build --notest --build-arg BASE="$BASE" --build-arg CK_FMHA_SRC="$CK_FMHA_SRC" \
    "$OUT" containers/jax-kernels-lean.def
_jax_sif_align_partition "$OUT"
singularity exec "$OUT" true
sha256sum "$OUT"
echo "ATTENTION IMAGE BUILD OK: $OUT"
