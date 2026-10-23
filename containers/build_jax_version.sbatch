#!/bin/bash
# Retarget a stage before compiling its JAX-specific FFI library.
#SBATCH --job-name=jax-version-build
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=build-version-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

source "$REPO_DIR/tools/image_name.sh"
WORK="${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}"

VARIANT="${VARIANT:?set VARIANT, one of: $(jax_image_variants)}"
JAXVER="${LUMI_JAX_LINE:-${JAXVER:-}}"
[ -n "$JAXVER" ] || { echo "set LUMI_JAX_LINE (or the JAXVER alias), e.g. 0.11.0" >&2; exit 2; }

RCCL=227
VENVDIR=/opt/venv

case "$VARIANT" in
    stage-mpich|stage-net) ;;
    *) echo "Retarget stage-mpich or stage-net before building lean" >&2; exit 2 ;;
esac

BASE="$(jax_image_resolve "$VARIANT" 0.9.1)"     # retarget from the 0.9.1 line
OUT="$IMAGES_DIR/$(jax_image_name "$VARIANT" "$RCCL" "$JAXVER")"

echo "variant   $VARIANT"
echo "jax       0.9.1 -> $JAXVER"
echo "base      $BASE"
echo "venvdir   $VENVDIR"
echo "out       $OUT"
echo

source "$REPO_DIR/tools/build_common.sh"
time singularity build --notest --force \
    --build-arg BASE="$BASE" --build-arg JAXVER="$JAXVER" \
    --build-arg VENVDIR="$VENVDIR" \
    "$OUT" containers/jax-version.def

ls -la "$OUT"
jax_image_publish "$OUT" "$VARIANT" "$JAXVER"

echo
echo "next:"
echo "  sbatch containers/test_jax011.sbatch     # devices, GEMM, and the repo suite"
echo "  tools/probe_image.sh $OUT                # baseline it into tools/baselines/"
