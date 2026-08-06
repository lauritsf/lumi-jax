#!/bin/bash
# Build the four-kernel lean image (containers/jax-kernels-lean.def).
#
#   sbatch containers/build_jax_kernels_lean.sbatch                       # 0.9.1 line
#   sbatch --export=ALL,JAXVER=0.11.0 containers/build_jax_kernels_lean.sbatch
#
# Requires the `fmha` image for the same jax line and tools/build_causal_conv1d.sh output on
# /scratch (containers/build_causal_conv1d.sbatch regenerates it in ~25 min). JAXVER is not
# cosmetic: every FFI shim compiles against jaxlib's XLA FFI headers, so the output belongs
# to the jax line it was built on. %test needs a GPU, so --notest here and
# containers/test_lean_kernels_image.sbatch after.
# Set SBATCH_ACCOUNT to a project with CPU billing units; a GPU-only project fails
# here with a bare AssocMaxSubmitJobLimit that looks like a queue-depth problem.
#SBATCH --job-name=jax-kernels-lean
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=build-kernels-lean-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

source "$REPO_DIR/tools/image_name.sh"
WORK="${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}"

JAXVER="${LUMI_JAX_LINE:-${JAXVER:-0.9.1}}"   # JAXVER is an accepted alias
BASE="$(jax_image_resolve stage-fmha "$JAXVER")"
OUT="$WORK/images/$(jax_image_name lean 227 "$JAXVER")"

# Asserted here, not discovered in %files -- which fails minutes into the unpack.
CC="$WORK/causal-conv1d/lib/libcausal_conv1d.so"
test -f "$CC" || {
    echo "FATAL: no standalone causal_conv1d at $CC"
    echo "       run: sbatch containers/build_causal_conv1d.sbatch"
    exit 1
}

echo "jax   $JAXVER"
echo "base  $BASE"
echo "cc1d  $CC"
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
time singularity build --notest --force --build-arg BASE="$BASE" \
    --build-arg CONV1D_SRC="$WORK/causal-conv1d" \
    --build-arg FA_SRC="${FA_SRC_ROOT:?set it to the upstream sources checkout}" \
    "$OUT" containers/jax-kernels-lean.def

ls -la "$OUT"
jax_image_publish "$OUT" lean "$JAXVER"

echo
echo "next:"
echo "  sbatch containers/test_lean_kernels_image.sbatch   # the kernels, on a GPU"
echo "  tools/probe_image.sh $OUT                          # baseline it"
