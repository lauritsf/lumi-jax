#!/bin/bash
# Retarget any of our images onto a newer jax. VARIANT names the
# image to build *from* and *to*; the base is always the 0.9.1 line. No GPU needed; dev-g
# schedules soonest. --notest, then containers/test_jax011.sbatch.
#
#   sbatch --export=ALL,VARIANT=stage-net,JAXVER=0.11.0 containers/build_jax_version.sbatch
#
# THE DIRECTIVES BELOW ARE SIZED FOR THE LEAN VARIANTS (~6.6 GB, 6m23s). `stage-torch` is
# 15.0 GB and wants the memory and cores, not more walltime -- 22m01s with the override
# below, inside the 2 h here. Override at submit time rather than making every lean build
# queue for resources it does not use:
#
#   sbatch --export=ALL,VARIANT=stage-torch,JAXVER=0.11.0 \
#          --mem=128G --cpus-per-task=32 \
#          containers/build_jax_version.sbatch
# Set SBATCH_ACCOUNT before submitting.
#SBATCH --job-name=jax-version-build
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
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

# Everything above `mpich` carries our 2.27.7 leaf upgrade; `mpich` has the stock 2.26.6.
case "$VARIANT" in
    stage-mpich) RCCL=226 ;;
    *)     RCCL=227 ;;
esac

# Which venv holds jax, per variant: wrong here puts the new jax behind the old on sys.path.
case "$VARIANT" in
    torch|stage-torch) VENVDIR=/opt/jax-venv ;;
    *)              VENVDIR=/opt/venv ;;
esac

# `torch` carries kernels compiled against its jax; go stage-torch -> here -> kernels.
if [ "$VARIANT" = torch ]; then
    echo "FATAL: refusing to retarget 'torch' -- its kernels are already compiled." >&2
    echo "       Build stage-torch at $JAXVER with this script, then run" >&2
    echo "       sbatch --export=ALL,JAXVER=$JAXVER containers/build_jax_torch_kernels.sbatch" >&2
    exit 1
fi

BASE="$(jax_image_resolve "$VARIANT" 0.9.1)"     # retarget from the 0.9.1 line
OUT="$WORK/images/$(jax_image_name "$VARIANT" "$RCCL" "$JAXVER")"

echo "variant   $VARIANT"
echo "jax       0.9.1 -> $JAXVER"
echo "base      $BASE"
echo "venvdir   $VENVDIR"
echo "out       $OUT"
echo

# The systools proot: /appl/local/containers/proot is group-restricted.
export PATH=/appl/lumi/SW/CrayEnv/EB/systools/25.03/bin:$PATH
export SINGULARITY_CACHEDIR="$WORK/sing-cache"
export SINGULARITY_TMPDIR="$WORK/sing-tmp"
export TMPDIR="$WORK/sing-tmp"

mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" "$WORK/images"

command -v proot
singularity --version

cd "$REPO_DIR"   # %files paths are relative to the build's working directory
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
