#!/bin/bash
# Build stage 2 of the full line: our five kernel targets into the stage-1 image, ~6 min.
# Do not reintroduce a persistent /flash sandbox with `build --update` to skip the unpack:
# measured 25m21s against 6m16s here, because PRoot's per-syscall ptrace overhead against
# Lustre dwarfs an unpack that tmpfs already cut to under a minute. A CPU partition, as in
# stage 1.
#
#   sbatch containers/build_jax_torch_kernels.sbatch
# Set SBATCH_ACCOUNT to a project with CPU billing units; a GPU-only project fails
# here with a bare AssocMaxSubmitJobLimit that looks like a queue-depth problem.
#SBATCH --job-name=jax-full-kernels
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
# Measured 6m20s-7m04s across four runs.
#SBATCH --time=01:00:00
#SBATCH --output=build-kernels-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

source "$REPO_DIR/tools/image_name.sh"
WORK="${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}"

# NOT jax-agnostic: the five FFI targets compile against the base's jax headers, so the
# 0.11.0 cell must be built from a 0.11.0 torch-base, never retargeted afterwards.
JAXVER="${LUMI_JAX_LINE:-${JAXVER:-0.9.1}}"   # JAXVER is an accepted alias
BASE="$(jax_image_resolve stage-torch "$JAXVER")"
OUT="$WORK/images/$(jax_image_name torch 227 "$JAXVER")"

echo "jax   $JAXVER"
echo "base  $BASE"
echo "out   $OUT"

export PATH=/appl/lumi/SW/CrayEnv/EB/systools/25.03/bin:$PATH
# Build scratch in /tmp, i.e. tmpfs: the build is I/O-bound, not CPU-bound. /tmp counts
# against the job's memory (~54 GB here, covered by 128G) and is freed at job end.
BUILD_SCRATCH="${BUILD_SCRATCH:-/tmp/$USER-sing-build}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}/sing-cache}"
export SINGULARITY_TMPDIR="$BUILD_SCRATCH/sing-tmp"
export TMPDIR="$BUILD_SCRATCH/sing-tmp"

mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" "$WORK/images"

# singularity removes its build temps only on success, and the rootfs is ~414k files (~20%
# of the /scratch inode quota); load-bearing when BUILD_SCRATCH is persistent.
find "$SINGULARITY_TMPDIR" -maxdepth 1 \( -name 'build-temp-*' -o -name 'bundle-temp-*' \) \
    -mmin +120 -print -exec rm -rf {} + 2>/dev/null || true
cleanup() {
    kill "${PP:-}" 2>/dev/null || true
    find "$SINGULARITY_TMPDIR" -maxdepth 1 \( -name 'build-temp-*' -o -name 'bundle-temp-*' \) \
        -exec rm -rf {} + 2>/dev/null || true
}
trap cleanup EXIT INT TERM

test -f "$BASE" || { echo "FATAL: $BASE missing -- run build_jax_torch_base.sbatch first"; exit 1; }

# Progress: singularity is silent through unpack and squash, so report growth every 30 s.
progress() {
    local watch="$1"
    while true; do
        echo "[progress $(date +%H:%M:%S)] $(du -sh "$watch" 2>/dev/null | cut -f1) elapsed=${SECONDS}s"
        sleep 30
    done
}

cd "$REPO_DIR"

progress "$SINGULARITY_TMPDIR" & PP=$!
time singularity build --notest --force --build-arg BASE="$BASE" --build-arg CK_FMHA_SRC="$WORK/ck-fmha" \
    "$OUT" containers/jax-torch-kernels.def
kill $PP 2>/dev/null || true
ls -la "$OUT"
jax_image_publish "$OUT" torch "$JAXVER"
