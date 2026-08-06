#!/bin/bash
# Build stage 1 of the full line: JAX on the LAIF full layer, no kernels.
#
#   sbatch containers/build_jax_torch_base.sbatch
#
# A CPU partition. Filesystem access follows unix group membership, not the job account.
# Set SBATCH_ACCOUNT to a project with CPU billing units; a GPU-only project fails
# here with a bare AssocMaxSubmitJobLimit that looks like a queue-depth problem.
#SBATCH --job-name=jax-torch-base
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
# Measured 5m20s; the unpack, not the %post, is what varies with Lustre.
#SBATCH --time=01:00:00
#SBATCH --output=build-base-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

source "$REPO_DIR/tools/image_name.sh"
WORK="${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}"

# The full line starts from LAIF's full layer, not our mpich chain: its own 0.9.1 entry.
BASE="$LAIF_DIR/lumi-multitorch-full-${LAIF_PINS}-${LAIF_STAMP}.sif"
OUT="$WORK/images/$(jax_image_name stage-torch 227 0.9.1)"

test -f "$BASE" || { echo "FATAL: LAIF base missing: $BASE"; exit 1; }
echo "base  $BASE"
echo "out   $OUT"

export PATH=/appl/lumi/SW/CrayEnv/EB/systools/25.03/bin:$PATH

# Build scratch in /tmp, which on LUMI compute nodes is tmpfs: the build is I/O-bound on
# metadata (63 min end to end on Lustre), so where the sandbox lives is the lever, not cores.
# /tmp counts against the job's memory -- ~54 GB here, hence the 128G above -- and is freed
# at job end. BUILD_SCRATCH=/flash/... uses NVMe: slower, but not job memory.
BUILD_SCRATCH="${BUILD_SCRATCH:-/tmp/$USER-sing-build}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}/sing-cache}"
export SINGULARITY_TMPDIR="$BUILD_SCRATCH/sing-tmp"
export TMPDIR="$BUILD_SCRATCH/sing-tmp"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" "$WORK/images"

# singularity cleans its temps up only on a *successful* build, and the rootfs is ~414k
# files: one cancelled run stranded ~20% of the /scratch inode quota. Hence both sweeps.
find "$SINGULARITY_TMPDIR" -maxdepth 1 \( -name 'build-temp-*' -o -name 'bundle-temp-*' \) \
    -mmin +120 -print -exec rm -rf {} + 2>/dev/null || true
cleanup() {
    kill "${PROGRESS_PID:-}" 2>/dev/null || true
    find "$SINGULARITY_TMPDIR" -maxdepth 1 \( -name 'build-temp-*' -o -name 'bundle-temp-*' \) \
        -exec rm -rf {} + 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# singularity prints nothing during unpack and squash; sandbox size is the only signal.
progress() {
    while true; do
        sz=$(du -sh "$SINGULARITY_TMPDIR" 2>/dev/null | cut -f1)
        echo "[progress $(date +%H:%M:%S)] sandbox=$sz elapsed=${SECONDS}s"
        sleep 30
    done
}
progress & PROGRESS_PID=$!   # cleanup() above kills this and sweeps the temps on exit

command -v proot
singularity --version
df -h "$BUILD_SCRATCH" | tail -1

cd "$REPO_DIR"   # %files paths are relative to the build's working directory
time singularity build --notest --force --build-arg BASE="$BASE" --build-arg FA_SRC="${FA_SRC_ROOT:?set it to the upstream sources checkout}" \
    "$OUT" containers/jax-torch-base.def

ls -la "$OUT"
jax_image_publish "$OUT" stage-torch 0.9.1
