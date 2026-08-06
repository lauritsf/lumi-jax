#!/bin/bash
# Build the standalone, torch-free causal_conv1d library.
#
#   sbatch containers/build_causal_conv1d.sbatch
#
# Same shape as build_ck_fmha.sbatch, for the same reasons: a CPU partition, outside any
# %post (PRoot's ~4x ptrace overhead). Runs INSIDE THE
# FULL IMAGE despite its torch-free output, because the build reads c10's Half/BFloat16
# headers; vendoring those two would remangle every instantiation.
# Set SBATCH_ACCOUNT to a project with CPU billing units; a GPU-only project fails
# here with a bare AssocMaxSubmitJobLimit that looks like a queue-depth problem.
#SBATCH --job-name=conv1d-build
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=conv1d-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO_DIR/tools/image_name.sh"
WORK="${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}"
OUT="${OUT:-$WORK/causal-conv1d}"
SIF="${SIF:-$WORK/images/lumi-jax-torch-j091-latest.sif}"

test -f "$SIF" || { echo "FATAL: image missing: $SIF"; exit 1; }

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

echo "image $SIF"
echo "out   $OUT"
echo

# --no-home so the build cannot pick up a stray torch or hipcc from $HOME.
time singularity exec --no-home -B "$REPO_DIR:$REPO_DIR" -B "$WORK:$WORK" "$SIF" \
    bash "$REPO_DIR/tools/build_causal_conv1d.sh" "$OUT"

echo
echo "next:"
echo "  CAUSAL_CONV1D_ROOT=$OUT bash csrc/conv1d/build_conv1d.sh   # link against it"
