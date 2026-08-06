#!/bin/bash
# Build the standalone, torch-free CK FMHA library: ~2700 translation units, ~32 core-hours,
# so 20-30 min on a full CPU node.
# NOT inside a %post: 2700 compiler invocations against PRoot's ~4x ptrace overhead.
#
#   sbatch containers/build_ck_fmha.sbatch
# Set SBATCH_ACCOUNT to a project with CPU billing units; a GPU-only project fails
# here with a bare AssocMaxSubmitJobLimit that looks like a queue-depth problem.
#SBATCH --job-name=ck-fmha-build
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=ckfmha-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO_DIR/tools/image_name.sh"
OUT="${OUT:-${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}/ck-fmha}"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Headroom: these TUs are a few hundred MB of RSS each, so memory binds before cores do.
export JOBS="${JOBS:-$((SLURM_CPUS_PER_TASK - 8))}"

echo "cores=$SLURM_CPUS_PER_TASK jobs=$JOBS out=$OUT"
time bash "$REPO_DIR/tools/build_ck_fmha.sh" "$OUT"
