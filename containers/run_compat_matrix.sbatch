#!/bin/bash
# Run the full compatibility matrix against a candidate image. tools/compat_matrix.sh runs
# its probe and build stages anywhere; submit this when the answer has to include the GPU
# tests. One node, 8 GCDs, the acceptance jobs' shape, so a pass is comparable with the
# acceptance runs. Compare counts run-to-run, never against a recorded fixed number.
#
#   sbatch containers/run_compat_matrix.sbatch [/path/to/image.sif] [baseline.txt]
# Set SBATCH_ACCOUNT before submitting.
#SBATCH --job-name=compat-matrix
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=matrix-%j.out

set -uo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO_DIR/tools/image_name.sh"
SIF="${1:-${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}/images/lumi-jax-torch-j091-latest.sif}"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

"$REPO_DIR/tools/compat_matrix.sh" "$SIF" ${2:+"$2"}
