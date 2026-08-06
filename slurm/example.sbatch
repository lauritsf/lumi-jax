#!/bin/bash
# Set SBATCH_ACCOUNT before submitting.
#SBATCH --job-name=jax
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=jax-%j.out

set -euo pipefail

# SLURM relocates $0 to /var/spool/slurmd, so resolve the repo root from
# SLURM_SUBMIT_DIR (where sbatch ran); fall back to $0 for non-SLURM runs.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# The image a script RUNS must be the image the venv was BUILT in -- they layer over its
# /opt/venv -- so resolve it rather than pinning a name here.
source "$REPO_DIR/tools/resolve_image.sh"
SIF="$(jax_resolve_image lean)"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

srun /usr/bin/singularity exec "$SIF" bash -c "
    source $REPO_DIR/activate.sh
    # Replace with your own script. Add `source $REPO_DIR/tools/ray_visible_devices.sh`
    # before it if you import ray: without it `import ray` raises on a GPU node.
    python3 -c 'import jax; print(jax.devices())'
"
