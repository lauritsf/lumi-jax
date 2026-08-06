#!/bin/bash
# Set SBATCH_ACCOUNT before submitting.
#SBATCH --job-name=jax-suite
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --time=00:45:00
#SBATCH --output=jax-suite-%j.out

set -euo pipefail

# SLURM relocates $0 to /var/spool/slurmd, so resolve the repo root from
# SLURM_SUBMIT_DIR (where sbatch ran); fall back to $0 for non-SLURM runs.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
# The image a script RUNS must be the image the venv was BUILT in -- they layer over its
# /opt/venv -- so resolve it rather than pinning a name here.
source "$REPO_DIR/tools/resolve_image.sh"
SIF="$(jax_resolve_image lean)"
JAX_SRC="${SLURM_TMPDIR:-/tmp}/jax-src-$SLURM_JOB_ID"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

git clone --depth=1 --branch jax-v0.9.1 https://github.com/jax-ml/jax.git "$JAX_SRC"

srun /usr/bin/singularity exec "$SIF" bash -c "
    source $REPO_DIR/activate.sh
    pip install --quiet absl-py portpicker pytest pytest-xdist hypothesis cloudpickle

    cd $JAX_SRC
    pytest -v --tb=short tests/pmap_test.py \
        -k 'testBasic or testMean or testGather or testReduceScatter or testAllToAll or testTrees or testComplexPsum or testCollectivesWithTreesOfDifferentDtypes or testNestedPmap'
"
