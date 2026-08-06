#!/bin/bash
# Acceptance for the lean JAX image: 8 GCDs, a matmul, a pmap all-reduce and the repo suite,
# with no shim dir on LD_LIBRARY_PATH and no host LD_PRELOAD (the image supplies its own).
#   sbatch containers/test_jax.sbatch [/path/to/image.sif]
# Set SBATCH_ACCOUNT before submitting.
#SBATCH --job-name=jax-image-test
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=test-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO_DIR/tools/image_name.sh"
SIF="${1:-${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}/images/lumi-jax-stage-mpich-j091-latest.sif}"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

srun /usr/bin/singularity exec "$SIF" bash -c "
    set -eux
    echo \"LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-}\"
    echo \"LD_PRELOAD=\${LD_PRELOAD:-}\"
    python3 -c '
import jax, jax.numpy as jnp
d = jax.devices(); print(\"devices:\", len(d)); assert len(d) == 8, d
x = jnp.ones((1024, 1024)); print(\"matmul:\", float((x @ x).sum()))
import numpy as np
from jax import pmap
n = len(d)
y = pmap(lambda v: jax.lax.psum(v, \"i\"), axis_name=\"i\")(jnp.arange(n, dtype=jnp.float32))
print(\"psum:\", y)
assert np.allclose(np.asarray(y), n * (n - 1) / 2), y
print(\"PMAP ALLREDUCE OK\")
'
    cd $REPO_DIR && python3 -m pytest tests -x -q
"
