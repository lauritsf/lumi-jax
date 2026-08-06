#!/bin/bash
# Single-node acceptance for the torch image. It has torch, so the whole suite runs --
# including the torch-marked kernel tests that skip on the lean image -- against the image's
# own prebuilt kernels.
#
#   sbatch containers/test_jax_torch.sbatch [/path/to/image.sif]
# Set SBATCH_ACCOUNT before submitting.
#SBATCH --job-name=jax-full-test
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
# Measured 25m15s on the 0.11.0 line against 7m07s on 0.9.1; the difference is test_memory's
# 32768^2 GEMM autotune sweep, paid in full because no gate sets an autotune cache dir.
#SBATCH --time=01:00:00
#SBATCH --output=fulltest-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO_DIR/tools/image_name.sh"
SIF="${1:-${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}/images/lumi-jax-torch-j091-latest.sif}"

# Without this XLA re-autotunes every fusion on every run. Keyed to the image under test, so
# rebuilding it does not invalidate the entries -- the XLA build is what they belong to.
AUTOTUNE_CACHE="${LUMI_JAX_AUTOTUNE_CACHE:-${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}/xla-autotune}/$(basename "$SIF" .sif | grep -oE 'j(091|0110)' || echo other)"
mkdir -p "$AUTOTUNE_CACHE"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

srun /usr/bin/singularity exec --env PYTHONUNBUFFERED=1 "$SIF" bash -c "
    set -eu
    echo \"LD_PRELOAD='\${LD_PRELOAD:-}'\"
    test -z \"\${LD_PRELOAD:-}\" || { echo 'FATAL: LD_PRELOAD is set'; exit 1; }

    # TEST THE IMAGE'S KERNELS, NOT THE SUBMITTER'S: pytest runs with cwd=\$REPO_DIR and the
    # repo is installed editable, so \`import lumi_kernels\` resolves to the CHECKOUT, whose
    # lib/ is populated after any local csrc/*/build_*.sh -- so this gate would pass on an
    # image shipping no usable kernels at all. LUMI_KERNELS_LIB wins.
    export LUMI_KERNELS_LIB=/opt/lumi-jax/lumi_kernels/lib
    echo \"LUMI_KERNELS_LIB=\$LUMI_KERNELS_LIB\"

    export XLA_FLAGS=\"--xla_gpu_per_fusion_autotune_cache_dir=$AUTOTUNE_CACHE \${XLA_FLAGS:-}\"

    # Ray needs HIP_VISIBLE_DEVICES derived from Slurm's ROCR_VISIBLE_DEVICES, or `import
    # ray` raises on any GPU node.
    . $REPO_DIR/tools/ray_visible_devices.sh

    python3 -c '
import os, numpy as np
import jax, jax.numpy as jnp
d = jax.devices(); print(\"devices:\", len(d)); assert len(d) == 8, d
x = jnp.ones((1024, 1024)); print(\"matmul:\", float((x @ x).sum()))
y = jax.pmap(lambda v: jax.lax.psum(v, \"i\"), axis_name=\"i\")(jnp.arange(8, dtype=jnp.float32))
assert np.allclose(np.asarray(y), 28.0), y
print(\"PMAP ALLREDUCE OK\")

import torch; print(\"torch:\", torch.__version__, \"hip:\", torch.version.hip)
import ray; print(\"ray:\", ray.__version__)

# Assert the RESOLVED .so PATH, not the package dir: the package legitimately comes from the
# checkout, while each library must come from the image.
from lumi_kernels import _lib
for so in [\"liblumi_aotriton.so\", \"liblumi_fa_ffi.so\", \"liblumi_sscan.so\",
           \"liblumi_conv1d.so\", \"liblumi_vllm_ffi.so\"]:
    p = _lib.so_path(so)
    print(\"  \", so, \"->\", p)
    assert p.startswith(\"/opt/lumi-jax\"), (so, p)
    open(p, \"rb\").read(1)   # readable as THIS user, not just present

# Flash must be on our CK, not the torch extension -- the only check that tells them apart.
# NOTE: no apostrophes anywhere in this python block -- it is a single-quoted python3 -c
# string, so one apostrophe ends it and the rest runs as shell. bash -n cannot catch that.
import subprocess
d = subprocess.run([\"readelf\", \"-d\", _lib.so_path(\"liblumi_fa_ffi.so\")],
                   capture_output=True, text=True).stdout
assert \"libck_fmha\" in d, \"flash did not link the standalone CK\"
assert \"flash_attn_2_cuda\" not in d, \"flash still links the torch extension\"
print(\"all 5 kernels resolve to the image, readable, flash on standalone CK\")
'

    cd $REPO_DIR && python3 -m pytest tests -q
"
