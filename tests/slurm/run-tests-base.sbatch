#!/bin/bash
# Set SBATCH_ACCOUNT before submitting.
#SBATCH --job-name=jax-tests
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
# Sized for a COLD autotune cache: test_memory's 32768^2 GEMM costs ~23 min on the 0.11.0
#SBATCH --time=01:00:00
#SBATCH --output=jax-tests-%j.out

# tests auto-skip via conftest; the -m filter avoids collecting them at all.
# Submit from the repo root.
set -euo pipefail

# SLURM relocates $0 to /var/spool/slurmd, so resolve the repo root from
# SLURM_SUBMIT_DIR (where sbatch ran); fall back to $0 for non-SLURM runs.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
# The image a script RUNS must be the image the venv was BUILT in -- they layer over its
# /opt/venv -- so resolve it rather than pinning a name here.
source "$REPO_DIR/tools/resolve_image.sh"
SIF="$(jax_resolve_image lean)"

# Keep runtime libraries and JAX builds separate by using the immutable image name.
AUTOTUNE_CACHE="${LUMI_JAX_AUTOTUNE_CACHE:-${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}/xla-autotune}/$(basename "$(readlink -f "$SIF")" .sif)"
mkdir -p "$AUTOTUNE_CACHE"
AUTOTUNE_CACHE="$(readlink -f "$AUTOTUNE_CACHE")"
VENV_DIR="$(readlink -f "$(_jax_venv_dir lean)")"
test -f "$VENV_DIR/bin/activate"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# srun refuses inherited mem env from the submitting shell
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

LOG_DIR="${LUMI_JAX_TEST_LOG_DIR:-$LUMI_JAX_WORK/test-logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/jax-tests-${SLURM_JOB_ID:-$$}.pytest.out"

set +e
srun /usr/bin/singularity exec \
    -B "$AUTOTUNE_CACHE:$AUTOTUNE_CACHE" -B "$VENV_DIR:$VENV_DIR" \
    --env VENV_DIR="$VENV_DIR" "$SIF" bash -c "
    set -e
    export XLA_FLAGS=\"--xla_gpu_per_fusion_autotune_cache_dir=$AUTOTUNE_CACHE \${XLA_FLAGS:-}\"
    source $REPO_DIR/activate.sh
    export LUMI_KERNELS_LIB=/opt/lumi-jax/lumi_kernels/lib
    cd $REPO_DIR
    PY=$VENV_DIR/bin/python3
    \$PY - <<'PYGPU'
import jax
from lumi_kernels.attention import flash_ffi
devices = jax.devices()
assert len(devices) == 8 and all(d.platform == 'gpu' for d in devices), devices
assert flash_ffi.is_available(), 'CK attention library unavailable'
print('BASE PREREQUISITES OK', devices)
PYGPU
    \$PY -m pytest tests -q
" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo
echo "===== summary"
# Every marker auto-skips on a missing prerequisite, so a wrong image skips the whole suite
# and still exits 0. "0 passed" is a pass by pytest's rules and a non-result by ours.
if ! grep -qE '[1-9][0-9]* passed' "$LOG"; then
    echo "FATAL: no test actually executed -- everything was skipped or deselected."
    echo "       This job cannot say anything about the image; treat it as UNKNOWN."
    exit 1
fi
[ "$rc" = 0 ] || { echo "FATAL: pytest failed (rc=$rc)"; exit "$rc"; }
echo "BASE SUITE OK"
