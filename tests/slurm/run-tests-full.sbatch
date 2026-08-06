#!/bin/bash
# Set SBATCH_ACCOUNT before submitting.
#SBATCH --job-name=kernel-tests
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
# Sized for a COLD autotune cache: measured 25m15s end to end on the 0.11.0 line, which
# `torch` resolves by default. A warm cache brings the same run under 10 min.
#SBATCH --time=01:00:00
#SBATCH --output=kernel-tests-%j.out

# Full pytest suite in the `torch` image (torch + flash_attn + vllm +
# causal_conv1d): every kernel binding is exercised. multigpu tests auto-skip
# on this 1-GCD allocation. Submit from the repo root.
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
# The image a script RUNS must be the image the venv was BUILT in -- they layer over its
# /opt/venv -- so resolve it rather than pinning a name here.
source "$REPO_DIR/tools/resolve_image.sh"
FULL="$(jax_resolve_image torch)"

# Without this XLA re-autotunes every fusion on every run. Split by jax line: entries are
# keyed per fusion for one XLA build, so a shared directory just misses and re-autotunes.
AUTOTUNE_CACHE="${LUMI_JAX_AUTOTUNE_CACHE:-${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}/xla-autotune}/j$(jax_line_tag "$(jax_line torch)")"
mkdir -p "$AUTOTUNE_CACHE"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

LOG="${SLURM_SUBMIT_DIR:-.}/kernel-tests-${SLURM_JOB_ID:-$$}.pytest.out"

set +e
srun /usr/bin/singularity exec "$FULL" bash -c "
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_FLAGS=\"--xla_gpu_per_fusion_autotune_cache_dir=$AUTOTUNE_CACHE \${XLA_FLAGS:-}\"
    source $REPO_DIR/activate-torch.sh
    cd $REPO_DIR
    python3 -m pytest --version >/dev/null 2>&1 || pip install --quiet pytest
    python3 -m pytest tests -q
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
echo "FULL SUITE OK"
