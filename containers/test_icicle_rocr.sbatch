#!/bin/bash -l
#SBATCH --job-name=icicle-rocr
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:35:00
#SBATCH --no-requeue
set -euo pipefail
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE VSCODE_PROXY_URI
source "${ICICLE_ENV_FILE:-/projappl/project_465003095/icicle/tools/lumi/lumi_env.sh}"
SIF=${SIF:?candidate image}
RUN=${RUN:?output directory}
BATCH=${BATCH:?1 or 32}
export PROBE_ROOT=/scratch/project_465003095/icicle/camera-ready-eval-20261021/analysis/native-debug/alternatives
export PYTHONPATH=/scratch/project_465003095/icicle/camera-ready-eval-20261021/source-v1
export PYTHONDONTWRITEBYTECODE=1
export JAX_COMPILATION_CACHE_DIR="$RUN/icicle-cache"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 XLA_PYTHON_CLIENT_PREALLOCATE=false
unset AMD_SERIALIZE_KERNEL HSA_TOOLS_DISABLE_REGISTER ROCPROFILER_QUEUE_INTERPOSITION LD_PRELOAD
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
export RUN BATCH
srun --export=ALL /usr/bin/singularity exec --bind "$ICICLE_SINGULARITY_BIND,$RUN:$RUN" "$SIF" bash -c '
set -euo pipefail
export VENV_DIR="$ICICLE_VENV"
source "$ICICLE_LUMI_JAX/activate.sh"
export PYTHONPATH=/opt/lumi-jax:/scratch/project_465003095/icicle/camera-ready-eval-20261021/source-v1
export PYTHONSAFEPATH=1
ulimit -c 0
python - <<"PY"
import hashlib,json,pathlib,lumi_kernels
assert pathlib.Path(lumi_kernels.__file__).is_relative_to("/opt/lumi-jax"), lumi_kernels.__file__
print("Candidate adapters:",lumi_kernels.__version__,lumi_kernels.__file__,flush=True)
m=json.loads(pathlib.Path("/opt/lumi-jax-rocr/manifest.json").read_text())
f=pathlib.Path("/opt/rocm/lib/libhsa-runtime64.so.1").resolve()
assert hashlib.sha256(f.read_bytes()).hexdigest()==m["runtime_sha256"]
print("Candidate runtime verified:",f,flush=True)
PY
for repetition in 1 2 3; do
 export PROBE_RESULT="$RUN/icicle-b${BATCH}-${SLURM_JOB_ID}-r${repetition}.npz"
 echo "BEGIN batch=$BATCH repeat=$repetition $(date -Is)"
 timeout --signal=TERM --kill-after=10s 600s python -u "$PROBE_ROOT/probe-runtime.py" --weights /scratch/project_465003095/icicle/ftpfn-adaptive-ablation-20261007/raw/published-ftpfn.safetensors --data-dir "$ICICLE_DATA" --batch "$BATCH" --iters 1000 --ctx-buckets 1
 test -s "$PROBE_RESULT"
 echo "END batch=$BATCH repeat=$repetition $(date -Is)"
done
'
echo 'ICICLE REPEATS OK'
