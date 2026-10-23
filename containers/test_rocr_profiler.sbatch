#!/bin/bash -l
#SBATCH --job-name=rocr-profiler
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:25:00
#SBATCH --no-requeue
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE VSCODE_PROXY_URI
unset HSA_TOOLS_DISABLE_REGISTER ROCPROFILER_QUEUE_INTERPOSITION AMD_SERIALIZE_KERNEL LD_PRELOAD
export RUN=${RUN:?output directory}
export REPO="$PWD"
srun --export=ALL /usr/bin/singularity exec -B "$RUN:$RUN" "${SIF:?candidate image}" bash -c '
set -euo pipefail
cd "$RUN"
/opt/rocm/bin/hipcc --offload-arch=gfx90a "$REPO/tests/runtime/queue_stress.cpp" -o "$RUN/queue-stress"
for repeat in 1 2 3; do
 timeout --signal=TERM --kill-after=10s 300s /opt/rocm/bin/rocprofv3 --pmc SQ_WAVES -d "$RUN/counters-$repeat" -o queue --output-format csv -- "$RUN/queue-stress"
done
python3 - <<"PY"
import csv,os,pathlib
for repeat in (1,2,3):
 paths=list((pathlib.Path(os.environ["RUN"])/f"counters-{repeat}").rglob("*counter_collection.csv"))
 assert paths, "No counter capture"
 rows=[row for path in paths for row in csv.DictReader(path.open())]
 rows=[r for r in rows if "queue_tick" in r["Kernel_Name"] and r["Counter_Name"]=="SQ_WAVES"]
 assert len(rows)==10000, (repeat,len(rows))
 assert all(float(r["Counter_Value"])>0 for r in rows)
 print("COUNTER CAPTURE OK",repeat,len(rows))
PY
timeout --signal=TERM --kill-after=10s 300s python3 "$REPO/tests/runtime/profiler_capture.py" "$RUN/jax-trace"
'
echo 'ROCR PROFILER OK'
