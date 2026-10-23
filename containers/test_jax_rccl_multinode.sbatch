#!/bin/bash
# Multi-node psum on the RCCL 2.27+ image, with no window-API stub and no LD_PRELOAD.
# 2 nodes x 8 GCDs, one process per GCD. The only reason to trust this image multi-node.
#
#   sbatch containers/test_jax_rccl_multinode.sbatch [/path/to/image.sif]
# Set SBATCH_ACCOUNT before submitting.
#SBATCH --job-name=jax-rccl-multinode
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH --output=multinode-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO_DIR/tools/image_name.sh"
SIF="${1:-${SIF:-${IMAGES_DIR:?set LUMI_JAX_WORK}/lumi-jax-lean-j0112-latest.sif}}"
SIF="$(readlink -f "$SIF")"
test -f "$SIF" || { echo "FATAL: image missing: $SIF"; exit 1; }
echo "image $SIF"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE VSCODE_PROXY_URI

# Deliberately no FI_* or NCCL_* tuning: this measures the image as shipped.
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
# From the job id, not a constant: two of these jobs on one node would collide on the port.
export MASTER_PORT="1${SLURM_JOB_ID: -4}"

# Records which transport aws-ofi-nccl selects; INIT,NET keeps out the per-op spam.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
# Shared scratch, not /tmp: the batch script runs on the first node only, /tmp is node-local.
WORKDIR="${IMAGES_DIR:?set LUMI_JAX_WORK}"
PYFILE="$WORKDIR/rccl_mn_$SLURM_JOB_ID.py"
cat > "$PYFILE" <<'EOF'
import os

# Pin one GCD per process BEFORE importing jax, but only when it sees more than one: on
# dev-g every task gets ROCR_VISIBLE_DEVICES=0..7, so without a pick all 8 claim all 8, while
# on standard-g picking again selects a device the task's cgroup does not expose.
_local = int(os.environ["SLURM_LOCALID"])
_vis = os.environ.get("ROCR_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES")
_devs = _vis.split(",") if _vis else None
if _devs is None:
    os.environ["ROCR_VISIBLE_DEVICES"] = str(_local)
elif len(_devs) > 1:
    os.environ["ROCR_VISIBLE_DEVICES"] = _devs[_local % len(_devs)]
# else: exactly one GCD already -- leave Slurm's choice alone.
print(f"rank {os.environ['SLURM_PROCID']}: ROCR_VISIBLE_DEVICES="
      f"{os.environ.get('ROCR_VISIBLE_DEVICES')} (was {_vis})", flush=True)

import numpy as np
import jax, jax.numpy as jnp

jax.distributed.initialize(
    coordinator_address=f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    num_processes=int(os.environ["SLURM_NTASKS"]),
    process_id=int(os.environ["SLURM_PROCID"]),
    # [0], not [SLURM_LOCALID]: the index is relative to what this process can see, already
    # one device. Unset, JAX picks by local rank -- rank 7 asks for device 7 of one.
    local_device_ids=[0],
)
assert jax.local_device_count() == 1, jax.local_devices()
pid, n = jax.process_index(), jax.process_count()
if pid == 0:
    print(f"processes: {n}  global devices: {jax.device_count()}", flush=True)
assert jax.device_count() == n, (jax.device_count(), n)

from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
mesh = Mesh(np.array(jax.devices()), ("d",))
x = jax.make_array_from_process_local_data(
    NamedSharding(mesh, P("d")), np.full((1,), float(pid), dtype=np.float32), (n,)
)
total = float(jax.jit(lambda v: jnp.sum(v), out_shardings=NamedSharding(mesh, P()))(x))
want = n * (n - 1) / 2
assert abs(total - want) < 1e-3, (total, want)
if pid == 0:
    print(f"PSUM {total} == {want}", flush=True)
    print("MULTINODE ALLREDUCE OK", flush=True)
EOF

# ALLOW_LD_PRELOAD=1 runs this against the older stub image; otherwise a set LD_PRELOAD
# means a stale stub is back in play.
NCCLLOG="$WORKDIR/ncclnet_$SLURM_JOB_ID.log"
set -o pipefail
srun --export=ALL /usr/bin/singularity exec "$SIF" bash -c "
    if [ -n \"\${LD_PRELOAD:-}\" ] && [ \"\${ALLOW_LD_PRELOAD:-0}\" != 1 ]; then
        echo \"FATAL: LD_PRELOAD is set: \$LD_PRELOAD\"; exit 1
    fi
    python3 $PYFILE
" 2>&1 | tee "$NCCLLOG"
rm -f "$PYFILE"

# A correct psum is not sufficient: without the aws-ofi plugin RCCL falls back to TCP and
# every collective still returns the right answer -- so assert the transport instead.
echo "=== transport ==="
grep -aoE "NET/OFI Selected provider is [a-z]+|NET/OFI Using transport protocol [A-Z]+|Assigned NET plugin [A-Za-z]+|Initializing aws-ofi-nccl [0-9.]+" "$NCCLLOG" | sort | uniq -c
if grep -aq "Assigned NET plugin Socket" "$NCCLLOG"; then
    echo "FATAL: RCCL fell back to NET/Socket -- the aws-ofi plugin did not load."
    grep -aoE "Could not find: [a-z-]+\.so" "$NCCLLOG" | sort -u
    exit 1
fi
grep -aq "NET/OFI" "$NCCLLOG" || { echo "FATAL: no NET/OFI in the log"; exit 1; }
grep -aq "NET/OFI Selected provider is cxi" "$NCCLLOG" || {
    echo "FATAL: NET/OFI did not select CXI"; exit 1;
}
echo "transport OK: NET/OFI with CXI"
