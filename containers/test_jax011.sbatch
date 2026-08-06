#!/bin/bash
# Acceptance for the jax 0.11.0 image, covering two claims the suite cannot check: that all
# four of jax/jaxlib/plugin/pjrt are really 0.11.0 (a half-pinned pair still imports and
# runs), and that the hipBLASLt stub is NEVER CALLED, which is what the image rests on since
# the stub makes a workspace cap silently not apply. Hence a real workload, grepped at the
# end for the stub's warning.
#
#   sbatch containers/test_jax011.sbatch [/path/to/image.sif]
# Set SBATCH_ACCOUNT before submitting.
#SBATCH --job-name=jax011-test
# standard-g: dev-g allocates no 8-GCD full-node shape, and caps submitted jobs per user.
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
# Measured 21m53s: the GEMM epilogue sweep plus the suite, both on the 0.11.0 line.
# NO AUTOTUNE CACHE HERE, unlike the other gates: autotuning is what drives XLA to ask
# hipBLASLt for a workspace cap, so a warm cache would skip the calls this job exists to
# probe. The sweep must run cold every time.
#SBATCH --time=01:00:00
#SBATCH --output=test011-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO_DIR/tools/image_name.sh"
WORK="${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}"
SIF="${1:-$WORK/images/lumi-jax-stage-net-j0110-latest.sif}"

test -f "$SIF" || { echo "FATAL: image missing: $SIF"; \
    echo "       sbatch --export=ALL,VARIANT=stage-net,JAXVER=0.11.0 containers/build_jax_version.sbatch"; exit 1; }

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# A separate file: the job's own --output cannot be grepped from inside that job.
LOG="$WORK/images/test011-${SLURM_JOB_ID:-local}-workload.log"
: > "$LOG"

set +e
# PYTHONUNBUFFERED: piping into tee makes python block-buffer, so the gate looks hung.
srun --export=ALL,PYTHONUNBUFFERED=1 singularity exec --env PYTHONUNBUFFERED=1 "$SIF" bash -c "
    set -eux
    # The stub is a real .so in libhipblaslt's NEEDED list precisely so this stays empty.
    test -z \"\${LD_PRELOAD:-}\" || { echo 'FATAL: LD_PRELOAD is set'; exit 1; }
    echo \"LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-}\"

    python3 - <<'PY'
import importlib.metadata as md
want = '0.11.0'
for p in ('jax', 'jaxlib', 'jax-rocm7-plugin', 'jax-rocm7-pjrt'):
    got = md.version(p)
    print(f'{p:<20} {got}')
    assert got == want, f'{p} is {got}, want {want}'
print('QUADRUPLE OK')
PY

    # Re-checked on the built image: %post can gate and a later layer still shadow the stub.
    HB=\$(readlink -f /opt/rocm/lib/libhipblaslt.so.1)
    readelf -d \"\$HB\" | grep -E 'NEEDED.*lumi_hipblaslt_stub' \
        || { echo 'FATAL: stub not in libhipblaslt NEEDED'; exit 1; }
    PLUGIN=/opt/venv/lib/python3.12/site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so
    if ldd -r \"\$PLUGIN\" 2>&1 | grep 'setMaxWorkspaceBytes'; then
        echo 'FATAL: setMaxWorkspaceBytes still unresolved for the plugin'; exit 1
    fi
    echo 'STUB WIRED OK'

    python3 - <<'PY'
import numpy as np
import jax, jax.numpy as jnp
from jax import pmap

print('jax', jax.__version__)
d = jax.devices()
print('devices:', len(d))
assert len(d) == 8, d

x = jnp.ones((1024, 1024))
print('matmul:', float((x @ x).sum()))

n = len(d)
y = pmap(lambda v: jax.lax.psum(v, 'i'), axis_name='i')(jnp.arange(n, dtype=jnp.float32))
print('psum:', y)
assert np.allclose(np.asarray(y), n * (n - 1) / 2), y
print('PMAP ALLREDUCE OK')

# The shapes and epilogues most likely to make XLA ask hipBLASLt for a workspace cap. TIMED
# AND FLUSHED PER CASE: this loop can run 45+ minutes, and untimed it is one opaque stall.
import time
key = jax.random.key(0)
for dt in (jnp.float32, jnp.bfloat16, jnp.float16):
    a = jax.random.normal(key, (2048, 2048), dtype=dt)
    b = jax.random.normal(key, (2048, 1031), dtype=dt)   # odd N, off the fast tile sizes
    bias = jnp.zeros((1031,), dtype=dt)
    c = jax.random.normal(key, (8, 512, 512), dtype=dt)
    cases = (
        ('dot',    lambda: jnp.dot(a, b)),
        ('gelu',   lambda: jax.nn.gelu(jnp.dot(a, b) + bias)),
        ('relu',   lambda: jax.nn.relu(jnp.dot(a, b) + bias)),
        ('einsum', lambda: jnp.einsum('bij,bjk->bik', c, c)),
    )
    for name, fn in cases:
        t0 = time.perf_counter()
        fn().block_until_ready()
        print(f'  sweep {dt.__name__:<9} {name:<7} {time.perf_counter() - t0:8.1f} s',
              flush=True)
print('GEMM EPILOGUE SWEEP OK')
PY

    cd $REPO_DIR && python3 -m pytest tests -q
" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo
echo "===== stub tripwire"
if grep -q '\[lumi-jax\] WARNING' "$LOG"; then
    echo "FINDING: the hipBLASLt stub WAS called. The 'unreachable' assumption behind this"
    echo "image is wrong, the workspace cap was silently not applied, and the fix is a ROCm"
    echo "7.1+ base rather than a stub."
    grep '\[lumi-jax\] WARNING' "$LOG"
    rc=1
else
    echo "the stub was not called during any of the above (devices, pmap, GEMM sweep, suite)"
fi

echo
echo "workload log: $LOG"
exit "$rc"
