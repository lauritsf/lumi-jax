#!/bin/bash
# Stage tests separately to exercise the image package, never checkout artifacts.
#SBATCH --job-name=lean-kernels-img
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:40:00
#SBATCH --output=leanimg-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO_DIR/tools/image_name.sh"
WORK="${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}"
JAXVER="${LUMI_JAX_LINE:-${JAXVER:-0.11.2}}"   # JAXVER is an accepted alias
SIF="${1:-$IMAGES_DIR/lumi-jax-lean-j${JAXVER//./}-latest.sif}"
SIF="$(readlink -f "$SIF")"
# PER-JOB, not fixed: two runs overlap normally, and a shared dir lets the first to finish
# rm -rf the tree the second is reading.
STAGE="$WORK/lean-img-tests-${SLURM_JOB_ID:-$$}"

test -f "$SIF" || { echo "FATAL: no image at $SIF"; exit 1; }

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

echo "image  $SIF"
singularity inspect "$SIF" | sed 's/^/  /' || true

# Tests and their config, deliberately NOT lumi_kernels/ or the package tree.
rm -rf "$STAGE"; mkdir -p "$STAGE"
trap 'rm -rf "$STAGE"' EXIT   # per-job dir, so cleaning up cannot disturb a sibling run
cp -r "$REPO_DIR/tests" "$STAGE/tests"
cp "$REPO_DIR/pyproject.toml" "$STAGE/"       # marker declarations live here
rm -rf "$STAGE"/tests/**/__pycache__ "$STAGE"/tests/__pycache__ 2>/dev/null || true

# --pwd and PYTHONSAFEPATH are both load-bearing: singularity keeps the caller's cwd, LUMI's
# bindings expose /projappl inside the container, and python puts the script's dir first on
# sys.path -- so without them this job imports the CHECKOUT.
run() { singularity exec --no-home --pwd "$STAGE" \
            --env PYTHONUNBUFFERED=1 --env PYTHONSAFEPATH=1 \
            -B "$STAGE:$STAGE" -B "$WORK:$WORK" "$SIF" "$@"; }

echo
echo "===== 1. which lumi_kernels, and what is in it"
run bash /opt/lumi-jax-tools/verify_native_rocm.sh
run python3 - "$JAXVER" <<'PY'
import importlib.metadata as md
import os, sys, json, pathlib, importlib.util, jax, lumi_kernels
for name in ('jax', 'jaxlib', 'jax-rocm7-plugin', 'jax-rocm7-pjrt'):
    want = sys.argv[1]
    if name in ('jax-rocm7-plugin', 'jax-rocm7-pjrt') and want == '0.9.1':
        want += '.post5'
    assert md.version(name) == want, (name, md.version(name), want)
print('JAX pins OK:', jax.__version__)
if jax.__version__ == '0.11.2':
    import _lumi_xla_flags as hook
    assert hook._needs_smi_isolation
    assert len(hook._smi_handles) == 2
    assert hook._smi_binding_mode & os.RTLD_DEEPBIND
    print('Native SMI isolation OK')
devices = jax.devices()
assert devices and all(d.platform == 'gpu' for d in devices), devices
print('GPU discovery OK:', devices)
p = os.path.dirname(lumi_kernels.__file__)
print("lumi_kernels:", p)
assert p.startswith("/opt/"), f"NOT the image's copy: {p}"
lib = os.path.join(p, "lib")
sos = sorted(f for f in os.listdir(lib) if f.endswith(".so"))
print("kernels:", sos)
assert sos == ["liblumi_fa_ffi.so"], sos
assert lumi_kernels.__version__ == "0.2.0"
for removed in ("conv1d", "selective_scan", "moe", "paged_attention", "vllm_ffi"):
    assert importlib.util.find_spec("lumi_kernels." + removed) is None, removed
from lumi_kernels.attention import flash_ffi
assert not hasattr(flash_ffi, "flash_attention_aotriton")
assert not hasattr(flash_ffi, "flash_attention_hybrid")
labels = json.loads(pathlib.Path("/.singularity.d/labels.json").read_text())
assert labels["Kernels"] == "flash", labels

PY

echo
echo "===== 2. no torch, and nothing pointing outside the image"
run bash -c '
set -e
python3 -c "import torch" 2>/dev/null && { echo "FATAL: torch is present"; exit 1; }
LIB=/opt/lumi-jax/lumi_kernels/lib
for so in $LIB/liblumi_*.so; do
    readelf -d "$so" | grep -qE "NEEDED.*(torch|c10)" && { echo "FATAL: $so links torch"; exit 1; }
    readelf -d "$so" | grep -E "R(UN)?PATH" | grep -qE "/scratch|/pfs|/projappl" \
        && { echo "FATAL: $so has an RPATH outside the image"; exit 1; }
    ldd "$so" | grep -q "not found" && { echo "FATAL: $so has unresolved deps"; ldd "$so" | grep "not found"; exit 1; }
    echo "ok  $(basename $so)"
done
exit 0'

echo
echo "===== 3. the tests that can run without torch"
cd "$STAGE"
set +e
run python3 -m pytest tests/kernels tests/attention \
    -q --no-header -p no:cacheprovider --rootdir "$STAGE" 2>&1 | tee "$STAGE/pytest.out"
rc=${PIPESTATUS[0]}
set -e

echo
echo "===== summary"
tail -3 "$STAGE/pytest.out"
# "0 passed" with a zero exit status is a pass by pytest's rules and a non-result by ours.
if ! grep -qE '[1-9][0-9]* passed' "$STAGE/pytest.out"; then
    echo "FATAL: no test actually executed -- everything was skipped or deselected."
    echo "       This job cannot say anything about the image; treat it as UNKNOWN."
    exit 1
fi
[ "$rc" = 0 ] || { echo "FATAL: pytest failed (rc=$rc)"; exit "$rc"; }
echo "LEAN KERNELS IMAGE OK -- CK attention, no torch, resolvable from inside the SIF"
