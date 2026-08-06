#!/bin/bash
# The gate for the four-kernel lean image (containers/jax-kernels-lean.def). The claim under
# test is "these kernels SHIP", so every check is against the kernels INSIDE the image --
# never the checkout's lumi_kernels/lib, which is populated after any local csrc/*/build_*.sh
# and would pass this job on an empty image. Hence the staging dir below and the assertion on
# lumi_kernels.__file__ before any test runs. Torch-comparison tests skip here, so the
# executed count is asserted non-zero: "green, 0 failed" is a non-result.
#
#   sbatch containers/test_lean_kernels_image.sbatch [/path/to/some.sif]
# Set SBATCH_ACCOUNT before submitting.
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
JAXVER="${LUMI_JAX_LINE:-${JAXVER:-0.9.1}}"   # JAXVER is an accepted alias
SIF="${1:-$WORK/images/lumi-jax-lean-j${JAXVER//./}-latest.sif}"
# PER-JOB, not fixed: two runs overlap normally, and a shared dir lets the first to finish
# rm -rf the tree the second is reading.
STAGE="$WORK/lean-img-tests-${SLURM_JOB_ID:-$$}"

test -f "$SIF" || { echo "FATAL: no image at $SIF"; exit 1; }

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

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
run python3 - <<'PY'
import os, lumi_kernels
p = os.path.dirname(lumi_kernels.__file__)
print("lumi_kernels:", p)
assert p.startswith("/opt/"), f"NOT the image's copy: {p}"
lib = os.path.join(p, "lib")
sos = sorted(f for f in os.listdir(lib) if f.endswith(".so"))
print("kernels:", sos)
assert len(sos) == 4, sos
assert not any("vllm" in s for s in sos), "vllm has no place on a torch-free image"
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
run python3 -m pytest tests/kernels tests/attention -m "not torch" \
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
echo "LEAN KERNELS IMAGE OK -- four kernels, no torch, resolvable from inside the SIF"
