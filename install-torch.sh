#!/bin/bash
# Install JAX into our `torch` image (torch + the CK flash_attn kernels), layered over
# its /opt/venv, so JAX and fused attention are importable in the SAME process.
# Additive; does not touch the main install (jax-venv / `lean` image).
# Creates ./jax-venv-torch and reuses the repo's shim/ + activate.sh's env vars.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
VENV_DIR="${VENV_DIR:-$REPO_DIR/jax-venv-torch}"
SHIM_DIR="$REPO_DIR/shim"

# Our `torch` image by default, stock LAIF `full` layer as the fallback (the image is
# absent in a fresh clone or another project), SIF to override. Ours ships all five
# kernels prebuilt, so no build step is needed on it.
source "$REPO_DIR/tools/resolve_image.sh"
JAX_LINE="$(jax_line torch)"
SIF="$(jax_resolve_image torch "$JAX_LINE")" || exit 1
test -f "$SIF" || { echo "FATAL: no such image: $SIF"; exit 1; }
mapfile -t JAX_PINS < <(jax_pip_pins "$JAX_LINE")
echo "jax line: $JAX_LINE  (${JAX_PINS[*]})"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Probed via an actual load, never `ldconfig -p`: ldconfig indexes by the SONAME inside
# the file, so a .so.1 symlink onto a .so.7-SONAME library never appears in the cache
# while loading fine.
loads_in_image() {  # loads_in_image <soname>
    singularity exec "$SIF" python3 -c "import ctypes,sys
try: ctypes.CDLL('$1'); sys.exit(0)
except OSError: sys.exit(1)" 2>/dev/null
}

mkdir -p "$SHIM_DIR"

if loads_in_image librocm_smi64.so.1; then
    echo "image resolves librocm_smi64.so.1 -- skipping shim"
    rm -f "$SHIM_DIR/librocm_smi64.so.1"
else
    [ -e "$SHIM_DIR/librocm_smi64.so.1" ] || ln -sf /opt/rocm/lib/librocm_smi64.so.7 "$SHIM_DIR/librocm_smi64.so.1"
fi

# Removed, not just skipped: activate-torch.sh puts shim/ FIRST on LD_LIBRARY_PATH, so a
# leftover copy would shadow a good in-image elfutils with Cray PE's.
if loads_in_image libdw.so.1; then
    echo "image provides libdw.so.1 -- skipping Cray PE shim"
    rm -f "$SHIM_DIR/libdw.so.1"
else
    [ -e "$SHIM_DIR/libdw.so.1" ] || cp -f /opt/cray/pe/cdst-support/2.14.6/lib/libdw.so.1 "$SHIM_DIR/libdw.so.1"
fi

if [ "$(singularity exec "$SIF" bash -c \
        'nm -D "$(ldconfig -p | awk "\$1 == \"librccl.so.1\" {print \$NF; exit}")" 2>/dev/null \
         | grep -c "ncclCommWindowRegister"')" -gt 0 ]; then
    echo "image RCCL exports ncclCommWindow* -- skipping stub (real RCCL 2.27+)"
    rm -f "$SHIM_DIR/librccl_window_stub.so"
else
    [ -e "$SHIM_DIR/librccl_window_stub.so" ] || \
        singularity exec "$SIF" gcc -shared -fPIC -o "$SHIM_DIR/librccl_window_stub.so" "$REPO_DIR/csrc/rccl_window_stub.c"
fi

singularity exec "$SIF" bash -c "
    set -e
    python3 -m venv '$VENV_DIR' --system-site-packages --clear
    # Layer the container's /opt/venv (torch, flash_attn, ...) onto this venv.
    echo /opt/venv/lib/python3.12/site-packages > '$VENV_DIR'/lib/python3.12/site-packages/_optvenv.pth
    source '$VENV_DIR'/bin/activate
    pip install --upgrade pip
    printf '%s\n' ${JAX_PINS[*]} > '$VENV_DIR/lumi-jax-constraints.txt'
    PIP=\"pip install -c $VENV_DIR/lumi-jax-constraints.txt\"
    \$PIP ${JAX_PINS[*]}
    \$PIP pytest
    # Ray Tune must be installed HERE, not only in the image: this venv records
    # \`home = /usr/bin\`, so /opt/venv's site-packages is never on its sys.path.
    # Constrained like everything else -- ray's deps are how a pinned jax moves.
    \$PIP 'ray[tune]'
    \$PIP -e '$REPO_DIR'
"

# Record what this venv was built against, so sbatch scripts that source
# resolve_image.sh pick the matching image rather than the current default.
echo "$JAX_LINE" > "$VENV_DIR/lumi-jax-line"

echo "Installed JAX-in-full at: $VENV_DIR  (jax $JAX_LINE)"
echo "Activate with: source $SCRIPT_DIR/activate-torch.sh"
