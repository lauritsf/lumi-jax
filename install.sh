#!/bin/bash
# Install JAX with ROCm GPU support on LUMI: a venv at ./jax-venv inside the LAIF
# multitorch container. Default jax 0.11.0; JAXVER=0.9.1 picks the other line.
# tools/resolve_image.sh picks line, image and pip pins together, so the venv and the
# container can never disagree about jax.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/jax-venv}"
SHIM_DIR="$SCRIPT_DIR/shim"

# Our `lean` image by default, stock LAIF mpich base as the fallback (the image is absent
# in a fresh clone or another project); SIF= or LUMI_JAX_VARIANT=stage-net overrides.
# Nothing below depends on WHICH image: each shim is probed and installed only if needed.
source "$SCRIPT_DIR/tools/resolve_image.sh"
# Resolve the line ONCE: calling jax_line again later would re-read the venv's
# recorded line, which this run is about to rewrite.
JAX_LINE="$(jax_line lean)"
JAX_VARIANT="$(jax_variant lean)"
SIF="$(jax_resolve_image lean "$JAX_LINE")"
test -f "$SIF" || { echo "FATAL: no such image: $SIF"; exit 1; }
mapfile -t JAX_PINS < <(jax_pip_pins "$JAX_LINE")
echo "jax line: $JAX_LINE  (${JAX_PINS[*]})"
echo "variant:  $JAX_VARIANT"

# The lines are not interchangeable: 0.9.2-0.11.0 reference
# hipblaslt_ext::GemmInstance::setMaxWorkspaceBytes, absent from ROCm 7.0.2 (7.1 adds
# it). Link flags decide the outcome: 0.9.2-0.10.1 (BIND_NOW) fail at load; 0.10.2+
# (LAZY) run. Our images stub the symbol as a loud no-op, which is why 0.11.0 on a
# *stock* LAIF image is unsupported.

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Shims for gaps between rocm-jax and the container:
#   librocm_smi64.so.1     plugin links the old soname; container has .so.7
#   libdw.so.1             plugin needs elfutils; Cray PE on the host has a copy
#   librccl_window_stub.so JAX 0.8+ links ncclCommWindow{Register,Deregister} (RCCL
#                          2.27+); container has 2.26.6. No-op stubs; normal
#                          collectives never call them.
# Probed via an actual load, never `ldconfig -p`: ldconfig indexes by the SONAME inside
# the file, so a .so.1 symlink onto a .so.7-SONAME library never appears in the cache
# while loading fine. Ask the loader, which is what the plugin will do.
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
    ln -sf /opt/rocm/lib/librocm_smi64.so.7 "$SHIM_DIR/librocm_smi64.so.1"
fi

# Conditional because activate.sh puts shim/ *first* on LD_LIBRARY_PATH: an
# unconditional copy would shadow a good in-image elfutils with Cray PE's.
if loads_in_image libdw.so.1; then
    echo "image provides libdw.so.1 -- skipping Cray PE shim"
    rm -f "$SHIM_DIR/libdw.so.1"
else
    cp -f /opt/cray/pe/cdst-support/2.14.6/lib/libdw.so.1 "$SHIM_DIR/libdw.so.1"
fi

# The stub no-ops an API JAX links but never calls; if JAX ever did call it, a silent
# wrong answer is the failure mode, so an image whose RCCL exports the symbols drops it.
if [ "$(singularity exec "$SIF" bash -c \
        'nm -D "$(ldconfig -p | awk "\$1 == \"librccl.so.1\" {print \$NF; exit}")" 2>/dev/null \
         | grep -c "ncclCommWindowRegister"')" -gt 0 ]; then
    echo "image RCCL exports ncclCommWindow* -- skipping stub (real RCCL 2.27+)"
    rm -f "$SHIM_DIR/librccl_window_stub.so"
else
    singularity exec "$SIF" gcc -shared -fPIC -o "$SHIM_DIR/librccl_window_stub.so" \
        "$SCRIPT_DIR/csrc/rccl_window_stub.c"
fi

singularity exec "$SIF" python3 -m venv "$VENV_DIR" --system-site-packages --clear

# A constraints file, not just pinned args: flax/orbax resolve jax upward and pip
# replaces it without complaint (`pip install -e .` can pull them).
printf '%s\n' "${JAX_PINS[@]}" > "$VENV_DIR/lumi-jax-constraints.txt"

singularity exec "$SIF" bash -c "
    set -e
    source $VENV_DIR/bin/activate
    pip install --upgrade pip
    PIP=\"pip install -c $VENV_DIR/lumi-jax-constraints.txt\"
    \$PIP ${JAX_PINS[*]}
    \$PIP pytest
    # Ray Tune must be installed HERE, not only in the image: this venv records
    # \`home = /usr/bin\`, so /opt/venv's site-packages is never on its sys.path.
    # Constrained like everything else -- ray's deps are how a pinned jax moves.
    \$PIP 'ray[tune]'
    \$PIP -e '$SCRIPT_DIR'
"

# Against what the interpreter reports, not what pip was asked for.
singularity exec "$SIF" "$VENV_DIR/bin/python3" -c "
import importlib.metadata as m, sys
want = dict(p.split('==') for p in '''${JAX_PINS[*]}'''.split())
bad = [(k, v, m.version(k)) for k, v in want.items() if m.version(k) != v]
for k, v, got in bad:
    print(f'FATAL: {k} is {got}, pinned {v}', file=sys.stderr)
sys.exit(1 if bad else 0)"

# The all-reduce workaround, into THIS venv: it records `home = /usr/bin`, so
# /opt/venv's site-packages and .pth are never on its sys.path and the image's copy is
# not there -- symptom: a first multi-GCD psum that never returns. Unconditional on
# both lines (needed from 0.10.2, measured harmless on 0.9.1; a wrong version gate is
# a silent hang). Run INSIDE the image: the venv's python3 is a symlink chain into the
# image and does not exist on the host.
singularity exec "$SIF" "$SCRIPT_DIR/tools/install_allreduce_hook.sh" \
    "$VENV_DIR/lib/python3.12/site-packages" "$VENV_DIR/bin/python3"

# Record line and variant so every sbatch that sources resolve_image.sh picks the
# image this venv was built against, not whatever the default is that week.
echo "$JAX_LINE" > "$VENV_DIR/lumi-jax-line"
echo "$JAX_VARIANT" > "$VENV_DIR/lumi-jax-variant"

echo "Installed at: $VENV_DIR  (jax $JAX_LINE, $JAX_VARIANT image)"
if [ "$JAX_VARIANT" = lean ]; then
    echo "Kernels:      aotriton, flash, sscan, conv1d -- prebuilt in the image, no build step."
    echo "              (csrc/*/build_*.sh still works and its output wins for that kernel)"
fi
echo "Activate with: source $SCRIPT_DIR/activate.sh"
