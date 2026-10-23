#!/bin/bash
# Install the selected four-package JAX stack.
set -euo pipefail
line="$1"
case "$line" in
    0.9.1) backend=0.9.1.post5 ;;
    0.11.0|0.11.2) backend="$line" ;;
    *) echo "FATAL: unsupported JAX line: $line"; exit 2 ;;
esac
. /opt/venv/bin/activate
cat > /opt/venv/lumi-jax-constraints.txt <<EOF
jax==$line
jaxlib==$line
jax-rocm7-plugin==$backend
jax-rocm7-pjrt==$backend
EOF
pip install --no-cache-dir -c /opt/venv/lumi-jax-constraints.txt \
    "jax==$line" "jaxlib==$line" \
    "jax-rocm7-plugin==$backend" "jax-rocm7-pjrt==$backend" \
    flax optax orbax-checkpoint grain chex jaxtyping pytest
if command -v mpicc >/dev/null 2>&1; then
    MPICC="$(command -v mpicc)" pip install --no-cache-dir \
        -c /opt/venv/lumi-jax-constraints.txt --no-binary=mpi4py mpi4py
    pip install --no-cache-dir -c /opt/venv/lumi-jax-constraints.txt mpi4jax
fi
python3 - "$line" "$backend" <<'CHECK'
import sys
import importlib.metadata as m
for name, want in (("jax", sys.argv[1]), ("jaxlib", sys.argv[1]),
                   ("jax-rocm7-plugin", sys.argv[2]), ("jax-rocm7-pjrt", sys.argv[2])):
    got = m.version(name)
    assert got == want, (name, got, want)
    print("pin OK:", name, got)
import flax, optax, orbax.checkpoint, grain, chex, jaxtyping
CHECK
bash /opt/lumi-jax-tools/install_xla_flags_hook.sh \
    /opt/venv/lib/python3.12/site-packages /opt/venv/bin/python3
