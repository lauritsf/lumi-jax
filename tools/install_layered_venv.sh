#!/bin/bash
# Layer a writable development venv over the selected image's /opt/venv.
set -euo pipefail
SIF="${1:?image}"
VENV_DIR="$(realpath -m "${2:?venv directory}")"
JAX_LINE="${3:?JAX line}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$(dirname "$VENV_DIR")"
BIND=( -B "$REPO_DIR:$REPO_DIR" -B "$(dirname "$VENV_DIR"):$(dirname "$VENV_DIR")" )
singularity exec "${BIND[@]}" "$SIF" bash "$REPO_DIR/tools/verify_native_rocm.sh"
singularity exec "${BIND[@]}" "$SIF" bash -s -- "$VENV_DIR" "$JAX_LINE" "$REPO_DIR" <<'EOF'
set -euo pipefail
VENV_DIR="$1"; JAX_LINE="$2"; REPO_DIR="$3"
unset PYTHONPATH
python3 -m venv "$VENV_DIR" --system-site-packages --clear
SP="$VENV_DIR/lib/python3.12/site-packages"
echo /opt/venv/lib/python3.12/site-packages > "$SP/_optvenv.pth"
source "$VENV_DIR/bin/activate"
source "$REPO_DIR/tools/resolve_image.sh"
jax_pip_pins "$JAX_LINE" > "$VENV_DIR/lumi-jax-constraints.txt"
pip install -c "$VENV_DIR/lumi-jax-constraints.txt" $(jax_pip_pins "$JAX_LINE")
pip install -c "$VENV_DIR/lumi-jax-constraints.txt" pytest 'ray[tune]' -e "$REPO_DIR"
bash "$REPO_DIR/tools/install_xla_flags_hook.sh" "$SP" "$VENV_DIR/bin/python3"
python3 - "$JAX_LINE" "$REPO_DIR" "$VENV_DIR/lumi-jax-constraints.txt" <<'PY'
import importlib.metadata as md
import pathlib
import sys
import jax
import lumi_kernels
assert jax.__version__ == sys.argv[1], (jax.__version__, sys.argv[1])
assert pathlib.Path(lumi_kernels.__file__).resolve().is_relative_to(
    pathlib.Path(sys.argv[2]).resolve()), lumi_kernels.__file__
for pin in pathlib.Path(sys.argv[3]).read_text().splitlines():
    name, version = pin.split("==")
    assert md.version(name) == version, (name, md.version(name), version)
print("layered venv pins and checkout imports OK")
PY
EOF
printf '%s\n' "$JAX_LINE" > "$VENV_DIR/lumi-jax-line"
readlink -f "$SIF" > "$VENV_DIR/lumi-jax-image"
printf '%s\n' native > "$VENV_DIR/lumi-jax-native-rocm"
