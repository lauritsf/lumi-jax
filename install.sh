#!/bin/bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/tools/resolve_image.sh"
KIND=lean
VENV_DIR="$(_jax_venv_dir "$KIND")"
JAX_LINE="$(jax_line "$KIND")"
JAX_VARIANT="$(jax_variant "$KIND")"
SIF="$(jax_resolve_image "$KIND" "$JAX_LINE" install)"
test -f "$SIF" || { echo "FATAL: no such image: $SIF" >&2; exit 1; }
echo "jax line: $JAX_LINE; variant: $JAX_VARIANT"
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
bash "$REPO_DIR/tools/install_layered_venv.sh" "$SIF" "$VENV_DIR" "$JAX_LINE"
echo "$JAX_VARIANT" > "$VENV_DIR/lumi-jax-variant"
echo "Installed at: $VENV_DIR (JAX $JAX_LINE, $JAX_VARIANT)"
