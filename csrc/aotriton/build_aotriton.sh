#!/bin/bash
# Build the AOTriton forward FFI -> liblumi_aotriton.so.
#
# Two sources for AOTriton, tried in this order:
#
#   1. AOTRITON_ROOT -- a standalone AOTriton install (what the stage-aotriton image ships).
#      Preferred: the ABI is then a pin we choose rather than whatever PyTorch bundles, and
#      since this binding uses torch for nothing else the target builds on an image with no
#      torch present at all.
#   2. torch's bundled copy in torch/lib, the only source on a stock LAIF full image.
#
# Both are kept so a checkout on either image builds unchanged.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(cd "$HERE/../.." && pwd)/lumi_kernels/lib"; mkdir -p "$OUT"

# /opt/aotriton is where containers/jax-aotriton.def installs it.
AOTRITON_ROOT="${AOTRITON_ROOT:-/opt/aotriton}"

if [ -n "$(ls "$AOTRITON_ROOT"/lib/libaotriton_v2.so* 2>/dev/null | head -1)" ]; then
    AOT_DIR="$AOTRITON_ROOT/lib"
    AOT_INC="$AOTRITON_ROOT/include"
    SOURCE="standalone ($AOTRITON_ROOT)"
else
    TORCH="$(python3 -c 'import torch,os; print(os.path.dirname(torch.__file__))' 2>/dev/null || true)"
    [ -n "$TORCH" ] || TORCH=/opt/venv/lib/python3.12/site-packages/torch
    AOT_DIR="$TORCH/lib"
    AOT_INC="$TORCH/include"
    SOURCE="torch ($TORCH)"
fi

AOT_SO="$(ls "$AOT_DIR"/libaotriton_v2.so* 2>/dev/null | head -1)"
[ -n "$AOT_SO" ] || { echo "no libaotriton_v2.so under $AOT_DIR" >&2; exit 1; }
XLA_INC="$(python3 -c 'import jaxlib,os; print(os.path.join(os.path.dirname(jaxlib.__file__),"include"))')"

echo "AOTriton source: $SOURCE"
echo "AOT_INC=$AOT_INC"; echo "AOT_SO=$AOT_SO"

hipcc -std=c++20 --offload-arch=gfx90a -fPIC -shared -O2 \
  -I"$AOT_INC" -I"$XLA_INC" \
  "$HERE/lumi_aotriton_ffi.hip" \
  -o "$OUT/liblumi_aotriton.so" \
  -L"$AOT_DIR" -l:"$(basename "$AOT_SO")" -Wl,-rpath,"$AOT_DIR"

echo "Built: $OUT/liblumi_aotriton.so"
nm -D "$OUT/liblumi_aotriton.so" | grep -i LumiAotFwd || echo "WARN: symbol missing"
