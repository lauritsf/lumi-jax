#!/bin/bash
# Build the causal_conv1d forward FFI -> liblumi_conv1d.so. Links a causal_conv1d kernel
# library and compiles against the causal-conv1d @ 0d2252d header (ConvParamsBase) plus
# header-only c10 half/bf16 types. Runs in either image; the sources it links differ.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(cd "$HERE/../.." && pwd)/lumi_kernels/lib"; mkdir -p "$OUT"
FA_SRC_ROOT="${FA_SRC_ROOT:?set it to the upstream sources checkout (flash-attention, mamba, causal-conv1d); the container builds export /opt/fa-src}"
CC_SRC="$FA_SRC_ROOT/causal-conv1d/csrc"          # causal-conv1d @ 0d2252d
XLA_INC="$(python3 -c 'import jaxlib,os; print(os.path.join(os.path.dirname(jaxlib.__file__),"include"))')"

# c10's Half/BFloat16 are needed as template storage types, never as a link dependency.
# Prefer the vendored copies (csrc/shim-c10) when torch is absent, so this builds on a lean
# image; LUMI_C10_SHIM=1 forces them where torch exists, which is how tools/verify_c10_shim.sh
# proves both paths mangle identically.
SHIM_C10="$(cd "$HERE/../shim-c10" && pwd)"
TORCH_DIR="$(python3 -c 'import torch,os; print(os.path.dirname(torch.__file__))' 2>/dev/null || true)"
if [ "${LUMI_C10_SHIM:-0}" = 1 ] || [ -z "$TORCH_DIR" ]; then
    TORCH_INC="$SHIM_C10"; echo "c10 half/bf16 headers: vendored (csrc/shim-c10)"
else
    TORCH_INC="$TORCH_DIR/include"; echo "c10 half/bf16 headers: torch ($TORCH_DIR)"
fi
# Prefer the standalone torch-free library (tools/build_causal_conv1d.sh), fall back to the
# container's PyTorch extension. Which one was used is printed: the difference is invisible
# in the built artifact's behaviour and total in its dependencies.
CAUSAL_CONV1D_ROOT="${CAUSAL_CONV1D_ROOT:-/opt/causal-conv1d}"
if [ -f "$CAUSAL_CONV1D_ROOT/lib/libcausal_conv1d.so" ]; then
    CC_SO="$CAUSAL_CONV1D_ROOT/lib/libcausal_conv1d.so"
    CC_SRC="$CAUSAL_CONV1D_ROOT/include"     # ships its own hipified ConvParamsBase header
    echo "causal_conv1d source: standalone ($CAUSAL_CONV1D_ROOT)"
else
    CC_SO="$(ls /opt/venv/lib/python3.12/site-packages/causal_conv1d_cuda*.so | head -1)"
    echo "causal_conv1d source: torch extension ($(basename "$CC_SO"))"
fi
CC_DIR="$(dirname "$CC_SO")"

hipcc -std=c++20 --offload-arch=gfx90a -fPIC -shared -O2 \
  -I"$CC_SRC" -I"$TORCH_INC" -I"$XLA_INC" \
  "$HERE/lumi_conv1d.hip" -o "$OUT/liblumi_conv1d.so" \
  -L"$CC_DIR" -l:"$(basename "$CC_SO")" -Wl,-rpath,"$CC_DIR"

echo "Built: $OUT/liblumi_conv1d.so"
nm -D "$OUT/liblumi_conv1d.so" | grep -i LumiConv1d || echo "WARN: symbol missing"

# Report the torch coupling rather than assert it away: on the fallback path libtorch in
# NEEDED is correct and expected, and only on the standalone path is its absence a claim.
echo "NEEDED:"; readelf -d "$OUT/liblumi_conv1d.so" | grep NEEDED | sed 's/^/  /'
if [ -f "$CAUSAL_CONV1D_ROOT/lib/libcausal_conv1d.so" ]; then
    readelf -d "$OUT/liblumi_conv1d.so" | grep -qE 'NEEDED.*(torch|c10)' \
        && { echo "FATAL: standalone build still links libtorch"; exit 1; }
    echo "links no libtorch: OK"
fi
