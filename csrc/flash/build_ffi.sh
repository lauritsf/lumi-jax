#!/bin/bash
# Build the XLA-FFI forward custom call (lumi_fa_ffi.hip) -> liblumi_fa_ffi.so.
#
# Two sources for the compiled CK kernels, tried in this order:
#
#   1. CK_FMHA_ROOT -- a standalone libck_fmha.so (built by tools/build_ck_fmha.sh, shipped
#      in the stage-fmha image and above). Preferred: it links no libtorch, so the flash
#      target builds and runs on an image with no torch in it.
#   2. flash_attn_2_cuda*.so from the container's venv -- a PyTorch extension, and the only
#      source on a stock LAIF full image.
#
# Both are kept so a checkout on either image builds unchanged.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(cd "$HERE/../.." && pwd)/lumi_kernels/lib"; mkdir -p "$OUT"
FA_SRC_ROOT="${FA_SRC_ROOT:?set it to the upstream sources checkout (flash-attention, mamba, causal-conv1d); the container builds export /opt/fa-src}"
FA_SRC="$FA_SRC_ROOT/flash-attention"             # ROCm/flash-attention @ bbe25ba
CK="$FA_SRC/csrc/composable_kernel"
FMHA="$CK/example/ck_tile/01_fmha"

# /opt/ck-fmha is where containers/jax-ck-fmha.def installs it.
CK_FMHA_ROOT="${CK_FMHA_ROOT:-/opt/ck-fmha}"

# XLA FFI headers ship inside jaxlib (header-only API).
XLA_INC="$(python3 -c 'import jaxlib,os; print(os.path.join(os.path.dirname(jaxlib.__file__),"include"))')"

if [ -e "$CK_FMHA_ROOT/lib/libck_fmha.so" ]; then
    KERNEL_DIR="$CK_FMHA_ROOT/lib"; KERNEL_LIB="libck_fmha.so"
    SOURCE="standalone ($CK_FMHA_ROOT)"
    # The standalone install ships its own headers, so no flash-attention checkout is
    # needed at all -- which is the point on an image that has neither torch nor /opt/fa-src.
    [ -d "$CK_FMHA_ROOT/include/ck_tile" ] && { CK="$CK_FMHA_ROOT"; FMHA="$CK_FMHA_ROOT/include"; }
else
    # Locate the .so by path (importing it standalone would need torch's libc10 loaded).
    FA_SO="$(ls /opt/venv/lib/python3.12/site-packages/flash_attn_2_cuda*.so 2>/dev/null | head -1)"
    [ -n "$FA_SO" ] || FA_SO="$(find / -maxdepth 8 -name 'flash_attn_2_cuda*.so' 2>/dev/null | head -1)"
    [ -n "$FA_SO" ] || { echo "no CK kernels: neither $CK_FMHA_ROOT nor flash_attn_2_cuda*.so" >&2; exit 1; }
    KERNEL_DIR="$(dirname "$FA_SO")"; KERNEL_LIB="$(basename "$FA_SO")"
    SOURCE="torch extension ($FA_SO)"
fi

echo "CK kernels: $SOURCE"
echo "XLA_INC   = $XLA_INC"

# CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT and CK_TILE_FMHA_FWD_FAST_EXP2 must match the values the
# kernel instances were compiled with (flash-attention's setup.py, and tools/build_ck_fmha.sh):
# this TU only calls the dispatchers, but it includes the same ck_tile headers, so a different
# value here is an ODR mismatch. Misspell either name and it silently becomes a no-op.
FLAGS="-std=c++20 --offload-arch=gfx90a -fPIC -O2 -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3 -DCK_TILE_FMHA_FWD_FAST_EXP2=1"
INC="-I$CK/include -I$FMHA -I$XLA_INC"

# fwd and bwd are separate TUs (their example headers can't co-include).
hipcc $FLAGS $INC -c "$HERE/lumi_fa_ffi.hip" -o "$HERE/lumi_fa_fwd.o"
hipcc $FLAGS $INC -c "$HERE/lumi_fa_bwd.hip" -o "$HERE/lumi_fa_bwd.o"
hipcc --offload-arch=gfx90a -fPIC -shared -O2 \
  "$HERE/lumi_fa_fwd.o" "$HERE/lumi_fa_bwd.o" \
  -o "$OUT/liblumi_fa_ffi.so" \
  -L"$KERNEL_DIR" -l:"$KERNEL_LIB" -Wl,-rpath,"$KERNEL_DIR"
rm -f "$HERE/lumi_fa_fwd.o" "$HERE/lumi_fa_bwd.o"

echo "Built: $OUT/liblumi_fa_ffi.so"
nm -D "$OUT/liblumi_fa_ffi.so" | grep -iE "LumiFaFwd|LumiFaBwd|LumiFaKvCache|LumiFaKvCacheStep" || echo "WARN: handler symbols not found"
