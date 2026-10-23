#!/bin/bash
# Build against the standalone CK library and the selected JAX FFI headers.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(cd "$HERE/../.." && pwd)/lumi_kernels/lib"
mkdir -p "$OUT"
CK="${CK_FMHA_ROOT:-/opt/ck-fmha}"
FMHA="$CK/include"
KERNEL_DIR="$CK/lib"
KERNEL_LIB=libck_fmha.so
test -f "$KERNEL_DIR/$KERNEL_LIB"
test -d "$CK/include/ck_tile"
XLA_INC="$(python3 -c 'import jaxlib,os; print(os.path.join(os.path.dirname(jaxlib.__file__),"include"))')"
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
