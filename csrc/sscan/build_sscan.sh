#!/bin/bash
# Build selective_scan (mamba v2.3.2 csrc, real weights) + FFI handler ->
# liblumi_sscan.so. Fully self-compiled for gfx90a, NO torch link: c10 half/bf16
# types are header-only and C10_CUDA_CHECK / cuda* names are shimmed (sscan_shim/).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(cd "$HERE/../.." && pwd)/lumi_kernels/lib"; mkdir -p "$OUT"
# Upstream sources live outside the repo. FA_SRC_ROOT lets a container build point at a
# vendored copy (/opt/fa-src) without changing the default for existing checkouts.
FA_SRC_ROOT="${FA_SRC_ROOT:?set it to the upstream sources checkout (flash-attention, mamba, causal-conv1d); the container builds export /opt/fa-src}"
SS_SRC="$FA_SRC_ROOT/mamba/csrc/selective_scan"   # mamba @ 77069de
XLA_INC="$(python3 -c 'import jaxlib,os; print(os.path.join(os.path.dirname(jaxlib.__file__),"include"))')"

# c10's Half/BFloat16 are needed as template STORAGE TYPES, never as a link dependency.
# csrc/shim-c10 vendors those header-only files, so prefer them when torch is absent and
# this builds on a lean image. LUMI_C10_SHIM=1 forces them even where torch exists;
# tools/verify_c10_shim.sh uses that to prove both paths produce byte-identical symbols,
# which is why the types are vendored rather than substituted with __half/__hip_bfloat16.
SHIM_C10="$(cd "$HERE/../shim-c10" && pwd)"
TORCH_INC="$(python3 -c 'import torch,os; print(os.path.dirname(torch.__file__))' 2>/dev/null || true)"
if [ "${LUMI_C10_SHIM:-0}" = 1 ] || [ -z "$TORCH_INC" ]; then
    C10_INC="$SHIM_C10"; C10_SRC="vendored (csrc/shim-c10)"
else
    C10_INC="$TORCH_INC/include"; C10_SRC="torch ($TORCH_INC)"
fi
echo "c10 half/bf16 headers: $C10_SRC"

FLAGS="-std=c++20 --offload-arch=gfx90a -fPIC -O3 -DUSE_ROCM -D__HIP_PLATFORM_AMD__"
INC="-I$HERE/sscan_shim -I$SS_SRC -I$C10_INC -I$XLA_INC"

hipcc $FLAGS $INC -include "$HERE/sscan_shim/cuda_compat.h" \
  -c "$HERE/lumi_sscan_inst.hip" -o "$HERE/lumi_sscan_inst.o"
hipcc $FLAGS $INC -c "$HERE/lumi_sscan.hip" -o "$HERE/lumi_sscan_ffi.o"
hipcc --offload-arch=gfx90a -fPIC -shared -O3 \
  "$HERE/lumi_sscan_inst.o" "$HERE/lumi_sscan_ffi.o" -o "$OUT/liblumi_sscan.so"
rm -f "$HERE/lumi_sscan_inst.o" "$HERE/lumi_sscan_ffi.o"

echo "Built: $OUT/liblumi_sscan.so"
nm -D "$OUT/liblumi_sscan.so" | grep -E "LumiSscan" || echo "WARN: symbols missing"
ldd "$OUT/liblumi_sscan.so" | grep -i torch && echo "WARN: torch leaked into link" || echo "torch-free: OK"
