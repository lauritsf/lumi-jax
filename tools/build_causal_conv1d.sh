#!/bin/bash
# Build causal_conv1d's kernels as a standalone shared library, with no torch link:
#   bash tools/build_causal_conv1d.sh [outdir]
# Upstream's "needs torch" is really "needs at::Half / at::BFloat16 as TEMPLATE STORAGE
# TYPES (arithmetic is in float) plus C10_CUDA_CHECK / C10_CUDA_KERNEL_LAUNCH_CHECK"; those
# headers link nothing, so they stay and the macros are shimmed below. Substituting __half /
# __hip_bfloat16 is deliberately NOT done -- it changes every mangled name and every
# conversion's numerics.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/image_name.sh"
REPO="$(cd "$HERE/.." && pwd)"
OUT="${1:-${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}/causal-conv1d}"
FA_SRC_ROOT="${FA_SRC_ROOT:?set it to the upstream sources checkout (flash-attention, mamba, causal-conv1d); the container builds export /opt/fa-src}"
CC_SRC="$FA_SRC_ROOT/causal-conv1d/csrc"

test -d "$CC_SRC" || { echo "FATAL: no causal-conv1d source at $CC_SRC"; exit 1; }

# The vendored headers make the build torch-free; LUMI_C10_SHIM=1 forces them for comparison.
SHIM_C10="$REPO/csrc/shim-c10"
TORCH_DIR="$(python3 -c 'import torch,os; print(os.path.dirname(torch.__file__))' 2>/dev/null || true)"
if [ "${LUMI_C10_SHIM:-0}" = 1 ] || [ -z "$TORCH_DIR" ]; then
    TORCH_INC="$SHIM_C10"
    test -f "$TORCH_INC/c10/util/Half.h" || {
        echo "FATAL: no torch and no vendored c10 headers at $SHIM_C10"; exit 1; }
    echo "c10 half/bf16 headers: vendored (csrc/shim-c10)"
else
    TORCH_INC="$TORCH_DIR/include"
    echo "c10 half/bf16 headers: torch ($TORCH_DIR)"
fi

mkdir -p "$OUT/lib" "$OUT/include" "$OUT/shim"

# c10's macros throw c10::Error, which drags libtorch in; the FFI could not unwind anyway.
cat > "$OUT/shim/c10_cuda_shim.h" <<'EOF'
// Replaces c10/cuda/CUDAException.h for a torch-free build of causal_conv1d: c10's
// macros throw c10::Error, which drags libtorch in.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define C10_CUDA_CHECK(EXPR)                                                    \
  do {                                                                          \
    hipError_t _err = (EXPR);                                                    \
    if (_err != hipSuccess) {                                                    \
      std::fprintf(stderr, "causal_conv1d: HIP error %s at %s:%d\n",             \
                   hipGetErrorString(_err), __FILE__, __LINE__);                 \
      std::abort();                                                              \
    }                                                                            \
  } while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(hipGetLastError())
EOF

# Included by that exact path, so shadow it rather than edit the pinned upstream sources.
mkdir -p "$OUT/shim/c10/cuda"
cp "$OUT/shim/c10_cuda_shim.h" "$OUT/shim/c10/cuda/CUDAException.h"

# Hipify into copies with sed, not torch.utils.hipify: a library whose purpose is removing
# torch must not need a torch build tool. Note `#include <cuda_fp16.h>` sits OUTSIDE the
# file's own `#ifndef USE_ROCM` block -- that guard is what makes this look already done.
mkdir -p "$OUT/src"
cp "$CC_SRC"/*.cu "$CC_SRC"/*.h "$OUT/src/"
sed -i \
    -e 's/\bcudaStream_t\b/hipStream_t/g' \
    -e 's/\bcudaFuncSetAttribute\b/hipFuncSetAttribute/g' \
    -e 's/\bcudaFuncAttributeMaxDynamicSharedMemorySize\b/hipFuncAttributeMaxDynamicSharedMemorySize/g' \
    -e 's/\bcub::/hipcub::/g' \
    -e 's|#include <cuda_fp16.h>|#include <hip/hip_fp16.h>|' \
    -e 's|#include <cub/block/\([a-z_]*\)\.cuh>|#include <hipcub/hipcub.hpp>|' \
    "$OUT/src"/*.cu "$OUT/src"/*.h

cp "$OUT/src"/causal_conv1d.h "$OUT/src"/causal_conv1d_common.h "$OUT/src"/static_switch.h \
   "$OUT/include/"

echo "== compiling causal_conv1d kernels for gfx90a, no torch link =="
OBJS=()
for src in causal_conv1d_fwd.cu causal_conv1d_bwd.cu causal_conv1d_update.cu; do
    obj="$OUT/$(basename "$src" .cu).o"
    echo "  $src"
    # -I"$OUT/shim" FIRST so our CUDAException.h wins over torch's.
    hipcc -c -std=c++17 --offload-arch=gfx90a -fPIC -O2 \
        -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 \
        -I"$OUT/shim" -I"$TORCH_INC" -I"$OUT/src" \
        "$OUT/src/$src" -o "$obj"
    OBJS+=("$obj")
done

echo "== linking =="
hipcc -shared -fPIC --offload-arch=gfx90a -o "$OUT/lib/libcausal_conv1d.so" "${OBJS[@]}"

echo "== verifying it is genuinely torch-free =="
if readelf -d "$OUT/lib/libcausal_conv1d.so" | grep -qE 'NEEDED.*(torch|c10)'; then
    echo "FATAL: links libtorch/c10 -- the whole point of this build is that it does not"
    readelf -d "$OUT/lib/libcausal_conv1d.so" | grep NEEDED
    exit 1
fi
echo "  NEEDED:"; readelf -d "$OUT/lib/libcausal_conv1d.so" | grep NEEDED | sed 's/^/   /'

# Templates, so the pattern must allow a template argument list.
echo "== exported dispatchers =="
for s in causal_conv1d_fwd_cuda causal_conv1d_bwd_cuda; do
    n=$(nm -DC "$OUT/lib/libcausal_conv1d.so" | grep -cE " (T|W) (void )?${s}<" || true)
    printf "  %-28s %s instantiations\n" "$s" "$n"
    [ "$n" -gt 0 ] || { echo "FATAL: ${s} not exported"; exit 1; }
done

echo
echo "Built: $OUT/lib/libcausal_conv1d.so"
echo "Link csrc/conv1d against it with CAUSAL_CONV1D_ROOT=$OUT"
