#!/bin/bash
# Build the libtorch-in-FFI proof -> liblumi_vllm_ffi.so. Links libtorch (HIP) and the
# container's vLLM extension module(s). torch is linked into the .so but stays out of the
# JAX graph; the resulting FFI call is jittable and does no clones/sync. Run in the full
# container.
#
# THE PROBE, INSTEAD OF A HARDCODED MODULE
#
# vLLM moved silu_and_mul into a second extension module and changed its signature with it:
#
#   v0.20.1  vllm/_C.abi3.so                  void silu_and_mul(at::Tensor&, at::Tensor&)
#   v0.22.1  vllm/_C_stable_libtorch.abi3.so  void silu_and_mul(torch::stable::Tensor&, ...)
#
# So the script asks the .so which symbol it exports rather than inferring from a version
# string; vllm_version can disagree with the commit actually built.
#
# The v0.22.1 branch compiles and its mangled name is checked against torch 2.10's headers,
# but no image carrying v0.22.1 exists yet, so it has never been linked or run. Treat the
# first successful test run on such an image as its validation.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(cd "$HERE/../.." && pwd)/lumi_kernels/lib"; mkdir -p "$OUT"
TORCH_DIR="$(python3 -c 'import torch,os; print(os.path.dirname(torch.__file__))')"
TORCH_INC="$TORCH_DIR/include"
TORCH_LIB="$TORCH_DIR/lib"
XLA_INC="$(python3 -c 'import jaxlib,os; print(os.path.join(os.path.dirname(jaxlib.__file__),"include"))')"
VLLM_DIR="$(python3 -c 'import vllm,os; print(os.path.dirname(vllm.__file__))')"

# paged_attention_v1 lives in _C on both tags and is required either way.
CORE_SO="$VLLM_DIR/_C.abi3.so"
test -f "$CORE_SO" || { echo "FATAL: no $CORE_SO"; exit 1; }

# Which silu_and_mul is actually exported? Ask, do not assume.
SYM_ATEN='_Z12silu_and_mulRN2at6TensorES1_'
SYM_STABLE='_Z12silu_and_mulRN5torch6stable6TensorES2_'
STABLE_SO="$VLLM_DIR/_C_stable_libtorch.abi3.so"

# `grep -c ... -gt 0` rather than `grep -q`: under `set -o pipefail` a short-circuiting
# `grep -q` SIGPIPEs its producer, so the pipeline reports failure on a *successful* match.
# `nm -D` on a 100 MB extension module is far past the pipe buffer that makes this fire.
has_sym() {  # has_sym <file> <mangled>
    [ -f "$1" ] || return 1
    [ "$(nm -D "$1" 2>/dev/null | grep -c "$2")" -gt 0 ]
}

EXTRA_DEFS=() ; EXTRA_LIBS=()
if has_sym "$STABLE_SO" "$SYM_STABLE"; then
    echo "vLLM: stable-ABI silu_and_mul in $(basename "$STABLE_SO")  [v0.22.1+ path]"
    echo "  WARNING: this path has never been linked or run before -- treat the first"
    echo "           successful test run as the validation, not this build."
    EXTRA_DEFS+=(-DLUMI_VLLM_STABLE_ABI=1)
    EXTRA_LIBS+=(-l:"$(basename "$STABLE_SO")")
elif has_sym "$CORE_SO" "$SYM_ATEN"; then
    echo "vLLM: at::Tensor silu_and_mul in $(basename "$CORE_SO")  [v0.20.1 path]"
else
    echo "FATAL: neither silu_and_mul symbol found."
    echo "  looked for $SYM_ATEN   in $CORE_SO"
    echo "  looked for $SYM_STABLE in $STABLE_SO"
    echo "  vLLM has moved the op again -- nm -D the vllm/*.abi3.so modules for the new"
    echo "  module and mangled name, then extend the mapping above."
    exit 1
fi

hipcc -std=c++20 --offload-arch=gfx90a -fPIC -shared -O2 \
  -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 "${EXTRA_DEFS[@]}" \
  -I"$TORCH_INC" -I"$TORCH_INC/torch/csrc/api/include" -I"$XLA_INC" \
  "$HERE/lumi_vllm_ffi.hip" -o "$OUT/liblumi_vllm_ffi.so" \
  -L"$TORCH_LIB" -lc10 -lc10_hip -ltorch_cpu -ltorch_hip -ltorch \
  -L"$VLLM_DIR" -l:"$(basename "$CORE_SO")" "${EXTRA_LIBS[@]}" \
  -Wl,-rpath,"$TORCH_LIB" -Wl,-rpath,"$VLLM_DIR"

echo "Built: $OUT/liblumi_vllm_ffi.so"
nm -D "$OUT/liblumi_vllm_ffi.so" | grep -iE "LumiSiluAndMul|LumiPagedAttnV1" || echo "WARN: symbol missing"

# An unresolved silu_and_mul here means the probe picked a module that does not actually
# define it -- a link that "succeeded" against the wrong .so.
if ldd -r "$OUT/liblumi_vllm_ffi.so" 2>&1 | grep -q 'silu_and_mul'; then
    echo "FATAL: silu_and_mul is unresolved in the built library"; exit 1
fi
echo "silu_and_mul resolves"
