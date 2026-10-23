#!/bin/bash
# Keep compiler flags matched to flash-attention setup.py; ABI checks cannot detect
# incompatible template instantiation flags.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/image_name.sh"

OUT="${1:-${BUILD_ARTIFACT_DIR:?set LUMI_JAX_WORK}/ck-fmha}"
SIF="${SIF:-${IMAGES_DIR:?set LUMI_JAX_WORK}/lumi-jax-stage-net-j091-latest.sif}"
FA_SRC_ROOT="${FA_SRC_ROOT:?set it to the upstream sources checkout containing flash-attention}"
JOBS="${JOBS:-$(nproc)}"
TARGETS="${TARGETS:-gfx90a}"
# flash-attention's default; narrowing it makes dropped dims raise "no CK kernel instance".
OPTDIM="${OPTDIM:-32,64,128,256}"
# All six: csrc/flash binds the first four, the other two cost ~14% more build for later.
APIS="${APIS:-fwd bwd fwd_splitkv fwd_appendkv batch_prefill pagedkv_prefill}"

CK="$FA_SRC_ROOT/flash-attention/csrc/composable_kernel"
FMHA="$CK/example/ck_tile/01_fmha"
[ -d "$FMHA" ] || { echo "no ck_tile 01_fmha under $FMHA" >&2; exit 1; }
[ -e "$SIF" ]  || { echo "no image at $SIF" >&2; exit 1; }

OUTR="$(readlink -f "$OUT" 2>/dev/null || echo "$OUT")"
CKR="$(readlink -f "$CK")"
FMHAR="$CKR/example/ck_tile/01_fmha"
mkdir -p "$OUTR"/{gen,obj,lib,include}

singularity exec -B /pfs "$SIF" bash -euo pipefail -c "
cd '$FMHAR'

# receipt 2 = the instance set flash_attn_2_cuda.so contains: a like-for-like replacement.
for api in $APIS; do
    python3 generate.py --api \$api --receipt 2 --targets '$TARGETS' \
        --optdim '$OPTDIM' --output_dir '$OUTR/gen' > /dev/null
done
n=\$(ls '$OUTR'/gen/*.cpp | wc -l)
echo \"generated \$n translation units\"

FLAGS=\"-O3 -std=c++20 -fPIC --offload-arch=$TARGETS
  -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero
  -DCK_ENABLE_BF16 -DCK_ENABLE_BF8 -DCK_ENABLE_FP16 -DCK_ENABLE_FP32
  -DCK_ENABLE_FP64 -DCK_ENABLE_FP8 -DCK_ENABLE_INT8 -DCK_USE_XDL
  -DUSE_PROF_API=1 -D__HIP_PLATFORM_HCC__=1
  -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3
  -Wno-undefined-func-template -Wno-float-equal
  -mllvm --lsr-drop-solution=1 -fno-offload-uniform-block
  -mllvm -enable-post-misched=0
  -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false\"
# No quotes: INC is word-split when used, so they would survive inside the -I argument.
INC=\"-I$CKR/include -I$FMHAR\"

# One hipcc per TU. Template-heavy and memory-hungry, so JOBS is the memory knob too.
compile_one() {
    src=\"\$1\"; o='$OUTR'/obj/\$(basename \"\$src\" .cpp).o
    [ -e \"\$o\" ] && return 0                       # resumable: skip what already built
    hipcc \$FLAGS \$INC -c \"\$src\" -o \"\$o\" 2>&1 | tail -3
}
export -f compile_one; export FLAGS INC OUTR='$OUTR'

echo \"compiling with -j$JOBS ...\"
ls '$OUTR'/gen/*.cpp | xargs -P '$JOBS' -I{} bash -c 'compile_one {}'

built=\$(ls '$OUTR'/obj/*.o 2>/dev/null | wc -l)
echo \"compiled \$built / \$n objects\"
[ \"\$built\" -eq \"\$n\" ] || { echo 'FATAL: not every TU produced an object'; exit 1; }

# Static archive, not one hipcc over obj/*.o: 2700 object paths is ~320 KB of argv, past
# MAX_ARG_STRLEN, surfacing only as hipcc's bare 'failed to execute:'. --whole-archive
# because dispatchers pick instances at runtime, so a dead-strip would silently drop them.
rm -f '$OUTR'/lib/libck_fmha.a
find '$OUTR'/obj -name '*.o' -print0 | xargs -0 -n 400 ar qc '$OUTR'/lib/libck_fmha.a
ranlib '$OUTR'/lib/libck_fmha.a
hipcc -shared -fPIC --offload-arch=$TARGETS -o '$OUTR'/lib/libck_fmha.so \
    -Wl,--whole-archive '$OUTR'/lib/libck_fmha.a -Wl,--no-whole-archive
echo \"linked \$(du -h '$OUTR'/lib/libck_fmha.so | cut -f1) libck_fmha.so\"

# A .so that picked up libtorch would work on the full image and fail only on the lean one.
if readelf -d '$OUTR'/lib/libck_fmha.so | grep -qE 'NEEDED.*(torch|c10)'; then
    echo 'FATAL: libck_fmha.so links libtorch'; readelf -d '$OUTR'/lib/libck_fmha.so | grep NEEDED
    exit 1
fi
echo 'links no libtorch: OK'

# Exported and defined (T, not U); fmha_bwd is exported as 'float fmha_bwd<2>(...)'.
for s in fmha_fwd fmha_bwd fmha_fwd_splitkv fmha_fwd_appendkv fmha_batch_prefill; do
    c=\$(nm -DC '$OUTR'/lib/libck_fmha.so | grep -cE \" T (float )?\$s(<[0-9]+>)?\\(\") || true
    printf '  %-22s %s defined\n' \$s \"\$c\"
    [ \"\$c\" -ge 1 ] || { echo \"FATAL: \$s not exported\"; exit 1; }
done

# csrc/flash needs fmha_fwd.hpp, rotary.hpp and ck_tile. Ship the WHOLE CK surface (~28 MB)
# anyway: header-only, so every other CK kernel stays buildable without rebuilding this.
cp '$FMHAR'/*.hpp '$OUTR/include/'
mkdir -p '$OUTR/include/ck_tile' '$OUTR/src'
cp -r '$CKR/include/.' '$OUTR/include/'
cp -r '$CKR/example/ck_tile/.' '$OUTR/src/'
echo \"headers + all ck_tile example sources installed (\$(du -sh '$OUTR/include' '$OUTR/src' | tr '\n' ' '))\"
"

echo
echo "built: $OUTR/lib/libck_fmha.so"
echo "next:  CK_FMHA_ROOT=$OUTR csrc/flash/build_ffi.sh"
