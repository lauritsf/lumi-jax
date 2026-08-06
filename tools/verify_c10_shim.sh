#!/bin/bash
# Prove the vendored c10 headers (csrc/shim-c10) are interchangeable with torch's:
#   tools/verify_c10_shim.sh [full-image.sif] [lean-image.sif]
# The vendored types must be THE SAME types as torch's: drift is silent, since a
# differently-mangled instantiation still builds, it just stops being the symbol the FFI
# dispatches to. So compile one TU three ways and diff the demangled symbols --
# (1) full image + torch headers is the reference, (2) full image + vendored isolates
# headers from toolchain, (3) lean image + vendored is the case that matters, an image
# where `import torch` raises.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/image_name.sh"
WORK="${LUMI_JAX_WORK:?set it, or create tools/site.local.sh}/images"
REPO="$(cd "$HERE/.." && pwd)"
FULL="${1:-$WORK/lumi-jax-torch-j091-latest.sif}"
LEAN="${2:-$WORK/lumi-jax-stage-fmha-j091-latest.sif}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Both types, both directions: the kernels store as Half/BFloat16 and compute in float.
cat > "$TMP/probe.cpp" <<'EOF'
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
template <typename T> T to_t(float x) { return static_cast<T>(x); }
template <typename T> float to_f(T x) { return static_cast<float>(x); }
template c10::Half     to_t<c10::Half>(float);
template c10::BFloat16 to_t<c10::BFloat16>(float);
template float to_f<c10::Half>(c10::Half);
template float to_f<c10::BFloat16>(c10::BFloat16);
EOF

compile() {   # <sif> <include-dir> <outfile>
    singularity exec --no-home -B "$TMP:$TMP" -B "$REPO:$REPO" "$1" bash -c "
        g++ -std=c++17 -O2 -c -I'$2' '$TMP/probe.cpp' -o '$TMP/o.o' && nm -C '$TMP/o.o' \
            | awk '{\$1=\"\"; print}' | sort" > "$3"
}

TORCH_INC=/opt/venv/lib/python3.12/site-packages/torch/include
compile "$FULL" "$TORCH_INC"        "$TMP/1-full-torch.txt"
compile "$FULL" "$REPO/csrc/shim-c10" "$TMP/2-full-shim.txt"
compile "$LEAN" "$REPO/csrc/shim-c10" "$TMP/3-lean-shim.txt"

# A lean image that still had torch would make case 3 prove nothing, so say so out loud.
# `|| true`: the FAILING import is the wanted outcome, and set -e would abort before printing.
echo -n "lean image torch check: "
{ singularity exec --no-home "$LEAN" python3 -c 'import torch' 2>&1 || true; } | tail -1

rc=0
for f in 2-full-shim 3-lean-shim; do
    if diff -q "$TMP/1-full-torch.txt" "$TMP/$f.txt" >/dev/null; then
        echo "  $f  identical to torch's headers"
    else
        echo "  $f  DIFFERS:"; diff "$TMP/1-full-torch.txt" "$TMP/$f.txt" | sed 's/^/    /'
        rc=1
    fi
done

echo
sed 's/^/  /' "$TMP/1-full-torch.txt"
[ "$rc" = 0 ] && echo "C10 SHIM OK -- same types, same mangled names, no torch needed" \
              || echo "FATAL: the vendored headers are not the same types"
exit "$rc"
