# Vendored c10 half/bfloat16 types

Fourteen header files copied verbatim from PyTorch, so that `csrc/sscan` and `csrc/conv1d` can
be compiled on an image with no PyTorch in it.

## Why this exists

Neither target links torch — `csrc/sscan` shims `C10_CUDA_CHECK` and
`tools/build_causal_conv1d.sh` does the same. The coupling is at compile time only: the
upstream kernels use `at::Half` and `at::BFloat16` as **template storage types**, so without
these headers the two targets can only be built on an image that ships PyTorch.

## Why copy rather than substitute

Swapping in `__half` / `__hip_bfloat16` changes the mangled name of every kernel
instantiation and the numerics of every conversion. Copying the real types keeps both
**identical by construction**, which is checkable: `tools/verify_c10_shim.sh` builds a
translation unit both ways and compares the demangled symbol lists, and the build scripts
assert it.

## Provenance

Copied from our `torch` image, `lumi-jax-torch-j091-latest.sif` (whose torch comes
from LAIF's `full` layer):

```
torch 2.10.0+rocm7.0_lumi_aif_20260513142306
/opt/venv/lib/python3.12/site-packages/torch/include/
```

The set is **derived from the consumers**, not chosen by hand — compile the real translation
units against torch's headers with `-H` and take every torch-tree header they report:

```bash
hipcc -std=c++20 --offload-arch=gfx90a -DUSE_ROCM -D__HIP_PLATFORM_AMD__ \
      -I<shims> -I<upstream src> -I$TORCH/include -I$XLA -E -H -x hip <tu>.hip -o /dev/null \
  2>&1 | grep -oE "$TORCH/include/[^ ]*"
```

Fourteen files:

```
c10/util/Half.h                  torch/headeronly/util/Half.h
c10/util/BFloat16.h              torch/headeronly/util/BFloat16.h
c10/util/complex.h               torch/headeronly/util/complex.h
c10/util/complex_math.h          torch/headeronly/util/bit_cast.h
c10/util/complex_utils.h         torch/headeronly/util/floating_point_utils.h
c10/macros/Macros.h              torch/headeronly/macros/Export.h
                                 torch/headeronly/macros/Macros.h
                                 torch/headeronly/macros/cmake_macros.h
```

Eight of the fourteen are under `torch/headeronly/`, upstream's own explicitly header-only,
link-free subset. Derive the set from the consumers rather than picking by hand: a missing
header of this kind breaks one target only — `c10/util/complex.h`, for instance, reaches
only `csrc/sscan`, via `scalar_value_type`.

Licensed BSD-3-Clause; `LICENSE.pytorch` and `NOTICE.pytorch` are copied alongside, unmodified.

## When torch moves

These are pinned to torch 2.10.0 and are **not** automatically re-synced. `cmake_macros.h`
is generated at torch build time, so it is the one most likely to drift. Re-copy them when
the base's torch version moves, and let the symbol comparison catch a real change:

```bash
tools/verify_c10_shim.sh <full-image.sif>
```

A drift here is not silent — different types produce different mangled names, and every
build script that uses the shim asserts against the torch-header build.
