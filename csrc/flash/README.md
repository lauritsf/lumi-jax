# CK flash attention bound into JAX (XLA FFI)

`lumi_fa_ffi.hip` (fwd) + `lumi_fa_bwd.hip` (bwd) + `lumi_kernels/attention/flash_ffi.py`. A
real XLA custom call that hands JAX device buffers and the ROCm stream straight to the CK
`fmha_fwd` / `fmha_bwd<2>` dispatchers -- **no torch, no dlpack, no host sync, fully
jittable**. Forward and backward, wired through `jax.custom_vjp`.

Build: `bash build_ffi.sh` -> `liblumi_fa_ffi.so` (seconds; it links prebuilt kernels and
recompiles none). Call it from `lumi_kernels.attention.flash_ffi.flash_attention_ffi`.

**Full port of `mha_fwd`/`mha_bwd` for this CK build** -- not a subset: single/multi-head,
**GQA/MQA** (fwd+bwd with group reduction), **cross-attention** (S_q≠S_k),
**sliding-window/local** (`window=(left,right)`), **ALiBi**, **dropout** (seed/offset),
fp16+bf16, head dim ≤ 256. softcap is commented out upstream here, so it is unsupported.
Validated against `flash_attn_func` across the matrix in `tests/attention/`.

Constraints worth knowing:

- `mask_info::decode` mirrors flash_attn's mask construction exactly (causal = window
  right=0; local = `b:left,right`).
- bwd scratch (`dq_acc` zeroed + `softmax_d`) comes from XLA's `ScratchAllocator`;
  single-split and non-deterministic by default, `deterministic=True` uses N splits.
- GQA bwd returns dk/dv expanded to H_q heads; the Python `custom_vjp` sums over the
  H_q/H_k groups, as `mha_bwd`'s `at::sum_out` does.
- Dropout's philox (seed, offset) is a device operand, so `dropout_seed` may be traced:
  per-step dropout with no recompile. Full 64-bit seeds need `jax_enable_x64`.
- fwd and bwd are separate translation units -- their CK example headers cannot co-include
  -- and link into one `.so`.

Tests: `tests/attention/test_flash_ffi.py`, `test_flash_varlen.py`, `test_alibi.py`,
`test_kvcache.py`.

## Other backends

AOTriton (`../aotriton/`) is faster on the forward, CK faster on the backward; the
`flash_attention_hybrid` binding mixes them.
