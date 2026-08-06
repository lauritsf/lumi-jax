"""JAX binding to the container's CK flash kernels via XLA FFI -- a full port of flash_attn's
mha_fwd / mha_bwd for this CK build: MHA and GQA/MQA (H_k <= H_q, H_q % H_k == 0),
cross-attention (S_q != S_k), sliding-window/local (window=(left,right), causal is right=0),
ALiBi, dropout, fp16/bf16, D <= 256. (softcap is commented out in this CK build.)

Forward and backward are both XLA custom calls (jittable, no torch/dlpack/host-sync); for
GQA the backward returns dk/dv expanded to H_q heads and we sum the groups here. Dropout's
philox (seed, offset) is a 2-element uint64 DEVICE buffer threaded fwd->bwd as a custom_vjp
residual, so `dropout_seed` may be traced under jit -- per-step dropout, no recompile.
"""
from __future__ import annotations

import ctypes
import functools
import math
import os

import jax
import jax.numpy as jnp
import numpy as np

_NUM_CU = 110  # MI250X per-GCD compute units (gfx90a); split-KV heuristic target = 2*_NUM_CU

from lumi_kernels._lib import so_path as _so_path, have as _have

_SO_NAME = "liblumi_fa_ffi.so"
_registered = False
_DUMMY_ALIBI = None  # set after jax import


def _ensure_registered():
    global _registered, _DUMMY_ALIBI
    if _registered:
        return
    import torch  # noqa: F401  (pulls libtorch/libc10 in before we dlopen ours)
    import flash_attn  # noqa: F401
    lib = ctypes.CDLL(_so_path(_SO_NAME))
    for target, symbol in (("lumi_fa_fwd", "LumiFaFwd"), ("lumi_fa_bwd", "LumiFaBwd"),
                           ("lumi_fa_varlen_fwd", "LumiFaVarlenFwd"),
                           ("lumi_fa_varlen_bwd", "LumiFaVarlenBwd"),
                           ("lumi_fa_kvcache", "LumiFaKvCache"),
                           ("lumi_fa_kvcache_step", "LumiFaKvCacheStep")):
        addr = ctypes.cast(getattr(lib, symbol), ctypes.c_void_p).value
        jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(addr), platform="ROCM")
    # numpy, not jnp: first use inside a jit trace must not leak a tracer.
    _DUMMY_ALIBI = np.zeros((1,), np.float32)
    _registered = True


def is_available():
    try:
        _ensure_registered()
        return _have(_SO_NAME)
    except Exception:
        return False


def _norm_window(causal, window):
    wl, wr = window
    if causal:
        wr = 0
    return int(wl), int(wr)


def _rng_buf(seed):
    """(seed, offset) as a 16-byte device operand the FFI reads as 2x uint64; a Python int
    becomes a constant, an array/tracer stays traced (per-step dropout under jit).

    Little-endian uint32 limbs, so the buffer is exactly 16 bytes whatever jax_enable_x64
    says: under x64-off jnp cannot hold uint64 and would truncate to 8 bytes, making the
    FFI's `rng[1]` read OUT OF BOUNDS. Full 64-bit seeds need jax_enable_x64."""
    if isinstance(seed, (int, np.integer)):
        s = int(seed) & 0xFFFFFFFFFFFFFFFF
        return np.array([s & 0xFFFFFFFF, s >> 32, 0, 0], np.uint32)
    arr = jnp.asarray(seed).reshape(-1)
    if arr.shape[0] == 1:
        arr = jnp.concatenate([arr, jnp.zeros((1,), arr.dtype)])
    if jax.config.jax_enable_x64:  # full 64-bit (seed, offset) preserved
        return arr.astype(jnp.uint64)
    a = arr.astype(jnp.uint32)  # x64-off: <=32-bit already; pad hi limbs with 0 -> 16 bytes
    z = jnp.zeros((), jnp.uint32)
    return jnp.stack([a[0], z, a[1], z])


def _attrs(q, scale, causal, window, has_alibi, dropout_p, det=None):
    wl, wr = _norm_window(causal, window)
    a = dict(
        scale=np.float64(scale),
        is_bf16=np.int32(1 if q.dtype == jnp.bfloat16 else 0),
        is_causal=np.int32(1 if causal else 0),
        window_left=np.int32(wl),
        window_right=np.int32(wr),
        has_alibi=np.int32(1 if has_alibi else 0),
        p_drop=np.float64(dropout_p),
    )
    if det is not None:
        a["deterministic"] = np.int32(1 if det else 0)
    return a


def _fwd_call(q, k, v, alibi, rng, scale, causal, window, has_alibi, dropout_p):
    B, Sq, Hq, D = q.shape
    out_t = jax.ShapeDtypeStruct(q.shape, q.dtype)
    lse_t = jax.ShapeDtypeStruct((B, Hq, Sq), jnp.float32)
    return jax.ffi.ffi_call("lumi_fa_fwd", (out_t, lse_t))(
        q, k, v, alibi, rng, **_attrs(q, scale, causal, window, has_alibi, dropout_p))


def _bwd_call(dout, q, k, v, out, lse, alibi, rng, scale, causal, window, has_alibi,
              dropout_p, deterministic):
    B, Sq, Hq, D = q.shape
    Sk, Hk = k.shape[1], k.shape[2]
    dq_t = jax.ShapeDtypeStruct(q.shape, q.dtype)
    # dk/dv come back EXPANDED to H_q heads; reduce to H_k below for GQA.
    dk_t = jax.ShapeDtypeStruct((B, Sk, Hq, D), k.dtype)
    dv_t = jax.ShapeDtypeStruct((B, Sk, Hq, D), v.dtype)
    dq, dk_e, dv_e = jax.ffi.ffi_call("lumi_fa_bwd", (dq_t, dk_t, dv_t))(
        dout, q, k, v, out, lse, alibi, rng,
        **_attrs(q, scale, causal, window, has_alibi, dropout_p, deterministic))
    if Hk != Hq:
        ng = Hq // Hk
        dk = dk_e.reshape(B, Sk, Hk, ng, D).sum(axis=3)
        dv = dv_e.reshape(B, Sk, Hk, ng, D).sum(axis=3)
    else:
        dk, dv = dk_e, dv_e
    return dq, dk, dv


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9))
def _flash_ffi(q, k, v, rng, causal, scale, window, alibi, dropout_p, deterministic):
    _ensure_registered()
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    has_alibi = alibi is not None
    ab = alibi if has_alibi else _DUMMY_ALIBI
    out, _ = _fwd_call(q, k, v, ab, rng, scale, causal, window, has_alibi, dropout_p)
    return out


def flash_attention_ffi(q, k, v, *, causal=False, scale=None, window=(-1, -1),
                        alibi=None, dropout_p=0.0, dropout_seed=0, deterministic=False):
    """Fused CK flash attention via XLA FFI (jittable fwd+bwd). q/k/v (B,S,H,D) fp16/bf16.
    `window=(left,right)`, causal forces right=0. `alibi`: (Hq) or (B,Hq) f32 or None.
    `dropout_seed`: int, or uint64 array (seed, or (2,) seed+offset) -- may be traced under
    jit for per-step dropout without recompiling."""
    # nondiff_argnums needs positional args -> forward them positionally.
    return _flash_ffi(q, k, v, _rng_buf(dropout_seed), causal, scale, window, alibi,
                      dropout_p, deterministic)


def _vjp_fwd(q, k, v, rng, causal, scale, window, alibi, dropout_p, deterministic):
    # custom_vjp fwd uses the PRIMAL signature (diff args in their original spots).
    _ensure_registered()
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    has_alibi = alibi is not None
    ab = alibi if has_alibi else _DUMMY_ALIBI
    out, lse = _fwd_call(q, k, v, ab, rng, scale, causal, window, has_alibi, dropout_p)
    return out, (q, k, v, out, lse, ab, rng)


def _vjp_bwd(causal, scale, window, alibi, dropout_p, deterministic, res, dout):
    q, k, v, out, lse, ab, rng = res
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    has_alibi = alibi is not None
    dq, dk, dv = _bwd_call(dout, q, k, v, out, lse, ab, rng, scale, causal, window,
                           has_alibi, dropout_p, deterministic)
    return dq, dk, dv, _int_float0(rng)


_flash_ffi.defvjp(_vjp_fwd, _vjp_bwd)


# Varlen / packed (mha_varlen_fwd/bwd): q,k,v (total, H, D); cu_seqlens_{q,k} (B+1,) int32.
def _varlen_attrs(q, scale, causal, window, has_alibi, dropout_p,
                  max_sq, max_sk, det=None):
    a = _attrs(q, scale, causal, window, has_alibi, dropout_p, det)
    a["max_seqlen_q"] = np.int32(max_sq)
    a["max_seqlen_k"] = np.int32(max_sk)
    return a


def _vl_fwd_call(q, k, v, cu_q, cu_k, alibi, rng, scale, causal, window, has_alibi,
                 dropout_p, max_sq, max_sk):
    total_q, Hq, D = q.shape
    out_t = jax.ShapeDtypeStruct(q.shape, q.dtype)
    lse_t = jax.ShapeDtypeStruct((Hq, total_q), jnp.float32)
    return jax.ffi.ffi_call("lumi_fa_varlen_fwd", (out_t, lse_t))(
        q, k, v, cu_q, cu_k, alibi, rng,
        **_varlen_attrs(q, scale, causal, window, has_alibi, dropout_p, max_sq, max_sk))


def _vl_bwd_call(dout, q, k, v, out, lse, cu_q, cu_k, alibi, rng, scale, causal, window,
                 has_alibi, dropout_p, max_sq, max_sk, deterministic):
    total_q, Hq, D = q.shape
    total_k, Hk = k.shape[0], k.shape[1]
    dq_t = jax.ShapeDtypeStruct(q.shape, q.dtype)
    dk_t = jax.ShapeDtypeStruct((total_k, Hq, D), k.dtype)  # expanded
    dv_t = jax.ShapeDtypeStruct((total_k, Hq, D), v.dtype)
    dq, dk_e, dv_e = jax.ffi.ffi_call("lumi_fa_varlen_bwd", (dq_t, dk_t, dv_t))(
        dout, q, k, v, out, lse, cu_q, cu_k, alibi, rng,
        **_varlen_attrs(q, scale, causal, window, has_alibi, dropout_p, max_sq, max_sk, deterministic))
    if Hk != Hq:
        ng = Hq // Hk
        dk = dk_e.reshape(total_k, Hk, ng, D).sum(axis=2)
        dv = dv_e.reshape(total_k, Hk, ng, D).sum(axis=2)
    else:
        dk, dv = dk_e, dv_e
    return dq, dk, dv


@functools.partial(jax.custom_vjp, nondiff_argnums=tuple(range(4, 14)))
def _flash_varlen(q, k, v, rng, cu_q, cu_k, max_sq, max_sk, causal, scale, window,
                  alibi, dropout_p, deterministic):
    _ensure_registered()
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    has_alibi = alibi is not None
    ab = alibi if has_alibi else _DUMMY_ALIBI
    out, _ = _vl_fwd_call(q, k, v, cu_q, cu_k, ab, rng, scale, causal, window, has_alibi,
                          dropout_p, max_sq, max_sk)
    return out


def _vl_vjp_fwd(q, k, v, rng, cu_q, cu_k, max_sq, max_sk, causal, scale, window,
                alibi, dropout_p, deterministic):
    _ensure_registered()
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    has_alibi = alibi is not None
    ab = alibi if has_alibi else _DUMMY_ALIBI
    out, lse = _vl_fwd_call(q, k, v, cu_q, cu_k, ab, rng, scale, causal, window, has_alibi,
                            dropout_p, max_sq, max_sk)
    return out, (q, k, v, out, lse, cu_q, cu_k, ab, rng)


def _vl_vjp_bwd(cu_q, cu_k, max_sq, max_sk, causal, scale, window, alibi,
                dropout_p, deterministic, res, dout):
    q, k, v, out, lse, cu_q_r, cu_k_r, ab, rng = res
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    has_alibi = alibi is not None
    dq, dk, dv = _vl_bwd_call(dout, q, k, v, out, lse, cu_q_r, cu_k_r, ab, rng, scale,
                              causal, window, has_alibi, dropout_p,
                              max_sq, max_sk, deterministic)
    return dq, dk, dv, _int_float0(rng)


_flash_varlen.defvjp(_vl_vjp_fwd, _vl_vjp_bwd)


# Decode / KV-cache forward: CK split-KV, torch-free FFI, jittable. Forward-only; the cache
# write is a separate pure-JAX scatter.
_SPLIT_TARGET_CTAS = 2048   # empirically tuned (MI250X): CTAs needed to saturate HBM in decode
_SPLIT_CAP = 32             # combine overhead dominates past ~32 splits


def _num_splits_heuristic(B, Hq, Sq, Sk, block_m=128, block_n=128):
    """Split-KV count for decode, empirically tuned on MI250X (gfx90a, 110 CUs).

    Decode is HBM-bound: split K until ~`_SPLIT_TARGET_CTAS` CTAs run concurrently, capped
    by `_SPLIT_CAP` and the K-block count. Do NOT use CK's override_num_splits_if_necessary
    -- it hardcodes num_n_blocks=1 and so returns 1 always, disabling split-KV."""
    work = B * Hq * math.ceil(Sq / block_m)          # parallel m-block CTAs (no split)
    n_kblocks = math.ceil(Sk / block_n)
    return max(1, min(_SPLIT_CAP, n_kblocks, -(-_SPLIT_TARGET_CTAS // work)))


def flash_attn_kvcache_ffi(q, k_cache, v_cache, cache_seqlens, *, causal=False,
                           scale=None, num_splits=None):
    """Flash decode over a KV cache via CK split-KV (jittable, torch-free FFI, no backward).
    q (B,Sq,Hq,D); k_cache/v_cache (B,Sk,Hk,D); cache_seqlens (B,) int32 = per-batch valid
    length. num_splits None -> heuristic (worth 5-8x on small-batch decode)."""
    _ensure_registered()
    B, Sq, Hq, D = q.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    if num_splits is None:
        num_splits = _num_splits_heuristic(B, Hq, Sq, k_cache.shape[1])
    s = int(num_splits)
    out_t = jax.ShapeDtypeStruct(q.shape, q.dtype)
    # split-KV fp32 workspaces are allocated inside the handler via XLA's ScratchAllocator
    return jax.ffi.ffi_call("lumi_fa_kvcache", out_t)(
        q, k_cache, v_cache, cache_seqlens.astype(jnp.int32),
        scale=np.float64(scale), is_causal=np.int32(1 if causal else 0),
        num_splits=np.int32(s))


def flash_attn_kvcache_step(q, k_cache, v_cache, cache_seqlens, *, knew=None, vnew=None,
                            rotary_cos=None, rotary_sin=None, rotary_interleaved=False,
                            block_table=None, causal=False, scale=None, num_splits=None):
    """Full autoregressive decode step in one kernel call (jittable, torch-free FFI,
    inference only): appends knew/vnew at cache_seqlens, applies rotary to q and knew, then
    attends with CK split-KV (flash_attn's mha_fwd_kvcache composition).

    q (B,Sq,Hq,D). Contiguous cache: k/v_cache (B,Sk,Hk,D). Paged: (num_blocks,
    page_block_size, Hk, D) with page_block_size % 128 == 0 and block_table (B, max_blocks)
    int32. cache_seqlens (B,) int32 = pre-append lengths. knew/vnew (B,Sknew,Hk,D).
    rotary_cos/sin (seqlen_ro, rotary_dim/2) in q.dtype, rotary_dim % 16 == 0, seqlen_ro >=
    Sk. Returns (out, k_cache, v_cache); caches are updated IN PLACE via
    input_output_aliases (donate them under jit to avoid copies)."""
    _ensure_registered()
    B, Sq, Hq, D = q.shape
    Hk = k_cache.shape[2]
    paged = block_table is not None
    if paged:
        pbs = k_cache.shape[1]
        if pbs % 128:
            raise ValueError(f"page_block_size must be divisible by 128, got {pbs}")
        Sk = block_table.shape[1] * pbs
    else:
        Sk = k_cache.shape[1]
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    if num_splits is None:
        num_splits = _num_splits_heuristic(B, Hq, Sq, Sk)

    has_knew = knew is not None
    if has_knew != (vnew is not None):
        raise ValueError("knew and vnew must be passed together")
    rotary_dim = 0
    if rotary_cos is not None:
        if not has_knew:
            raise ValueError("rotary requires knew/vnew (rope offsets come from cache_seqlens)")
        rotary_dim = 2 * rotary_cos.shape[1]
        if rotary_dim % 16:
            raise ValueError(f"rotary_dim must be divisible by 16, got {rotary_dim}")
        if rotary_cos.shape[0] < Sk:
            raise ValueError("rotary_cos/sin seqlen must be >= cache seqlen")

    cache_seqlens = cache_seqlens.astype(jnp.int32)
    append_seqlens = cache_seqlens + (knew.shape[1] if has_knew else 0)
    dummy4 = np.zeros((1, 1, 1, 1), q.dtype)
    dummy2 = np.zeros((1, 1), q.dtype)
    rets = (jax.ShapeDtypeStruct(q.shape, q.dtype),         # out
            jax.ShapeDtypeStruct(q.shape, q.dtype),         # q_out (rotated; discarded)
            jax.ShapeDtypeStruct(k_cache.shape, k_cache.dtype),
            jax.ShapeDtypeStruct(v_cache.shape, v_cache.dtype))
    out, _, kc, vc = jax.ffi.ffi_call(
        "lumi_fa_kvcache_step", rets, input_output_aliases={0: 1, 1: 2, 2: 3})(
        q, k_cache, v_cache,
        knew if has_knew else dummy4, vnew if has_knew else dummy4,
        rotary_cos if rotary_dim else dummy2, rotary_sin if rotary_dim else dummy2,
        cache_seqlens, append_seqlens,
        block_table.astype(jnp.int32) if paged else np.zeros((1, 1), np.int32),
        scale=np.float64(scale), is_causal=np.int32(1 if causal else 0),
        num_splits=np.int32(int(num_splits)), rotary_dim=np.int32(rotary_dim),
        rotary_interleaved=np.int32(1 if rotary_interleaved else 0),
        has_knew=np.int32(1 if has_knew else 0), is_paged=np.int32(1 if paged else 0))
    return out, kc, vc


# AOTriton forward: ~20% faster than CK forward, ~25% slower on backward -- CK for training,
# AOTriton for forward-only. Separate liblumi_aotriton.so linking libaotriton_v2.so.
_AOT_SO_NAME = "liblumi_aotriton.so"
_aot_registered = False


def _ensure_aot_registered():
    global _aot_registered
    if _aot_registered:
        return
    # torch is a fallback, not a requirement: against torch's bundled libaotriton_v2.so the
    # .so resolves only after torch loads; against a standalone one it has its own RPATH.
    try:
        lib = ctypes.CDLL(_so_path(_AOT_SO_NAME))
    except OSError:
        import torch  # noqa: F401
        lib = ctypes.CDLL(_so_path(_AOT_SO_NAME))
    addr = ctypes.cast(lib.LumiAotFwd, ctypes.c_void_p).value
    jax.ffi.register_ffi_target("lumi_aot_fwd", jax.ffi.pycapsule(addr), platform="ROCM")
    _aot_registered = True


def aotriton_available():
    try:
        _ensure_aot_registered()
        return _have(_AOT_SO_NAME)
    except Exception:
        return False


def flash_attention_aotriton(q, k, v, *, causal=False, scale=None, window=(-1, -1)):
    """Forward-only fused attention via AOTriton (jittable). (B,S,H,D) bf16/fp16.
    Faster than CK on the forward; for training use flash_attention_ffi (CK)."""
    _ensure_aot_registered()
    B, S, H, D = q.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    wl, wr = window  # handler encodes causal itself (WindowedAttention magic value)
    if (wl, wr) != (-1, -1):
        # AOTriton's local-window alignment differs from CK and is not wired here.
        raise NotImplementedError(
            "AOTriton local window not wired; use flash_attention_ffi (CK) for windows")
    out_t = jax.ShapeDtypeStruct(q.shape, q.dtype)
    lse_t = jax.ShapeDtypeStruct((B * H, S), jnp.float32)
    out, _ = jax.ffi.ffi_call("lumi_aot_fwd", (out_t, lse_t))(
        q, k, v,
        scale=np.float64(scale),
        is_bf16=np.int32(1 if q.dtype == jnp.bfloat16 else 0),
        is_causal=np.int32(1 if causal else 0),
        window_left=np.int32(wl),
        window_right=np.int32(wr),
    )
    return out


# HYBRID: AOTriton forward + CK backward, which beats any single-backend path. Works only
# because AOTriton's softmax_lse is bit-identical to CK's, so CK's backward consumes it.
def _aot_fwd_lse(q, k, v, scale, causal):
    B, S, H, D = q.shape
    out_t = jax.ShapeDtypeStruct(q.shape, q.dtype)
    lse_t = jax.ShapeDtypeStruct((B * H, S), jnp.float32)
    out, lse = jax.ffi.ffi_call("lumi_aot_fwd", (out_t, lse_t))(
        q, k, v,
        scale=np.float64(scale),
        is_bf16=np.int32(1 if q.dtype == jnp.bfloat16 else 0),
        is_causal=np.int32(1 if causal else 0),
        window_left=np.int32(-1), window_right=np.int32(-1))
    return out, lse.reshape(B, H, S)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _flash_hybrid(q, k, v, causal, scale):
    _ensure_registered(); _ensure_aot_registered()
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    out, _ = _aot_fwd_lse(q, k, v, scale, causal)
    return out


def _hyb_fwd(q, k, v, causal, scale):
    _ensure_registered(); _ensure_aot_registered()
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    out, lse = _aot_fwd_lse(q, k, v, scale, causal)  # AOTriton forward
    return out, (q, k, v, out, lse)


def _hyb_bwd(causal, scale, res, dout):
    q, k, v, out, lse = res
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    return _bwd_call(dout, q, k, v, out, lse, _DUMMY_ALIBI, _rng_buf(0),  # CK backward
                     scale, causal, (-1, -1), False, 0.0, False)


_flash_hybrid.defvjp(_hyb_fwd, _hyb_bwd)


def flash_attention_hybrid(q, k, v, *, causal=False, scale=None):
    """Fastest-per-direction flash attention: AOTriton forward + CK backward (jittable).
    Beats any single-backend path. Causal/non-causal, MHA/GQA/MQA, fp16/bf16, D<=256."""
    return _flash_hybrid(q, k, v, causal, scale)


def flash_attention_varlen_ffi(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                               max_seqlen_k, *, causal=False, scale=None,
                               window=(-1, -1), alibi=None, dropout_p=0.0,
                               dropout_seed=0, deterministic=False):
    """Packed variable-length flash attention via XLA FFI (jittable fwd+bwd).
    q,k,v packed (total, H, D); cu_seqlens_{q,k} (B+1,) int32; max_seqlen_{q,k} ints.
    Same feature set as the dense path (GQA, window, ALiBi, dropout, fp16/bf16)."""
    return _flash_varlen(q, k, v, _rng_buf(dropout_seed), cu_seqlens_q, cu_seqlens_k,
                         int(max_seqlen_q), int(max_seqlen_k), causal, scale, window,
                         alibi, dropout_p, deterministic)


def _int_float0(x):
    """float0 (symbolic-zero) cotangent for an integer operand in a custom_vjp bwd."""
    return np.zeros(x.shape, jax.dtypes.float0)


# cu_q/cu_k/alibi/rng are ordinary OPERAND args -- NOT nondiff_argnums, NOT closed over --
# so this composes with remat (a closure version leaks tracers under remat's re-trace).
@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13, 14))
def _flash_varlen_traced(q, k, v, cu_q, cu_k, alibi, rng,
                         max_sq, max_sk, causal, scale, window, has_alibi,
                         dropout_p, deterministic):
    out, _ = _vl_fwd_call(q, k, v, cu_q, cu_k, alibi, rng, scale, causal, window, has_alibi,
                          dropout_p, max_sq, max_sk)
    return out


def _tr_fwd(q, k, v, cu_q, cu_k, alibi, rng, max_sq, max_sk, causal, scale, window,
            has_alibi, dropout_p, deterministic):
    out, lse = _vl_fwd_call(q, k, v, cu_q, cu_k, alibi, rng, scale, causal, window,
                            has_alibi, dropout_p, max_sq, max_sk)
    return out, (q, k, v, out, lse, cu_q, cu_k, alibi, rng)


def _tr_bwd(max_sq, max_sk, causal, scale, window, has_alibi, dropout_p,
            deterministic, res, dout):
    q, k, v, out, lse, cu_q, cu_k, alibi, rng = res
    dq, dk, dv = _vl_bwd_call(dout, q, k, v, out, lse, cu_q, cu_k, alibi, rng, scale,
                              causal, window, has_alibi, dropout_p, max_sq, max_sk,
                              deterministic)
    # float0 for the integer cu_seqlens/rng, zero cotangent for the fixed alibi.
    return dq, dk, dv, _int_float0(cu_q), _int_float0(cu_k), jnp.zeros_like(alibi), _int_float0(rng)


_flash_varlen_traced.defvjp(_tr_fwd, _tr_bwd)


def flash_attention_varlen_traced(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                                  max_seqlen_k, *, causal=False, scale=None,
                                  window=(-1, -1), alibi=None, dropout_p=0.0,
                                  dropout_seed=0, deterministic=False):
    """Packed varlen flash attention with TRACED (dynamic) cu_seqlens -- compiles ONCE for
    any cu_seqlens at runtime and composes with jax.checkpoint/remat. Same math and features
    as `flash_attention_varlen_ffi`, but `cu_seqlens_{q,k}` are ordinary device operands.

    q/k/v packed (total, H, D); cu_seqlens_{q,k} (B+1,) int32. Buffer/output shapes must be
    static (pad packing to a fixed `total`). Differentiable w.r.t. q/k/v only (cu_seqlens
    carry float0). Padded tails are UNINITIALIZED -- mask consumer-side: `out` rows beyond
    cu_seqlens_q[-1], dk/dv beyond cu_seqlens_k[-1] (same for the static `_ffi` path).

    CONTRACT: `max_seqlen_{q,k}` stay STATIC and every per-sequence length must satisfy
    seqlen_q <= max_seqlen_q and seqlen_k <= max_seqlen_k. Unchecked host-side, and with
    `deterministic=True` a violation UNDER-ALLOCATES the dq_acc workspace -> out-of-bounds
    device read."""
    _ensure_registered()
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    has_alibi = alibi is not None
    ab = alibi if has_alibi else _DUMMY_ALIBI
    return _flash_varlen_traced(
        q, k, v, cu_seqlens_q.astype(jnp.int32), cu_seqlens_k.astype(jnp.int32), ab,
        _rng_buf(dropout_seed),
        int(max_seqlen_q), int(max_seqlen_k), causal, scale, window, has_alibi,
        dropout_p, deterministic)
