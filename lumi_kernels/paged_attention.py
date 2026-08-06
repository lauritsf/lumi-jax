"""Paged-attention decode for JAX on gfx90a. The compute is the container's torch-ABI
`paged_attention_v1`, reachable eagerly (`paged_attention`, dlpack bridge with clones +
host sync) or jittably (`paged_attention_ffi`, libtorch-in-FFI); `write_kv_cache` is a
plain scatter, pure JAX. KV-cache layout (vLLM, matches the kernel), X = 16 // dtype_bytes:
  key_cache   (num_blocks, num_kv_heads, head_size//X, block_size, X)
  value_cache (num_blocks, num_kv_heads, head_size, block_size)
"""
from __future__ import annotations

import ctypes
import os

import jax
import jax.numpy as jnp
import numpy as np

_torch = None
_ops = None
_ffi_registered = False
from lumi_kernels._lib import so_path as _so_path, have as _have

_VLLM_SO_NAME = "liblumi_vllm_ffi.so"


def _lazy():
    global _torch, _ops
    if _torch is None:
        import torch
        import vllm._custom_ops as ops
        _torch = torch
        _ops = ops


def _to_torch(x):
    jax.block_until_ready(x)
    return _torch.utils.dlpack.from_dlpack(x.__dlpack__())


def _to_jax(t):
    return jax.dlpack.from_dlpack(t)


def is_available():
    try:
        _lazy()
        return True
    except Exception:
        return False


def x_factor(dtype):
    return 16 // jnp.dtype(dtype).itemsize  # bf16/fp16 -> 8


def empty_kv_cache(num_blocks, num_kv_heads, head_size, block_size, dtype):
    X = x_factor(dtype)
    kc = jnp.zeros((num_blocks, num_kv_heads, head_size // X, block_size, X), dtype)
    vc = jnp.zeros((num_blocks, num_kv_heads, head_size, block_size), dtype)
    return kc, vc


def write_kv_cache(key, value, key_cache, value_cache, slot_mapping):
    """Pure-JAX scatter of key/value (T, num_kv_heads, head_size) into the paged cache
    at the given physical slots (T,). Functional: returns updated caches. (jittable)"""
    T, H, D = key.shape
    nb, _, DX, bs, X = key_cache.shape
    blk = slot_mapping // bs
    off = slot_mapping % bs
    # key_cache[blk, h, d//X, off, d%X] = key[t, h, d]
    dd = jnp.arange(D)
    kc = key_cache.at[blk[:, None, None], jnp.arange(H)[None, :, None],
                      (dd // X)[None, None, :], off[:, None, None],
                      (dd % X)[None, None, :]].set(key)
    vc = value_cache.at[blk[:, None, None], jnp.arange(H)[None, :, None],
                        dd[None, None, :], off[:, None, None]].set(value)
    return kc, vc


def paged_attention(query, key_cache, value_cache, block_tables, seq_lens,
                    num_kv_heads, scale, block_size, max_seq_len):
    """Bridged vLLM paged_attention_v1 (eager). query (num_seqs, num_heads, head_size);
    block_tables (num_seqs, max_blocks) int32; seq_lens (num_seqs,) int32. -> out."""
    _lazy()
    qt = _to_torch(query)
    kc, vc = _to_torch(key_cache), _to_torch(value_cache)
    bt, sl = _to_torch(block_tables), _to_torch(seq_lens)
    out = _torch.empty_like(qt)
    one = _torch.tensor(1.0, device="cuda")
    _ops.paged_attention_v1(out, qt, kc, vc, int(num_kv_heads), float(scale), bt, sl,
                            int(block_size), int(max_seq_len), None, "auto", one, one)
    _torch.cuda.synchronize()
    return _to_jax(out)


def _ensure_ffi():
    global _ffi_registered
    if _ffi_registered:
        return
    import torch  # noqa: F401  (pull libtorch before dlopen)
    lib = ctypes.CDLL(_so_path(_VLLM_SO_NAME))
    addr = ctypes.cast(lib.LumiPagedAttnV1, ctypes.c_void_p).value
    jax.ffi.register_ffi_target("lumi_paged_attn_v1", jax.ffi.pycapsule(addr), platform="ROCM")
    _ffi_registered = True


def ffi_available():
    try:
        _ensure_ffi()
        return _have(_VLLM_SO_NAME)
    except Exception:
        return False


def paged_attention_ffi(query, key_cache, value_cache, block_tables, seq_lens,
                        num_kv_heads, scale, block_size, max_seq_len):
    """JITTABLE paged_attention_v1 via libtorch-in-FFI (no dlpack bridge, no host sync).
    Same args as `paged_attention`. block_tables/seq_lens int32; k/v scale fixed at 1
    (non-fp8 kv cache). Returns out (num_seqs, num_heads, head_size)."""
    _ensure_ffi()
    out_t = jax.ShapeDtypeStruct(query.shape, query.dtype)
    one = jnp.ones((1,), jnp.float32)
    return jax.ffi.ffi_call("lumi_paged_attn_v1", out_t)(
        query, key_cache, value_cache, block_tables.astype(jnp.int32),
        seq_lens.astype(jnp.int32), one, one,
        num_kv_heads=np.int32(num_kv_heads), scale=np.float64(scale),
        block_size=np.int32(block_size), max_seq_len=np.int32(max_seq_len))


def reference(query, key, value, seq_lens, num_kv_heads, scale):
    """Full-attention reference (numpy-ish in JAX) over contiguous per-seq K/V, for tests.
    key/value: list of (L_i, num_kv_heads, head_size); query (num_seqs, num_heads, head_size)."""
    outs = []
    H = query.shape[1]
    g = H // num_kv_heads
    for i, (K, V) in enumerate(zip(key, value)):
        Kr = jnp.repeat(K, g, axis=1).astype(jnp.float32)
        Vr = jnp.repeat(V, g, axis=1).astype(jnp.float32)
        q = query[i].astype(jnp.float32)
        s = jnp.einsum("hd,lhd->hl", q, Kr) * scale
        p = jax.nn.softmax(s, axis=-1)
        outs.append(jnp.einsum("hl,lhd->hd", p, Vr))
    return jnp.stack(outs).astype(query.dtype)
