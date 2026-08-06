"""RoPE (rotary embedding) for JAX on gfx90a.

USE `rope_ref` (pure JAX, NEOX, jittable; cos_sin_cache (max_pos, rotary_dim) = [cos|sin]).
The vLLM kernel is ~1.45x faster in isolation, but it is torch-ABI and in-place, so the
dlpack bridge's clones + host sync make `rope` ~2.5x SLOWER; `rope` is kept only as the
measurement and an FFI building block.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_torch = None


def _lazy():
    global _torch
    if _torch is None:
        import torch
        import vllm._C  # noqa: F401  (registers torch.ops._C.rotary_embedding)
        _torch = torch


def is_available():
    try:
        _lazy()
        return True
    except Exception:
        return False


def _to_torch(x):
    jax.block_until_ready(x)
    return _torch.utils.dlpack.from_dlpack(x.__dlpack__())


def rope(positions, query, key, head_size, cos_sin_cache, is_neox=True):
    """Bridged vLLM rotary_embedding (eager). query/key (num_tokens, num_heads*head_size)
    flattened; positions (num_tokens,) int; cos_sin_cache (max_pos, rotary_dim). Returns
    rotated (query, key) as new arrays."""
    _lazy()
    qt = _to_torch(query).clone()  # kernel mutates in place; keep JAX inputs intact
    kt = _to_torch(key).clone()
    pos = _to_torch(positions).long()  # kernel expects int64 positions
    csc = _to_torch(cos_sin_cache)
    _torch.ops._C.rotary_embedding(pos, qt, kt, int(head_size), csc, bool(is_neox))
    _torch.cuda.synchronize()
    return jax.dlpack.from_dlpack(qt), jax.dlpack.from_dlpack(kt)


def rope_ref(positions, query, key, head_size, cos_sin_cache, is_neox=True):
    """Pure-JAX NEOX RoPE (jittable), matching the kernel convention."""
    rd = cos_sin_cache.shape[1]           # rotary_dim
    d = rd // 2
    cos = cos_sin_cache[positions, :d]    # (T, d)
    sin = cos_sin_cache[positions, d:]    # (T, d)

    def rot(x):
        T = x.shape[0]
        x = x.reshape(T, -1, head_size)
        xr, xpass = x[..., :rd], x[..., rd:]      # rotate only the first rotary_dim
        a, b = xr[..., :d], xr[..., d:]           # NEOX: split halves
        c, s = cos[:, None, :], sin[:, None, :]
        rotated = jnp.concatenate([a * c - b * s, b * c + a * s], axis=-1)
        return jnp.concatenate([rotated, xpass], axis=-1).reshape(T, -1)

    return rot(query.astype(jnp.float32)).astype(query.dtype), \
        rot(key.astype(jnp.float32)).astype(key.dtype)
