"""Pure-JAX rotary embedding."""
import jax.numpy as jnp


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
