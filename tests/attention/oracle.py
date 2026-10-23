"""Float32 attention oracle independent of CK dispatch and FFI layouts."""
import jax
import jax.numpy as jnp


def attention(q, k, v, *, causal=False, scale=None, window=(-1, -1), alibi=None):
    q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
    repeats = q.shape[2] // k.shape[2]
    k, v = (jnp.repeat(x, repeats, axis=2) for x in (k, v))
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
    delta = jnp.arange(k.shape[1])[None, :] - (
        jnp.arange(q.shape[1])[:, None] + k.shape[1] - q.shape[1])
    mask = jnp.ones(delta.shape, dtype=bool)
    if causal:
        mask &= delta <= 0
    if window[0] >= 0:
        mask &= delta >= -window[0]
    if window[1] >= 0:
        mask &= delta <= window[1]
    if alibi is not None:
        slopes = alibi[None, :] if alibi.ndim == 1 else alibi
        scores -= slopes[:, :, None, None] * jnp.abs(delta)
    weights = jax.nn.softmax(jnp.where(mask, scores, -jnp.inf), axis=-1)
    weights = jnp.where(jnp.any(mask, axis=-1)[None, None, :, None], weights, 0)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)
