"""Reference attention: `naive_reference` (explicit einsum, correctness oracle) and
`xla_dpa` (jax.nn.dot_product_attention, implementation="xla"), the baseline everything is
measured against -- on ROCm it materializes the full (S_q x S_k) score matrix per head, so
peak HBM grows quadratically. Layout throughout attention/: (B, S, H, D), matching
jax.nn.dot_product_attention.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def naive_reference(q, k, v, *, causal: bool = False, scale: float | None = None):
    """Explicit O(S^2)-memory attention, (B,S,H,D) -> (B,S,H,D), computed in float32."""
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    if scale is None:
        scale = 1.0 / jnp.sqrt(jnp.asarray(D, jnp.float32))
    qf = q.astype(jnp.float32)
    kf = k.astype(jnp.float32)
    vf = v.astype(jnp.float32)
    # scores: (B, H, Sq, Sk)
    scores = jnp.einsum("bqhd,bkhd->bhqk", qf, kf) * scale
    if causal:
        qi = jnp.arange(Sq)[:, None]
        ki = jnp.arange(Sk)[None, :]
        mask = qi >= (ki - (Sk - Sq))
        scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bkhd->bqhd", probs, vf)
    return out.astype(q.dtype)


def xla_dpa(q, k, v, *, causal: bool = False, scale: float | None = None):
    """Baseline: jax.nn.dot_product_attention forced to the XLA math lowering."""
    return jax.nn.dot_product_attention(
        q, k, v, scale=scale, is_causal=causal, implementation="xla"
    )
