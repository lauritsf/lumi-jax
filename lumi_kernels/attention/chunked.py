"""Pure-JAX memory-efficient (chunked) attention: online softmax over K/V blocks, after
Rabe & Staats (arXiv:2112.05682). Standard XLA ops only. Layout (B, S, H, D).

Each query block is wrapped in `jax.checkpoint` so the backward RECOMPUTES its scores
rather than stashing every per-block score tile (quadratic again). That internal remat is
what makes the gradient O(S) and is always on; the public `remat=True` flag is a SECOND,
outer checkpoint around the whole call.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax

# Queries per outer step / keys per inner step; overridable via DEFAULT_CHUNK.
DEFAULT_CHUNK = 512
_MASK_FILL = -1e30  # large finite (not -inf) so all-masked tiles stay NaN-free


def _largest_divisor_at_most(n: int, cap: int) -> int:
    """Largest divisor of n that is <= cap (so blocks tile S exactly)."""
    cap = min(cap, n)
    for c in range(cap, 0, -1):
        if n % c == 0:
            return c
    return 1


def _attend_one_qblock(q_blk, k, v, q_start, kv_chunk, causal, scale):
    """Online scan of all K/V blocks: q_blk (B,Tq,H,D), k/v (B,Sk,H,D) -> out (B,Tq,H,D),
    lse (B,H,Tq)."""
    B, Tq, H, D = q_blk.shape
    Sk = k.shape[1]
    nkv = Sk // kv_chunk
    qs = (q_blk * scale).astype(jnp.float32)

    k_ch = k.reshape(B, nkv, kv_chunk, H, D).transpose(1, 0, 2, 3, 4).astype(jnp.float32)
    v_ch = v.reshape(B, nkv, kv_chunk, H, D).transpose(1, 0, 2, 3, 4).astype(jnp.float32)

    m0 = jnp.full((B, H, Tq), _MASK_FILL, jnp.float32)
    l0 = jnp.zeros((B, H, Tq), jnp.float32)
    acc0 = jnp.zeros((B, H, Tq, D), jnp.float32)

    qpos = q_start + jnp.arange(Tq)

    def body(carry, inp):
        m, l, acc = carry
        kc, vc, blk_idx = inp
        # scores (B,H,Tq,kv_chunk)
        s = jnp.einsum("bqhd,bkhd->bhqk", qs, kc)
        if causal:
            kpos = blk_idx * kv_chunk + jnp.arange(kv_chunk)
            mask = qpos[:, None] >= kpos[None, :]  # (Tq, kv_chunk)
            s = jnp.where(mask[None, None], s, _MASK_FILL)
        m_new = jnp.maximum(m, jnp.max(s, axis=-1))           # (B,H,Tq)
        p = jnp.exp(s - m_new[..., None])                     # (B,H,Tq,kv_chunk)
        corr = jnp.exp(m - m_new)                             # (B,H,Tq)
        l = l * corr + jnp.sum(p, axis=-1)
        acc = acc * corr[..., None] + jnp.einsum("bhqk,bkhd->bhqd", p, vc)
        return (m_new, l, acc), None

    idx = jnp.arange(nkv)
    (m, l, acc), _ = lax.scan(body, (m0, l0, acc0), (k_ch, v_ch, idx))
    out = acc / l[..., None]                                  # (B,H,Tq,D)
    lse = m + jnp.log(l)                                      # (B,H,Tq)
    out = out.transpose(0, 2, 1, 3).astype(q_blk.dtype)       # (B,Tq,H,D)
    return out, lse


def chunked_attention(q, k, v, *, causal=False, scale=None, q_chunk=None,
                      kv_chunk=None, remat=False, return_lse=False):
    """Memory-efficient attention, (B,S,H,D) -> (B,S,H,D). q_chunk/kv_chunk default to
    DEFAULT_CHUNK, snapped down to a divisor of the sequence length; `remat=True` adds an
    outer jax.checkpoint."""
    if remat and not return_lse:
        # Outer checkpoint modelling a checkpointed attention block; recurse with it off.
        inner = functools.partial(
            chunked_attention, causal=causal, scale=scale,
            q_chunk=q_chunk, kv_chunk=kv_chunk, remat=False, return_lse=False,
        )
        return jax.checkpoint(inner)(q, k, v)

    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    qc = _largest_divisor_at_most(Sq, q_chunk or DEFAULT_CHUNK)
    kc = _largest_divisor_at_most(Sk, kv_chunk or DEFAULT_CHUNK)
    nq = Sq // qc

    # Recompute each query block's inner K/V scan in the backward pass.
    qblock = jax.checkpoint(
        functools.partial(_attend_one_qblock, kv_chunk=kc, causal=causal, scale=scale)
    )

    q_blocks = q.reshape(B, nq, qc, H, D).transpose(1, 0, 2, 3, 4)
    q_starts = jnp.arange(nq) * qc

    def over_q(qb, q_start):
        return qblock(qb, k, v, q_start)

    outs, lses = lax.map(lambda xs: over_q(xs[0], xs[1]), (q_blocks, q_starts))
    # outs: (nq, B, qc, H, D) -> (B, Sq, H, D)
    out = outs.transpose(1, 0, 2, 3, 4).reshape(B, Sq, H, D)
    lse = lses.transpose(1, 0, 2, 3).reshape(B, H, Sq) if return_lse else None

    if return_lse:
        return out, lse
    return out


def chunked_attention_remat(q, k, v, **kw):
    return jax.checkpoint(functools.partial(chunked_attention, **kw))(q, k, v)
