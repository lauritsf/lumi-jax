"""flash_attention_varlen_traced: dynamic (traced) cu_seqlens, compiled ONCE,
bit-identical fwd+grad to the static flash_attention_varlen_ffi."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from lumi_kernels.attention import flash_ffi as F

pytestmark = [pytest.mark.gpu, pytest.mark.so("liblumi_fa_ffi.so")]

H, D, S, DT = 8, 128, 256, jnp.bfloat16


def _pack(seps, seed=0):
    g = np.random.default_rng(seed)
    tk = sum(seps)
    q = jnp.asarray(g.standard_normal((2 * S, H, D)) * .3, DT)
    k = jnp.asarray(g.standard_normal((tk, H, D)) * .3, DT)
    v = jnp.asarray(g.standard_normal((tk, H, D)) * .3, DT)
    cu_q = np.array([0, S, 2 * S], np.int32)
    cu_k = np.array([0, seps[0], seps[0] + seps[1]], np.int32)
    return q, k, v, cu_q, cu_k


def _loss(fn, q, k, v, cq, ck):
    return jnp.sum(fn(q, k, v, cq, ck, S, S, causal=False).astype(jnp.float32))


def test_parity_with_static():
    """Same kernel underneath -> bit-identical fwd and grads."""
    q, k, v, cuq, cuk = _pack([100, 180])
    o_s = F.flash_attention_varlen_ffi(q, k, v, cuq, cuk, S, S, causal=False)
    o_t = F.flash_attention_varlen_traced(q, k, v, jnp.asarray(cuq), jnp.asarray(cuk), S, S, causal=False)
    assert float(jnp.max(jnp.abs(o_t.astype(jnp.float32) - o_s.astype(jnp.float32)))) == 0.0
    gs = jax.grad(lambda a, b, c: _loss(F.flash_attention_varlen_ffi, a, b, c, cuq, cuk), (0, 1, 2))(q, k, v)
    gt = jax.grad(lambda a, b, c: _loss(F.flash_attention_varlen_traced, a, b, c,
                                        jnp.asarray(cuq), jnp.asarray(cuk)), (0, 1, 2))(q, k, v)
    for x, y in zip(gs, gt):
        assert float(jnp.max(jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32)))) == 0.0


def test_single_compile_for_dynamic_cu_seqlens():
    q, _, _, _, _ = _pack([100, 180])
    traces = [0]

    @jax.jit
    def step(q, k, v, cq, ck):
        traces[0] += 1
        return jnp.sum(F.flash_attention_varlen_traced(q, k, v, cq, ck, S, S, causal=False).astype(jnp.float32))

    for seps in ([50, 90], [130, 200], [10, 255]):  # fixed shapes (k buffer=2S), vary split
        kk = jnp.asarray(np.random.default_rng(1).standard_normal((2 * S, H, D)) * .3, DT)
        cq = jnp.array([0, S, 2 * S], jnp.int32)
        ck = jnp.array([0, seps[0], seps[0] + seps[1]], jnp.int32)
        jax.block_until_ready(step(q, kk, kk, cq, ck))
    assert traces[0] == 1, f"recompiled: traces={traces[0]}"


def test_composes_with_checkpoint():
    """remat grad == plain grad, no UnexpectedTracerError (the training-step failure mode)."""
    q, k, v, cuq, cuk = _pack([120, 160], seed=3)
    cuqj, cukj = jnp.asarray(cuq), jnp.asarray(cuk)

    def layer(q, k, v, cq, ck):
        return F.flash_attention_varlen_traced(q, k, v, cq, ck, S, S, causal=False)

    def loss_remat(q, k, v, cq, ck):
        return jnp.sum(jax.checkpoint(layer)(q, k, v, cq, ck).astype(jnp.float32))

    def loss_plain(q, k, v, cq, ck):
        return jnp.sum(layer(q, k, v, cq, ck).astype(jnp.float32))

    gr = jax.jit(jax.grad(loss_remat, (0, 1, 2)))(q, k, v, cuqj, cukj)
    gp = jax.jit(jax.grad(loss_plain, (0, 1, 2)))(q, k, v, cuqj, cukj)
    jax.block_until_ready(gr)
    for a, b in zip(gr, gp):
        assert float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))) == 0.0
