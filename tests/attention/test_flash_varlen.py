"""CK forward and gradients against an independent float32 JAX reference."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from lumi_kernels.attention import flash_ffi as F
from oracle import attention

pytestmark = [pytest.mark.gpu, pytest.mark.so("liblumi_fa_ffi.so")]


def _vs_reference(seqlens, Hq, Hk, D, causal, window, dtype, max_pad=0):

    jdt = {"bf16": jnp.bfloat16, "fp16": jnp.float16}[dtype]
    total = int(sum(seqlens))
    cu = np.concatenate([[0], np.cumsum(seqlens)]).astype(np.int32)
    max_s = int(max(seqlens)) + max_pad
    g = np.random.default_rng(0)
    qn = g.standard_normal((total, Hq, D)).astype(np.float32) * 0.5
    kn = g.standard_normal((total, Hk, D)).astype(np.float32) * 0.5
    vn = g.standard_normal((total, Hk, D)).astype(np.float32) * 0.5
    gn = g.standard_normal((total, Hq, D)).astype(np.float32) * 0.5
    scale = 1.0 / D ** 0.5

    qj, kj, vj = (jnp.asarray(x).astype(jdt) for x in (qn, kn, vn))
    cuj = jnp.asarray(cu)
    fn = lambda q, k, v: F.flash_attention_varlen_ffi(
        q, k, v, cuj, cuj, max_s, max_s, causal=causal, scale=scale, window=window)
    out_j = jax.jit(fn)(qj, kj, vj)
    _, vjp = jax.vjp(lambda q, k, v: jax.jit(fn)(q, k, v), qj, kj, vj)
    dqj, dkj, dvj = vjp(jnp.asarray(gn).astype(jdt))

    def ref(q, k, v):
        return jnp.concatenate([attention(q[a:b][None], k[a:b][None], v[a:b][None],
            causal=causal, scale=scale, window=window)[0] for a, b in zip(cu[:-1], cu[1:])])
    ot, pullback = jax.vjp(ref, *(x.astype(jnp.float32) for x in (qj, kj, vj)))
    dqt, dkt, dvt = pullback(jnp.asarray(gn).astype(jdt).astype(jnp.float32))

    tol = 3e-2 if dtype == "bf16" else 1.5e-2

    def err(aj, bt):
        return float(np.max(np.abs(np.asarray(aj, np.float32) -
                                   np.asarray(bt, np.float32))))

    assert err(out_j, ot) < tol, "fwd"
    assert err(dqj, dqt) < tol * 2, "dq"
    assert err(dkj, dkt) < tol * 2, "dk"
    assert err(dvj, dvt) < tol * 2, "dv"


CASES = {
    "MHA-causal": ([200, 312, 128], 16, 16, 128, True, (-1, -1)),
    "MHA-noncausal": ([200, 312, 128], 16, 16, 128, False, (-1, -1)),
    "GQA-16-4-causal": ([256, 256], 16, 4, 128, True, (-1, -1)),
    "MQA-16-1": ([300, 212], 16, 1, 128, True, (-1, -1)),
    "window-128-0": ([256, 256], 16, 16, 128, False, (128, 0)),
    "headdim-64": ([200, 312], 16, 16, 64, True, (-1, -1)),
}


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("name", list(CASES))
def test_vs_flash_attn_varlen(name, dtype):
    seqlens, Hq, Hk, D, causal, window = CASES[name]
    _vs_reference(seqlens, Hq, Hk, D, causal, window, dtype)
