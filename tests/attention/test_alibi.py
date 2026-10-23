"""CK forward and gradients against an independent float32 JAX reference."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from lumi_kernels.attention import flash_ffi as F
from oracle import attention

pytestmark = [pytest.mark.gpu, pytest.mark.so("liblumi_fa_ffi.so")]


def _slopes(Hq, B=None, seed=9):
    s = (2.0 ** (-8.0 * (np.arange(1, Hq + 1) / Hq))).astype(np.float32)
    if B is not None:
        g = np.random.default_rng(seed)
        s = np.stack([s * float(f) for f in g.uniform(0.5, 1.5, B)]).astype(np.float32)
    return s


def _run(B, Sq, Sk, Hq, Hk, D, causal, window, per_batch, dtype):

    jdt = {"bf16": jnp.bfloat16, "fp16": jnp.float16}[dtype]
    g = np.random.default_rng(0)
    qn = (g.standard_normal((B, Sq, Hq, D)) * 0.5).astype(np.float32)
    kn = (g.standard_normal((B, Sk, Hk, D)) * 0.5).astype(np.float32)
    vn = (g.standard_normal((B, Sk, Hk, D)) * 0.5).astype(np.float32)
    gn = (g.standard_normal((B, Sq, Hq, D)) * 0.5).astype(np.float32)
    slopes = _slopes(Hq, B if per_batch else None)
    scale = 1.0 / D ** 0.5

    qj, kj, vj = (jnp.asarray(x).astype(jdt) for x in (qn, kn, vn))
    aj = jnp.asarray(slopes)
    fn = lambda q, k, v: F.flash_attention_ffi(q, k, v, causal=causal, scale=scale,
                                               window=window, alibi=aj)
    out_j = jax.jit(fn)(qj, kj, vj)
    _, vjp = jax.vjp(fn, qj, kj, vj)
    dqj, dkj, dvj = vjp(jnp.asarray(gn).astype(jdt))

    ref = lambda q, k, v: attention(q, k, v, causal=causal, scale=scale, window=window, alibi=aj)
    ot, pullback = jax.vjp(ref, *(x.astype(jnp.float32) for x in (qj, kj, vj)))
    dqt, dkt, dvt = pullback(jnp.asarray(gn).astype(jdt).astype(jnp.float32))

    def err(aj_, bt_):
        return float(np.max(np.abs(np.asarray(aj_, np.float32) -
                                   np.asarray(bt_, np.float32))))

    tol = 3e-2 if dtype == "bf16" else 1.5e-2
    assert err(out_j, ot) < tol, "fwd"
    for nm, a, b in [("dq", dqj, dqt), ("dk", dkj, dkt), ("dv", dvj, dvt)]:
        assert err(a, b) < tol * 2, nm


CASES = {
    "per-head-causal": (2, 512, 512, 8, 8, 128, True, (-1, -1), False),
    "per-head-noncausal": (2, 512, 512, 8, 8, 128, False, (-1, -1), False),
    "per-batch-causal": (2, 512, 512, 8, 8, 128, True, (-1, -1), True),
    "GQA-16-4": (2, 512, 512, 16, 4, 128, True, (-1, -1), False),
    "MQA-16-1": (2, 256, 256, 16, 1, 128, True, (-1, -1), False),
    "window-256-0": (2, 512, 512, 8, 8, 128, False, (256, 0), False),
    "cross-256-512": (2, 256, 512, 8, 8, 128, False, (-1, -1), False),
    "headdim-64": (2, 512, 512, 8, 8, 64, True, (-1, -1), False),
}


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("name", list(CASES))
def test_alibi(name, dtype):
    B, Sq, Sk, Hq, Hk, D, causal, window, per_batch = CASES[name]
    _run(B, Sq, Sk, Hq, Hk, D, causal, window, per_batch, dtype)


def test_alibi_effect_nonzero():
    """Sanity: slopes actually change the output vs no-alibi (not silently ignored)."""
    g = np.random.default_rng(0)
    q, k, v = (jnp.asarray(g.standard_normal((2, 512, 8, 128)) * .5, jnp.bfloat16) for _ in range(3))
    o0 = F.flash_attention_ffi(q, k, v, causal=True)
    o1 = F.flash_attention_ffi(q, k, v, causal=True, alibi=jnp.asarray(_slopes(8)))
    assert float(jnp.max(jnp.abs(o0.astype(jnp.float32) - o1.astype(jnp.float32)))) > 1e-3
