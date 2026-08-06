"""Validate the full-feature XLA-FFI binding against flash_attn_func (the
authoritative torch binding of the same CK kernels) across the capability matrix:
MHA / single-head / GQA / MQA, causal + non-causal, cross-attention (S_q != S_k),
sliding-window/local, fp16 + bf16 -- forward and gradients. Needs the `torch` image.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from lumi_kernels.attention import flash_ffi as F

pytestmark = [pytest.mark.gpu, pytest.mark.torch, pytest.mark.so("liblumi_fa_ffi.so")]

JDT = {"bf16": jnp.bfloat16, "fp16": jnp.float16}


def _np(B, S, H, D, seed, dt=np.float32):
    g = np.random.default_rng(seed)
    return (g.standard_normal((B, S, H, D)).astype(np.float32) * 0.5).astype(dt)


def _vs_torch(B, Sq, Sk, Hq, Hk, D, causal, window, dtype, grad=True):
    import torch
    from flash_attn import flash_attn_func

    tdt = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    jdt = JDT[dtype]
    qn, kn, vn, gn = _np(B, Sq, Hq, D, 1), _np(B, Sk, Hk, D, 2), _np(B, Sk, Hk, D, 3), _np(B, Sq, Hq, D, 4)
    scale = 1.0 / D ** 0.5

    qj, kj, vj = (jnp.asarray(x).astype(jdt) for x in (qn, kn, vn))
    fn = lambda q, k, v: F.flash_attention_ffi(q, k, v, causal=causal, scale=scale, window=window)
    out_j = jax.jit(fn)(qj, kj, vj)
    if grad:
        _, vjp = jax.vjp(lambda q, k, v: jax.jit(fn)(q, k, v), qj, kj, vj)
        dqj, dkj, dvj = vjp(jnp.asarray(gn).astype(jdt))

    qt, kt, vt = (torch.tensor(x, device="cuda", dtype=tdt, requires_grad=grad) for x in (qn, kn, vn))
    ot = flash_attn_func(qt, kt, vt, softmax_scale=scale, causal=causal, window_size=window)
    if grad:
        dqt, dkt, dvt = torch.autograd.grad(ot, (qt, kt, vt), torch.tensor(gn, device="cuda", dtype=tdt))

    def err(a_jax, b_torch):
        a = np.asarray(a_jax, np.float32)
        b = b_torch.detach().to(torch.float32).cpu().numpy()
        return float(np.max(np.abs(a - b)))

    tol = 3e-2 if dtype == "bf16" else 1.5e-2
    assert err(out_j, ot) < tol, "fwd"
    if grad:
        for nm, aj, bt in [("dq", dqj, dqt), ("dk", dkj, dkt), ("dv", dvj, dvt)]:
            assert err(aj, bt) < tol * 2, nm


CASES = {
    "MHA-causal": (2, 512, 512, 16, 16, 128, True, (-1, -1)),
    "MHA-noncausal": (2, 512, 512, 16, 16, 128, False, (-1, -1)),
    "single-head": (2, 512, 512, 1, 1, 128, True, (-1, -1)),
    "GQA-16-4": (2, 512, 512, 16, 4, 128, True, (-1, -1)),
    "MQA-16-1": (2, 512, 512, 16, 1, 128, True, (-1, -1)),
    "cross-256-512": (2, 256, 512, 16, 16, 128, False, (-1, -1)),
    "window-256-0": (2, 512, 512, 16, 16, 128, False, (256, 0)),
    "window-128-128": (2, 512, 512, 16, 16, 128, False, (128, 128)),
    "headdim-64": (2, 512, 512, 16, 16, 64, True, (-1, -1)),
}


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("name", list(CASES))
def test_vs_flash_attn(name, dtype):
    B, Sq, Sk, Hq, Hk, D, causal, window = CASES[name]
    _vs_torch(B, Sq, Sk, Hq, Hk, D, causal, window, dtype)


def _qkv(B=2, S=512, H=8, D=64, dt=jnp.bfloat16):
    return tuple(jnp.asarray(_np(B, S, H, D, i) * 0.5, dt) for i in (1, 2, 3))


def test_dropout_per_step_jit():
    """(seed,offset) is a uint64[2] DEVICE operand: per-step seeds, no recompile."""
    p = 0.5
    q, k, v = _qkv()
    fwd = jax.jit(lambda q, k, v, s: F.flash_attention_ffi(q, k, v, dropout_p=p, dropout_seed=s))
    s1 = jnp.array([7, 0], jnp.uint64)
    s2 = jnp.array([1234, 0], jnp.uint64)
    o1a, o1b, o2 = fwd(q, k, v, s1), fwd(q, k, v, s1), fwd(q, k, v, s2)
    assert bool(jnp.all(o1a == o1b)), "same seed not deterministic"
    assert float(jnp.max(jnp.abs(o1a - o2))) > 1e-3, "seed does not vary output"
    assert fwd._cache_size() == 1, f"recompiled: cache={fwd._cache_size()}"

    o0 = jax.jit(lambda q, k, v: F.flash_attention_ffi(q, k, v))(q, k, v)
    m0, m1 = float(jnp.mean(o0.astype(jnp.float32))), float(jnp.mean(o1a.astype(jnp.float32)))
    assert abs(m1 - m0) < 0.05, f"dropout not ~mean-preserving: {m1:.4f} vs {m0:.4f}"


def test_dropout_grad_deterministic():
    p = 0.5
    q, k, v = _qkv()
    s1 = jnp.array([7, 0], jnp.uint64)
    s2 = jnp.array([1234, 0], jnp.uint64)
    gfn = jax.jit(jax.grad(lambda q, k, v, s: F.flash_attention_ffi(
        q, k, v, dropout_p=p, dropout_seed=s,
        deterministic=True).astype(jnp.float32).sum(), argnums=(0, 1, 2)))
    g1a, g1b, g2 = gfn(q, k, v, s1), gfn(q, k, v, s1), gfn(q, k, v, s2)
    assert all(bool(jnp.all(a == b)) for a, b in zip(g1a, g1b))
    assert float(jnp.max(jnp.abs(g1a[0] - g2[0]))) > 1e-3
    assert all(bool(jnp.all(jnp.isfinite(g.astype(jnp.float32)))) for g in g1a)
    assert gfn._cache_size() == 1


def test_dropout_varlen_traced_seed():
    B, S, H, D, p = 2, 512, 8, 64, 0.5
    q, k, v = _qkv(B, S, H, D)
    total = B * S
    qp, kp, vp = (x.reshape(total, H, D) for x in (q, k, v))
    cu = jnp.asarray([0, S, total], jnp.int32)
    s1 = jnp.array([7, 0], jnp.uint64)
    s2 = jnp.array([1234, 0], jnp.uint64)
    vfn = jax.jit(lambda q, k, v, cu, s: F.flash_attention_varlen_traced(
        q, k, v, cu, cu, S, S, dropout_p=p, dropout_seed=s))
    vo1a, vo1b, vo2 = vfn(qp, kp, vp, cu, s1), vfn(qp, kp, vp, cu, s1), vfn(qp, kp, vp, cu, s2)
    assert bool(jnp.all(vo1a == vo1b))
    assert float(jnp.max(jnp.abs(vo1a - vo2))) > 1e-3
    assert vfn._cache_size() == 1
