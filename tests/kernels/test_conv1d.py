"""causal_conv1d JAX binding (native fwd + native bwd, both FFI) vs the
container's torch causal_conv1d_fn. Needs the `torch` image."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from lumi_kernels import conv1d as C

pytestmark = [pytest.mark.gpu, pytest.mark.torch, pytest.mark.so("liblumi_conv1d.so")]

CASES = {
    "B2-d128-L256-w4-silu-bias": (2, 128, 256, 4, True, True),
    "B2-d128-L256-w4-silu-nobias": (2, 128, 256, 4, True, False),
    "B2-d256-L512-w4-nosilu-bias": (2, 256, 512, 4, False, True),
    "B4-d512-L1024-w3-silu-bias": (4, 512, 1024, 3, True, True),
}


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("name", list(CASES))
def test_vs_torch(name, dtype):
    import torch
    from causal_conv1d import causal_conv1d_fn

    B, dim, L, width, silu, has_bias = CASES[name]
    jdt = {"bf16": jnp.bfloat16, "fp16": jnp.float16}[dtype]
    tdt = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    g = np.random.default_rng(0)
    xn = (g.standard_normal((B, dim, L)) * 0.5).astype(np.float32)
    wn = (g.standard_normal((dim, width)) * 0.5).astype(np.float32)
    bn = (g.standard_normal((dim,)) * 0.3).astype(np.float32) if has_bias else None
    gn = (g.standard_normal((B, dim, L)) * 0.5).astype(np.float32)

    xj, wj = jnp.asarray(xn).astype(jdt), jnp.asarray(wn).astype(jdt)
    bj = jnp.asarray(bn).astype(jdt) if has_bias else None
    oj = jax.jit(lambda x, w: C.causal_conv1d(x, w, bj, silu=silu))(xj, wj)
    if has_bias:
        _, vjp = jax.vjp(lambda x, w, b: C.causal_conv1d(x, w, b, silu=silu), xj, wj, bj)
        dxj, dwj, dbj = vjp(jnp.asarray(gn).astype(jdt))
    else:
        _, vjp = jax.vjp(lambda x, w: C.causal_conv1d(x, w, None, silu=silu), xj, wj)
        dxj, dwj = vjp(jnp.asarray(gn).astype(jdt))
        dbj = None

    xt = torch.tensor(xn, device="cuda", dtype=tdt, requires_grad=True)
    wt = torch.tensor(wn, device="cuda", dtype=tdt, requires_grad=True)
    bt = torch.tensor(bn, device="cuda", dtype=tdt, requires_grad=True) if has_bias else None
    ot = causal_conv1d_fn(xt, wt, bias=bt, activation="silu" if silu else None)
    grads = torch.autograd.grad(ot, (xt, wt) + ((bt,) if has_bias else ()),
                                torch.tensor(gn, device="cuda", dtype=tdt))

    def err(aj, bt_):
        return float(np.max(np.abs(np.asarray(aj, np.float32) - bt_.detach().float().cpu().numpy())))

    tol = 3e-2 if dtype == "bf16" else 1.5e-2
    assert err(oj, ot) < tol, "fwd"
    assert err(dxj, grads[0]) < tol * 2, "dx"
    assert err(dwj, grads[1]) < tol * 2, "dweight"
    if has_bias:
        assert err(dbj, grads[2]) < tol * 2, "dbias"
