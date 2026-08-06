"""Validate the varlen / packed XLA-FFI binding vs flash_attn_varlen_func across
MHA / GQA, causal + non-causal, and a local window -- forward and gradients.
Needs the `torch` image.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from lumi_kernels.attention import flash_ffi as F

pytestmark = [pytest.mark.gpu, pytest.mark.torch, pytest.mark.so("liblumi_fa_ffi.so")]


def _vs_torch(seqlens, Hq, Hk, D, causal, window, dtype, max_pad=0):
    import torch
    from flash_attn import flash_attn_varlen_func

    jdt = {"bf16": jnp.bfloat16, "fp16": jnp.float16}[dtype]
    tdt = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
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

    qt, kt, vt = (torch.tensor(x, device="cuda", dtype=tdt, requires_grad=True) for x in (qn, kn, vn))
    cut = torch.tensor(cu, device="cuda")
    ot = flash_attn_varlen_func(qt, kt, vt, cut, cut, max_s, max_s,
                                softmax_scale=scale, causal=causal, window_size=window)
    dqt, dkt, dvt = torch.autograd.grad(ot, (qt, kt, vt), torch.tensor(gn, device="cuda", dtype=tdt))

    tol = 3e-2 if dtype == "bf16" else 1.5e-2

    def err(aj, bt):
        return float(np.max(np.abs(np.asarray(aj, np.float32) -
                                   bt.detach().to(torch.float32).cpu().numpy())))

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
    _vs_torch(seqlens, Hq, Hk, D, causal, window, dtype)
