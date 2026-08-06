"""AOTriton forward FFI vs the (flash_attn-validated) CK forward FFI across
MHA / GQA / MQA, causal + non-causal, head dim. Forward only (AOTriton bwd is
weak; CK is the training path). Needs the `torch` image."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from lumi_kernels.attention import flash_ffi as F

pytestmark = [pytest.mark.gpu, pytest.mark.torch,
              pytest.mark.so("liblumi_aotriton.so"), pytest.mark.so("liblumi_fa_ffi.so")]

CASES = {
    "MHA-noncausal": (2, 512, 16, 16, 128, False),
    "MHA-causal": (2, 512, 16, 16, 128, True),
    "GQA-16-4-causal": (2, 512, 16, 4, 128, True),
    "MQA-16-1-causal": (2, 512, 16, 1, 128, True),
    "headdim64-causal": (2, 512, 16, 16, 64, True),
    "noncausal-GQA-S1024": (2, 1024, 16, 4, 128, False),
}


def _qkv(B, S, Hq, Hk, D, jdt, seed=1):
    ks = jax.random.split(jax.random.key(seed), 3)
    mk = lambda kk, h: (jax.random.normal(kk, (B, S, h, D), jnp.float32) * 0.5).astype(jdt)
    return mk(ks[0], Hq), mk(ks[1], Hk), mk(ks[2], Hk)


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("name", list(CASES))
def test_aotriton_vs_ck(name, dtype):
    B, S, Hq, Hk, D, causal = CASES[name]
    jdt = {"bf16": jnp.bfloat16, "fp16": jnp.float16}[dtype]
    q, k, v = _qkv(B, S, Hq, Hk, D, jdt)
    scale = 1.0 / D ** 0.5
    aot = jax.jit(lambda q, k, v: F.flash_attention_aotriton(q, k, v, causal=causal, scale=scale))(q, k, v)
    ck = jax.jit(lambda q, k, v: F.flash_attention_ffi(q, k, v, causal=causal, scale=scale))(q, k, v)
    err = float(np.max(np.abs(np.asarray(aot, np.float32) - np.asarray(ck, np.float32))))
    assert err < (3e-2 if dtype == "bf16" else 1.5e-2), f"{name}: {err:.4f}"


@pytest.mark.parametrize("causal", [False, True])
def test_hybrid(causal):
    """Hybrid = AOTriton fwd + CK bwd: forward must equal AOTriton, grads must equal CK."""
    q, k, v = _qkv(2, 512, 16, 16, 128, jnp.bfloat16)
    scale = 1.0 / 128 ** 0.5
    oh = jax.jit(lambda q, k, v: F.flash_attention_hybrid(q, k, v, causal=causal, scale=scale))(q, k, v)
    oa = jax.jit(lambda q, k, v: F.flash_attention_aotriton(q, k, v, causal=causal, scale=scale))(q, k, v)
    ef = float(np.max(np.abs(np.asarray(oh, np.float32) - np.asarray(oa, np.float32))))
    grad = lambda fn: jax.jit(jax.grad(lambda q, k, v: jnp.sum(
        fn(q, k, v, causal=causal, scale=scale).astype(jnp.float32)), (0, 1, 2)))
    gh = grad(F.flash_attention_hybrid)(q, k, v)
    gc = grad(F.flash_attention_ffi)(q, k, v)
    eg = max(float(np.max(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))))
             for a, b in zip(gh, gc))
    assert ef < 1e-3, f"fwd vs AOT: {ef:.4f}"
    assert eg < 3e-2, f"grad vs CK: {eg:.4f}"
