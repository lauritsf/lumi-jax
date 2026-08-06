"""Bridged vLLM fused MoE vs a pure-JAX MoE reference across expert counts /
top-k / shapes. Needs the `torch` image."""
import numpy as np
import pytest

import jax.numpy as jnp

from lumi_kernels import moe as M

pytestmark = [pytest.mark.gpu, pytest.mark.torch]

CASES = {
    "E8-k2": (512, 2048, 1024, 8, 2),
    "E16-k2": (256, 2048, 1408, 16, 2),
    "E32-k4": (256, 1024, 768, 32, 4),
    "E4-k1": (128, 1024, 512, 4, 1),
}


def make_inputs(T, H, I, E, k, jdt, seed=0):
    g = np.random.default_rng(seed)
    x = jnp.asarray(g.standard_normal((T, H)) * 0.2).astype(jdt)
    w1 = jnp.asarray(g.standard_normal((E, 2 * I, H)) * 0.02).astype(jdt)
    w2 = jnp.asarray(g.standard_normal((E, H, I)) * 0.02).astype(jdt)
    tw, ti = M.router_topk(jnp.asarray(g.standard_normal((T, E)), jnp.float32), k)
    return x, w1, w2, tw, ti


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("name", list(CASES))
def test_vs_ref(name, dtype):
    T, H, I, E, k = CASES[name]
    jdt = {"bf16": jnp.bfloat16, "fp16": jnp.float16}[dtype]
    x, w1, w2, tw, ti = make_inputs(T, H, I, E, k, jdt)
    out = M.moe(x, w1, w2, tw, ti)
    ref = M.moe_ref(x, w1, w2, tw, ti)
    rel = float(np.max(np.abs(np.asarray(out, np.float32) - np.asarray(ref, np.float32)))
                / (np.max(np.abs(np.asarray(ref, np.float32))) + 1e-6))
    assert rel < 0.05, f"rel-err vs pure-JAX = {rel:.4f}"
