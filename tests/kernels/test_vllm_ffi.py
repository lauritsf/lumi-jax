"""libtorch-in-FFI mechanism (vLLM op called from a jittable XLA custom call,
no dlpack bridge). Needs the `torch` image."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from lumi_kernels import vllm_ffi as V

pytestmark = [pytest.mark.gpu, pytest.mark.torch, pytest.mark.so("liblumi_vllm_ffi.so")]

TOL = {jnp.float32: 1e-3, jnp.float16: 2e-2, jnp.bfloat16: 1e-1}


@pytest.mark.parametrize("N,twod", [(1024, 4096), (333, 1024)])
@pytest.mark.parametrize("dt", [jnp.float32, jnp.float16, jnp.bfloat16],
                         ids=["fp32", "fp16", "bf16"])
def test_silu_and_mul(dt, N, twod):
    g = np.random.default_rng(0).standard_normal((N, twod)).astype(np.float32)
    x = jnp.asarray(g, dt)
    o = jax.jit(V.silu_and_mul)(x)
    r = jax.jit(V.silu_and_mul_ref)(x)
    err = float(jnp.max(jnp.abs(o.astype(jnp.float32) - r.astype(jnp.float32))))
    assert o.shape == (N, twod // 2)
    assert err < TOL[dt], f"max|err|={err:.4f}"


def test_composes_under_jit():
    """torch fully encapsulated in the custom call -> composes inside a larger jit."""
    @jax.jit
    def pipe(x):
        return jnp.sum(V.silu_and_mul(x).astype(jnp.float32))

    x = jnp.asarray(np.random.default_rng(1).standard_normal((512, 2048)), jnp.bfloat16)
    assert np.isfinite(float(pipe(x)))
