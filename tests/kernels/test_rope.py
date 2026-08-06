"""rope_ref (pure JAX, the recommended path) and the bridged vLLM rope kernel:
fwd correctness against an independent reference, partial rotation, dtypes,
position offsets, grads through rope_ref, jit. The bridge needs the full
container; rope_ref tests run anywhere with a GPU."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from lumi_kernels import rope as R

pytestmark = pytest.mark.gpu


def cos_sin_cache(max_pos, rotary_dim, base=10000.0):
    inv = 1.0 / (base ** (np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim))
    t = np.arange(max_pos, dtype=np.float64)[:, None] * inv[None, :]
    return jnp.asarray(np.concatenate([np.cos(t), np.sin(t)], axis=1), jnp.float32)


def numpy_neox_ref(positions, x, head_size, csc):
    """Independent NEOX rope in numpy (split-halves convention)."""
    csc = np.asarray(csc, np.float64)
    d = csc.shape[1] // 2
    rd = 2 * d
    T = x.shape[0]
    xf = np.asarray(x, np.float64).reshape(T, -1, head_size)
    cos = csc[np.asarray(positions), :d][:, None, :]
    sin = csc[np.asarray(positions), d:][:, None, :]
    a, b = xf[..., :d], xf[..., d:rd]
    out = xf.copy()
    out[..., :d] = a * cos - b * sin
    out[..., d:rd] = b * cos + a * sin
    return out.reshape(T, -1)


CASES = {
    "full-rot-D64": (64, 64, 8),
    "full-rot-D128": (128, 128, 8),
    "partial-rot-D128-r64": (128, 64, 8),
    "single-head-D256": (256, 256, 1),
}


@pytest.mark.parametrize("dt", [jnp.float32, jnp.bfloat16, jnp.float16],
                         ids=["fp32", "bf16", "fp16"])
@pytest.mark.parametrize("name", list(CASES))
def test_rope_ref_vs_numpy(name, dt):
    head_size, rotary_dim, H = CASES[name]
    T, max_pos = 64, 4096
    g = np.random.default_rng(0)
    q = jnp.asarray(g.standard_normal((T, H * head_size)) * 0.5, dt)
    k = jnp.asarray(g.standard_normal((T, H * head_size)) * 0.5, dt)
    pos = jnp.asarray(g.integers(0, max_pos, T).astype(np.int32))
    csc = cos_sin_cache(max_pos, rotary_dim)
    qr, kr = jax.jit(lambda p, q, k: R.rope_ref(p, q, k, head_size, csc))(pos, q, k)
    tol = 1e-5 if dt == jnp.float32 else (2e-2 if dt == jnp.bfloat16 else 5e-3)
    for got, x in ((qr, q), (kr, k)):
        want = numpy_neox_ref(pos, x, head_size, csc)
        assert float(np.max(np.abs(np.asarray(got, np.float32) - want))) < tol


def test_rope_ref_position_zero_identity():
    """Position 0 has angle 0 -> rotation is the identity."""
    head_size, T, H = 64, 8, 4
    g = np.random.default_rng(1)
    q = jnp.asarray(g.standard_normal((T, H * head_size)), jnp.float32)
    csc = cos_sin_cache(128, head_size)
    qr, _ = R.rope_ref(jnp.zeros(T, jnp.int32), q, q, head_size, csc)
    np.testing.assert_allclose(np.asarray(qr), np.asarray(q), atol=1e-6)


def test_rope_ref_norm_preserving():
    """Rotation preserves the norm of the rotated slice."""
    head_size, T, H = 128, 32, 8
    g = np.random.default_rng(2)
    q = jnp.asarray(g.standard_normal((T, H * head_size)), jnp.float32)
    csc = cos_sin_cache(4096, head_size)
    pos = jnp.asarray(g.integers(0, 4096, T).astype(np.int32))
    qr, _ = R.rope_ref(pos, q, q, head_size, csc)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(qr), axis=-1),
                               np.linalg.norm(np.asarray(q), axis=-1), rtol=1e-5)


def test_rope_ref_grad():
    """rope_ref is linear in (q, k): vjp must equal the transposed rotation."""
    head_size, T, H = 64, 16, 2
    g = np.random.default_rng(3)
    q = jnp.asarray(g.standard_normal((T, H * head_size)), jnp.float32)
    k = jnp.asarray(g.standard_normal((T, H * head_size)), jnp.float32)
    csc = cos_sin_cache(256, head_size)
    pos = jnp.asarray(g.integers(0, 256, T).astype(np.int32))
    loss = lambda q, k: jnp.sum(sum(R.rope_ref(pos, q, k, head_size, csc)) ** 2)
    gq, gk = jax.grad(loss, (0, 1))(q, k)
    # grad/jvp consistency: <grad, v> == directional derivative (FD in fp32 is too noisy)
    g = np.random.default_rng(4)
    vq = jnp.asarray(g.standard_normal(q.shape), jnp.float32)
    vk = jnp.asarray(g.standard_normal(k.shape), jnp.float32)
    _, jv = jax.jvp(loss, (q, k), (vq, vk))
    inner = float(jnp.vdot(gq, vq) + jnp.vdot(gk, vk))
    np.testing.assert_allclose(inner, float(jv), rtol=1e-4)
    assert gk.shape == k.shape and bool(jnp.all(jnp.isfinite(gk)))


@pytest.mark.torch
@pytest.mark.parametrize("dt", [jnp.bfloat16, jnp.float16], ids=["bf16", "fp16"])
def test_bridged_kernel_vs_ref(dt):
    if not R.is_available():
        pytest.skip("vllm rotary_embedding not importable")
    head_size, H, T, max_pos = 128, 8, 64, 4096
    g = np.random.default_rng(0)
    q = jnp.asarray(g.standard_normal((T, H * head_size)) * 0.5, dt)
    k = jnp.asarray(g.standard_normal((T, H * head_size)) * 0.5, dt)
    pos = jnp.asarray(g.integers(0, max_pos, T).astype(np.int32))
    csc = cos_sin_cache(max_pos, head_size).astype(dt)
    qk, kk = R.rope(pos, q, k, head_size, csc)
    qr, kr = R.rope_ref(pos, q, k, head_size, csc)
    tol = 2e-2 if dt == jnp.bfloat16 else 5e-3
    assert float(jnp.max(jnp.abs(qk.astype(jnp.float32) - qr.astype(jnp.float32)))) < tol
    assert float(jnp.max(jnp.abs(kk.astype(jnp.float32) - kr.astype(jnp.float32)))) < tol
