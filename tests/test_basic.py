"""Single-GPU correctness, JIT, grad, mixed precision."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

pytestmark = pytest.mark.gpu


def test_rocm_device():
    assert any("Rocm" in repr(d) for d in jax.devices()), jax.devices()


def test_matmul_matches_numpy():
    key = jax.random.key(0)
    A = jax.random.normal(key, (512, 512))
    B = jax.random.normal(key, (512, 512))
    np.testing.assert_allclose(np.asarray(jnp.dot(A, B)), np.asarray(A) @ np.asarray(B),
                               rtol=1e-4, atol=1e-4)


def test_jit_grad_finite_difference():
    @jax.jit
    def loss(W, x, y):
        return jnp.mean((x @ W - y) ** 2)

    key = jax.random.key(0)
    W = jax.random.normal(key, (32, 8))
    x = jax.random.normal(key, (128, 32))
    y = jax.random.normal(key, (128, 8))
    g = jax.grad(loss)(W, x, y)
    assert g.shape == W.shape

    eps, i, j = 1e-3, 5, 3
    fd = (loss(W.at[i, j].add(eps), x, y) - loss(W.at[i, j].add(-eps), x, y)) / (2 * eps)
    np.testing.assert_allclose(float(g[i, j]), float(fd), rtol=1e-2)


def test_bf16_matmul_dtype():
    key = jax.random.key(0)
    x = jax.random.normal(key, (128, 32)).astype(jnp.bfloat16)
    W = jax.random.normal(key, (32, 8)).astype(jnp.bfloat16)
    assert jnp.dot(x, W).dtype == jnp.bfloat16
