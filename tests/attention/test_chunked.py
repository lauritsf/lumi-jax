"""Correctness of memory-efficient attention paths vs the explicit jnp reference:
chunked (pure JAX) and the CK FFI
(covered exhaustively in test_flash_ffi.py; spot-checked vs the naive ref here)."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from lumi_kernels.attention import reference, chunked

pytestmark = pytest.mark.gpu


def _inputs(B=2, S=512, H=4, D=64, dtype=jnp.float32, seed=0):
    ks = jax.random.split(jax.random.key(seed), 3)
    mk = lambda kk: (jax.random.normal(kk, (B, S, H, D), jnp.float32) * 0.5).astype(dtype)
    return mk(ks[0]), mk(ks[1]), mk(ks[2])


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype,atol", [(jnp.float32, 2e-4), (jnp.bfloat16, 2e-2)])
def test_chunked_fwd(causal, dtype, atol):
    q, k, v = _inputs(dtype=dtype)
    ref = reference.naive_reference(q, k, v, causal=causal)
    got = chunked.chunked_attention(q, k, v, causal=causal)
    assert got.shape == ref.shape
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(ref, np.float32),
                               atol=atol, rtol=atol)


def test_chunked_fwd_nonpow2_seqlen():
    """Non-square / non-power-of-two seqlen exercises the divisor fallback."""
    q, k, v = _inputs(S=384)
    ref = reference.naive_reference(q, k, v, causal=True)
    got = chunked.chunked_attention(q, k, v, causal=True)
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(ref, np.float32),
                               atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("causal", [False, True])
def test_chunked_grad(causal):
    q, k, v = _inputs(S=256)
    loss = lambda fn: lambda q, k, v: jnp.sum(fn(q, k, v).astype(jnp.float32))
    gref = jax.grad(loss(lambda q, k, v: reference.naive_reference(q, k, v, causal=causal)), (0, 1, 2))(q, k, v)
    gchk = jax.grad(loss(lambda q, k, v: chunked.chunked_attention(q, k, v, causal=causal)), (0, 1, 2))(q, k, v)
    for name, a, b in zip("qkv", gref, gchk):
        np.testing.assert_allclose(np.asarray(b, np.float32), np.asarray(a, np.float32),
                                   atol=2e-3, rtol=2e-3, err_msg=f"grad d{name}")


def test_chunked_remat_matches():
    """remat=True must produce identical forward + gradients to remat=False."""
    q, k, v = _inputs(S=512)
    f0 = chunked.chunked_attention(q, k, v, causal=True)
    f1 = chunked.chunked_attention(q, k, v, causal=True, remat=True)
    np.testing.assert_allclose(np.asarray(f0), np.asarray(f1), atol=1e-6, rtol=1e-6)
    g0 = jax.grad(lambda *a: jnp.sum(chunked.chunked_attention(*a, causal=True)), (0, 1, 2))(q, k, v)
    g1 = jax.grad(lambda *a: jnp.sum(chunked.chunked_attention(*a, causal=True, remat=True)), (0, 1, 2))(q, k, v)
    for a, b in zip(g0, g1):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6, rtol=1e-6)


@pytest.mark.so("liblumi_fa_ffi.so")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("kv_heads", [8, 2, 1])
def test_ffi_vs_naive_ref(causal, dtype, kv_heads):
    """Spot-check the FFI against the naive jnp reference (independent of flash_attn)."""
    from lumi_kernels.attention import flash_ffi as ffi

    q, k, v = _inputs(S=512, H=8, D=128, dtype=dtype)
    k, v = k[:, :, :kv_heads], v[:, :, :kv_heads]
    cotangent = _inputs(S=512, H=8, D=128, dtype=dtype, seed=1)[0]
    scale = 1.0 / (128 ** 0.5)
    fn = jax.jit(lambda q, k, v: ffi.flash_attention_ffi(q, k, v, causal=causal, scale=scale))
    def oracle(q, k, v):
        repeats = q.shape[2] // k.shape[2]
        return reference.naive_reference(q, jnp.repeat(k, repeats, axis=2),
                                         jnp.repeat(v, repeats, axis=2), causal=causal)
    ref = oracle(q, k, v)
    atol, rtol = (2e-3, 3e-2) if dtype == jnp.bfloat16 else (3e-4, 5e-3)
    np.testing.assert_allclose(np.asarray(fn(q, k, v), np.float32),
                               np.asarray(ref, np.float32), atol=atol, rtol=rtol)
    g = jax.jit(jax.grad(lambda q, k, v: jnp.sum(
        ffi.flash_attention_ffi(q, k, v, causal=causal, scale=scale).astype(jnp.float32)
        * cotangent.astype(jnp.float32)),
        (0, 1, 2)))(q, k, v)
    # Keep grouped-head gradient reductions in FP32 before storage rounding.
    gref = jax.grad(lambda q, k, v: jnp.sum(
        oracle(q, k, v).astype(jnp.float32) * cotangent.astype(jnp.float32)),
        (0, 1, 2))(*(x.astype(jnp.float32) for x in (q, k, v)))
    for nm, a, b in zip("qkv", g, gref):
        actual, expected = np.asarray(a, np.float32), np.asarray(b, np.float32)
        # Grouped-head cancellation makes elementwise relative error unstable.
        relative_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        assert relative_l2 <= rtol, f"d{nm}: relative L2 error {relative_l2} > {rtol}"
        np.testing.assert_allclose(actual, expected, rtol=0,
                                   atol=atol + rtol * np.max(np.abs(expected)),
                                   err_msg=f"d{nm} maximum error")
