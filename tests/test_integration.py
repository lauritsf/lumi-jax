"""Cross-kernel integration: a Mamba-style block (causal_conv1d -> selective_scan)
end-to-end vs the pure-JAX reference chain, loss + gradients, under one jit."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels"))

import jax
import jax.numpy as jnp

pytestmark = [pytest.mark.gpu, pytest.mark.torch,
              pytest.mark.so("liblumi_conv1d.so"), pytest.mark.so("liblumi_sscan.so")]


def _params(Bsz=2, dim=64, L=256, N=16, width=4, seed=0):
    g = np.random.default_rng(seed)
    f = lambda *s: g.standard_normal(s).astype(np.float32)
    return dict(
        x=jnp.asarray(f(Bsz, dim, L) * 0.5),
        conv_w=jnp.asarray(f(dim, width) * 0.5),
        conv_b=jnp.asarray(f(dim) * 0.3),
        delta=jnp.asarray(np.abs(f(Bsz, dim, L)) * 0.5),
        A=jnp.asarray(-np.abs(f(dim, N)) - 0.5),
        B=jnp.asarray(f(Bsz, 1, N, L)),
        C=jnp.asarray(f(Bsz, 1, N, L)),
        D=jnp.asarray(np.abs(f(dim))),
    )


def _block(conv_fn, scan_fn, p):
    h = conv_fn(p["x"], p["conv_w"], p["conv_b"], silu=True)
    y = scan_fn(h, p["delta"], p["A"], p["B"], p["C"], p["D"], None, None,
                delta_softplus=True)
    return jnp.sum(y.astype(jnp.float32) ** 2)


def test_mamba_block_loss_and_grads():
    from lumi_kernels import conv1d as C
    from lumi_kernels import selective_scan as S
    from test_selective_scan import ref as scan_ref

    p = _params()
    loss_k = jax.jit(lambda x, w, b: _block(C.causal_conv1d, S.selective_scan,
                                            {**p, "x": x, "conv_w": w, "conv_b": b}))
    ref_scan = lambda u, delta, A, B, C_, D, z, db, delta_softplus: scan_ref(
        u, delta, A, B, C_, D, z, db, delta_softplus)
    loss_r = lambda x, w, b: _block(C.causal_conv1d_ref, ref_scan,
                                    {**p, "x": x, "conv_w": w, "conv_b": b})

    lk = float(loss_k(p["x"], p["conv_w"], p["conv_b"]))
    lr = float(loss_r(p["x"], p["conv_w"], p["conv_b"]))
    assert abs(lk - lr) / max(1.0, abs(lr)) < 5e-3, f"loss {lk} vs ref {lr}"

    gk = jax.jit(jax.grad(loss_k, (0, 1, 2)))(p["x"], p["conv_w"], p["conv_b"])
    gr = jax.grad(loss_r, (0, 1, 2))(p["x"], p["conv_w"], p["conv_b"])
    for nm, a, r in zip(("dx", "dconv_w", "dconv_b"), gk, gr):
        rel = float(jnp.max(jnp.abs(a - r))) / max(1.0, float(jnp.max(jnp.abs(r))))
        assert rel < 1e-2, f"{nm} rel-err={rel:.4f}"
