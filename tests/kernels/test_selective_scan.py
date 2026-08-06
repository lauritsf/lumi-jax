"""selective_scan FFI (mamba v2.3.2 kernels, gfx90a) vs a pure-JAX lax.scan
reference: fwd all dtypes / option combos, grads vs reference grads, long-L
(n_chunks > 1), jit. Torch-free: needs only liblumi_sscan.so + a GPU."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from lumi_kernels import selective_scan as S

pytestmark = [pytest.mark.gpu, pytest.mark.so("liblumi_sscan.so")]


def ref(u, delta, A, B, C, D=None, z=None, db=None, softplus=False):
    u, delta, A, B, C = (x.astype(jnp.float32) for x in (u, delta, A, B, C))
    Bsz, dim, L = u.shape
    G, N = B.shape[1], B.shape[2]
    if db is not None:
        delta = delta + db.astype(jnp.float32)[None, :, None]
    if softplus:
        delta = jax.nn.softplus(delta)
    rep = dim // G
    Be = jnp.repeat(B, rep, axis=1)              # (Bsz, dim, N, L)
    Ce = jnp.repeat(C, rep, axis=1)
    dA = jnp.exp(delta[:, :, None, :] * A[None, :, :, None])   # (Bsz,dim,N,L)
    dBu = delta[:, :, None, :] * Be * u[:, :, None, :]

    def step(h, t):
        h = dA[..., t] * h + dBu[..., t]
        return h, jnp.sum(Ce[..., t] * h, axis=-1)
    _, ys = jax.lax.scan(step, jnp.zeros((Bsz, dim, N), jnp.float32), jnp.arange(L))
    y = jnp.moveaxis(ys, 0, -1)                                 # (Bsz,dim,L)
    if D is not None:
        y = y + D.astype(jnp.float32)[None, :, None] * u
    if z is not None:
        zf = z.astype(jnp.float32)
        y = y * (zf * jax.nn.sigmoid(zf))
    return y


def run_case(Bsz, dim, L, N, G, dt, *, use_d=True, use_z=True, use_db=True,
             softplus=True, grad=False):
    rng = np.random.default_rng(0)
    f = lambda *s: rng.standard_normal(s).astype(np.float32) * 0.5
    u = jnp.asarray(f(Bsz, dim, L), dt)
    delta = jnp.asarray(np.abs(f(Bsz, dim, L)) * 0.5, dt)
    A = jnp.asarray(-np.abs(f(dim, N)) - 0.5)
    Bv = jnp.asarray(f(Bsz, G, N, L), dt)
    Cv = jnp.asarray(f(Bsz, G, N, L), dt)
    D = jnp.asarray(np.abs(f(dim))) if use_d else None
    z = jnp.asarray(f(Bsz, dim, L), dt) if use_z else None
    db = jnp.asarray(f(dim) * 0.1) if use_db else None

    out = jax.jit(S.selective_scan, static_argnames=("delta_softplus",))(
        u, delta, A, Bv, Cv, D, z, db, delta_softplus=softplus)
    r = ref(u, delta, A, Bv, Cv, D, z, db, softplus)
    tol = 5e-4 if dt == jnp.float32 else (0.05 if dt == jnp.bfloat16 else 0.01)
    e = float(jnp.max(jnp.abs(out.astype(jnp.float32) - r)))
    assert e < tol, f"fwd err={e:.5f}"

    if grad:
        args = (u, delta, A, Bv, Cv) + ((D,) if use_d else ()) + ((z,) if use_z else ())
        iD = 5 if use_d else None
        iZ = (6 if use_d else 5) if use_z else None
        loss_f = lambda *a: (S.selective_scan(
            a[0], a[1], a[2], a[3], a[4], a[iD] if use_d else None,
            a[iZ] if use_z else None, db, delta_softplus=softplus)
            .astype(jnp.float32) ** 2).sum()
        loss_r = lambda *a: (ref(a[0], a[1], a[2], a[3], a[4], a[iD] if use_d else None,
                                 a[iZ] if use_z else None, db, softplus) ** 2).sum()
        gf = jax.jit(jax.grad(loss_f, argnums=tuple(range(len(args)))))(*args)
        gr = jax.grad(loss_r, argnums=tuple(range(len(args))))(*args)
        ge = max(float(jnp.max(jnp.abs(a.astype(jnp.float32) - b))) /
                 max(1.0, float(jnp.max(jnp.abs(b)))) for a, b in zip(gf, gr))
        # 5e-3 rel fp32: kernel recomputes fp32 atomically; mamba's own tests use 1e-2
        gtol = 5e-3 if dt == jnp.float32 else 0.05
        assert ge < gtol, f"grad rel-err={ge:.5f}"


CASES = {
    "fp32-full": dict(Bsz=2, dim=64, L=512, N=16, G=1, dt=jnp.float32, grad=True),
    "fp32-G2": dict(Bsz=2, dim=64, L=512, N=16, G=2, dt=jnp.float32, grad=True),
    "fp32-no-z": dict(Bsz=2, dim=64, L=512, N=16, G=1, dt=jnp.float32, use_z=False, grad=True),
    "fp32-no-D-no-bias": dict(Bsz=2, dim=64, L=512, N=16, G=1, dt=jnp.float32,
                              use_d=False, use_db=False, grad=True),
    "fp32-no-softplus": dict(Bsz=2, dim=64, L=256, N=16, G=1, dt=jnp.float32, softplus=False),
    "fp32-L4096-chunks": dict(Bsz=2, dim=32, L=4096, N=16, G=1, dt=jnp.float32, grad=True),
    "fp32-N64": dict(Bsz=2, dim=32, L=512, N=64, G=1, dt=jnp.float32),
    "bf16-full": dict(Bsz=2, dim=64, L=512, N=16, G=1, dt=jnp.bfloat16),
    "bf16-L4096": dict(Bsz=2, dim=32, L=4096, N=16, G=1, dt=jnp.bfloat16),
    "fp16-full": dict(Bsz=2, dim=64, L=512, N=16, G=1, dt=jnp.float16),
}


@pytest.mark.parametrize("name", list(CASES))
def test_selective_scan(name):
    run_case(**CASES[name])
