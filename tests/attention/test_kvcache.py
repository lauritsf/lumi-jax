"""flash_attn_kvcache_ffi (CK split-KV decode, RAW torch-free FFI, jittable) vs a
pure-JAX full-attention reference over the valid cache, and vs torch's
flash_attn_with_kvcache (the same kernel). Needs the `torch` image."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from lumi_kernels.attention import flash_ffi as F

pytestmark = [pytest.mark.gpu, pytest.mark.torch, pytest.mark.so("liblumi_fa_ffi.so")]


def ref(q, kc, vc, seqlens, scale):
    B, Sq, Hq, D = q.shape
    Hk = kc.shape[2]
    g = Hq // Hk
    outs = []
    for b in range(B):
        L = int(seqlens[b])
        K = jnp.repeat(kc[b, :L].astype(jnp.float32), g, axis=1)  # (L, Hq, D)
        V = jnp.repeat(vc[b, :L].astype(jnp.float32), g, axis=1)
        qb = q[b].astype(jnp.float32)                             # (Sq, Hq, D)
        s = jnp.einsum("qhd,khd->hqk", qb, K) * scale
        p = jax.nn.softmax(s, axis=-1)
        outs.append(jnp.einsum("hqk,khd->qhd", p, V))
    return jnp.stack(outs)


def rope_ref(x, cos, sin, pos, interleaved):
    """Reference rope. x (B,S,H,D); cos/sin (seqlen_ro, rdim/2); pos (B,) start offsets."""
    B, S, H, D = x.shape
    r2 = cos.shape[1]
    idx = pos[:, None] + jnp.arange(S)[None, :]               # (B,S)
    c = cos[idx][:, :, None, :].astype(jnp.float32)           # (B,S,1,r2)
    s = sin[idx][:, :, None, :].astype(jnp.float32)
    xf = x.astype(jnp.float32)
    if interleaved:
        ev, od = xf[..., 0:2 * r2:2], xf[..., 1:2 * r2:2]
        rot = jnp.stack([ev * c - od * s, od * c + ev * s], axis=-1).reshape(B, S, H, 2 * r2)
    else:
        x1, x2 = xf[..., :r2], xf[..., r2:2 * r2]
        rot = jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)
    return jnp.concatenate([rot, xf[..., 2 * r2:]], axis=-1).astype(x.dtype)


def ref_step(q, kc, vc, sl, knew, vnew, scale, causal):
    """Full-attention reference for the step: cache[:sl] ++ knew, bottom-right causal."""
    B, Sq, Hq, D = q.shape
    Hk = kc.shape[2]
    g = Hq // Hk
    outs = []
    for b in range(B):
        L = int(sl[b])
        K = jnp.concatenate([kc[b, :L], knew[b]]) if knew is not None else kc[b, :L]
        V = jnp.concatenate([vc[b, :L], vnew[b]]) if vnew is not None else vc[b, :L]
        K = jnp.repeat(K.astype(jnp.float32), g, axis=1)
        V = jnp.repeat(V.astype(jnp.float32), g, axis=1)
        s = jnp.einsum("qhd,khd->hqk", q[b].astype(jnp.float32), K) * scale
        if causal and Sq > 1:
            T = K.shape[0]
            m = jnp.arange(Sq)[:, None] + (T - Sq) >= jnp.arange(T)[None, :]
            s = jnp.where(m[None], s, -jnp.inf)
        outs.append(jnp.einsum("hqk,khd->qhd", jax.nn.softmax(s, axis=-1), V))
    return jnp.stack(outs)


DECODE = {
    "MHA-Sq1-bf16": (4, 1, 8, 8, 128, 2048, jnp.bfloat16),
    "GQA-16-4-bf16": (4, 1, 16, 4, 128, 1024, jnp.bfloat16),
    "MQA-16-1-fp16": (2, 1, 16, 1, 128, 4096, jnp.float16),
    "headdim64-bf16": (3, 1, 8, 8, 64, 512, jnp.bfloat16),
    "chunked-Sq4-bf16": (2, 4, 8, 8, 128, 1024, jnp.bfloat16),
}


@pytest.mark.parametrize("name", list(DECODE))
def test_decode(name):
    B, Sq, Hq, Hk, D, Sk, dt = DECODE[name]
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal((B, Sq, Hq, D)) * .5, dt)
    kc = jnp.asarray(rng.standard_normal((B, Sk, Hk, D)) * .5, dt)
    vc = jnp.asarray(rng.standard_normal((B, Sk, Hk, D)) * .5, dt)
    sl = jnp.asarray(rng.integers(Sk // 2, Sk, size=B).astype(np.int32))
    scale = 1.0 / D ** 0.5

    o = jax.jit(F.flash_attn_kvcache_ffi, static_argnames=("causal", "scale"))(
        q, kc, vc, sl, causal=False, scale=scale)
    e = float(jnp.max(jnp.abs(o.astype(jnp.float32) - ref(q, kc, vc, sl, scale).astype(jnp.float32))))
    tol = 6e-2 if dt == jnp.bfloat16 else 2e-2
    assert e < tol, f"vs JAX ref: {e:.4f}"

    import torch
    from flash_attn import flash_attn_with_kvcache
    tdt = torch.bfloat16 if dt == jnp.bfloat16 else torch.float16
    tq, tkc, tvc = (torch.tensor(np.asarray(x, np.float32), device="cuda", dtype=tdt)
                    for x in (q, kc, vc))
    tsl = torch.tensor(np.asarray(sl), device="cuda")
    to = flash_attn_with_kvcache(tq, tkc, tvc, cache_seqlens=tsl, softmax_scale=scale, causal=False)
    et = float(np.max(np.abs(np.asarray(o, np.float32) - to.float().cpu().numpy())))
    assert et < tol, f"vs torch kvcache: {et:.4f}"


def _step_case(B, Sq, Hq, Hk, D, Sk, Snew, dt, *, rotary=False, interleaved=False,
               causal=False, paged=False, torch_check=False, seqlens=None):
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.standard_normal((B, Sq, Hq, D)) * .5, dt)
    kc = jnp.asarray(rng.standard_normal((B, Sk, Hk, D)) * .5, dt)
    vc = jnp.asarray(rng.standard_normal((B, Sk, Hk, D)) * .5, dt)
    kn = jnp.asarray(rng.standard_normal((B, Snew, Hk, D)) * .5, dt) if Snew else None
    vn = jnp.asarray(rng.standard_normal((B, Snew, Hk, D)) * .5, dt) if Snew else None
    if seqlens is None:
        sl = jnp.asarray(rng.integers(Sk // 4, Sk - Snew - Sq, size=B).astype(np.int32))
    else:
        sl = jnp.asarray(np.asarray(seqlens, np.int32))
    scale = 1.0 / D ** 0.5
    rc = rs = None
    if rotary:
        rdim = D if D % 16 == 0 else (D // 16) * 16
        inv = 1.0 / (10000.0 ** (np.arange(0, rdim, 2) / rdim))
        t = np.arange(Sk)[:, None] * inv[None, :]
        rc, rs = jnp.asarray(np.cos(t), dt), jnp.asarray(np.sin(t), dt)

    if paged:
        pbs = 128
        nb = Sk // pbs
        bt = jnp.arange(B * nb, dtype=jnp.int32).reshape(B, nb)
        kcp = kc.reshape(B * nb, pbs, Hk, D)
        vcp = vc.reshape(B * nb, pbs, Hk, D)
        out, kcu, vcu = F.flash_attn_kvcache_step(
            q, kcp, vcp, sl, knew=kn, vnew=vn, rotary_cos=rc, rotary_sin=rs,
            rotary_interleaved=interleaved, block_table=bt, causal=causal, scale=scale)
        kcu, vcu = kcu.reshape(B, Sk, Hk, D), vcu.reshape(B, Sk, Hk, D)
    else:
        out, kcu, vcu = F.flash_attn_kvcache_step(
            q, kc, vc, sl, knew=kn, vnew=vn, rotary_cos=rc, rotary_sin=rs,
            rotary_interleaved=interleaved, causal=causal, scale=scale)

    qr = rope_ref(q, rc, rs, sl, interleaved) if rotary else q  # q tokens start at sl
    knr = rope_ref(kn, rc, rs, sl, interleaved) if (rotary and Snew) else kn
    r = ref_step(qr, kc, vc, sl, knr, vn, scale, causal)
    e = float(jnp.max(jnp.abs(out.astype(jnp.float32) - r)))
    tol = 6e-2 if dt == jnp.bfloat16 else 2e-2
    assert e < tol, f"step out err {e:.4f}"

    if Snew:  # appended cache contents (and untouched prefix)
        for b in range(B):
            L = int(sl[b])
            ek = float(jnp.max(jnp.abs(
                kcu[b, L:L + Snew].astype(jnp.float32) - (knr if knr is not None else kn)[b].astype(jnp.float32))))
            ek = max(ek, float(jnp.max(jnp.abs(vcu[b, L:L + Snew].astype(jnp.float32) - vn[b].astype(jnp.float32)))))
            ek = max(ek, float(jnp.max(jnp.abs(kcu[b, :L].astype(jnp.float32) - kc[b, :L].astype(jnp.float32)))))
            assert ek < tol, f"cache contents b={b} err {ek:.4f}"

    if torch_check:
        import torch
        from flash_attn import flash_attn_with_kvcache
        tdt = torch.bfloat16 if dt == jnp.bfloat16 else torch.float16
        tt = lambda x: (torch.tensor(np.asarray(x, np.float32), device="cuda", dtype=tdt)
                        if x.dtype == dt else torch.tensor(np.asarray(x), device="cuda"))
        to = flash_attn_with_kvcache(
            tt(q), tt(kc), tt(vc), k=tt(kn) if kn is not None else None,
            v=tt(vn) if vn is not None else None, cache_seqlens=tt(sl),
            rotary_cos=tt(rc) if rc is not None else None,
            rotary_sin=tt(rs) if rs is not None else None,
            rotary_interleaved=interleaved, softmax_scale=scale, causal=causal)
        et = float(np.max(np.abs(np.asarray(out, np.float32) - to.float().cpu().numpy())))
        assert et < tol, f"vs torch step: {et:.4f}"


STEP = {
    "append-Sq1-bf16": dict(B=4, Sq=1, Hq=8, Hk=8, D=128, Sk=2048, Snew=1, dt=jnp.bfloat16, torch_check=True),
    "append-chunk-Sq4-GQA-fp16": dict(B=2, Sq=4, Hq=16, Hk=4, D=128, Sk=1024, Snew=4, dt=jnp.float16,
                                      causal=True, torch_check=True),
    "rope-half-bf16": dict(B=4, Sq=1, Hq=8, Hk=8, D=128, Sk=1024, Snew=1, dt=jnp.bfloat16,
                           rotary=True, torch_check=True),
    "rope-interleaved-fp16": dict(B=2, Sq=1, Hq=16, Hk=4, D=128, Sk=1024, Snew=1, dt=jnp.float16,
                                  rotary=True, interleaved=True, torch_check=True),
    "rope-D64-bf16": dict(B=2, Sq=1, Hq=8, Hk=8, D=64, Sk=512, Snew=1, dt=jnp.bfloat16, rotary=True),
    "no-append-bf16": dict(B=4, Sq=1, Hq=8, Hk=8, D=128, Sk=2048, Snew=0, dt=jnp.bfloat16),
    "paged-append-bf16": dict(B=2, Sq=1, Hq=8, Hk=8, D=128, Sk=1024, Snew=1, dt=jnp.bfloat16, paged=True),
    "paged-rope-GQA-bf16": dict(B=2, Sq=1, Hq=16, Hk=4, D=128, Sk=1024, Snew=1, dt=jnp.bfloat16,
                                paged=True, rotary=True),
}


@pytest.mark.parametrize("name", list(STEP))
def test_step(name):
    _step_case(**STEP[name])


def test_step_jit_donate():
    B, Sq, Hq, Hk, D, Sk, Snew = 2, 1, 8, 8, 128, 1024, 1
    dt = jnp.bfloat16
    rng = np.random.default_rng(2)
    q = jnp.asarray(rng.standard_normal((B, Sq, Hq, D)) * .5, dt)
    kc = jnp.asarray(rng.standard_normal((B, Sk, Hk, D)) * .5, dt)
    vc = jnp.asarray(rng.standard_normal((B, Sk, Hk, D)) * .5, dt)
    kn = jnp.asarray(rng.standard_normal((B, Snew, Hk, D)) * .5, dt)
    vn = jnp.asarray(rng.standard_normal((B, Snew, Hk, D)) * .5, dt)
    sl = jnp.asarray([100, 700], dtype=jnp.int32)
    step = jax.jit(lambda *a: F.flash_attn_kvcache_step(a[0], a[1], a[2], a[3],
                                                        knew=a[4], vnew=a[5]),
                   donate_argnums=(1, 2))
    r = ref_step(q, kc, vc, sl, kn, vn, 1.0 / D ** 0.5, False)  # before donation invalidates kc/vc
    out, _, _ = step(q, kc, vc, sl, kn, vn)
    assert float(jnp.max(jnp.abs(out.astype(jnp.float32) - r))) < 6e-2
