"""JAX binding for Mamba's selective_scan via XLA FFI (jittable fwd+bwd, torch-free).

Layouts (contiguous): u/delta/z (B, dim, L) fp32/fp16/bf16; A (dim, N) fp32, N <= 256;
B/C (B, G, N, L) in u.dtype; D/delta_bias (dim,) fp32. dB/dC come back fp32 and are cast.
"""
from __future__ import annotations

import ctypes
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

from lumi_kernels._lib import so_path as _so_path, have as _have

_SO_NAME = "liblumi_sscan.so"
_registered = False
_CHUNK = 2048


def _ensure():
    global _registered
    if _registered:
        return
    lib = ctypes.CDLL(_so_path(_SO_NAME))  # torch-free, no import torch needed
    for target, sym in (("lumi_sscan_fwd", "LumiSscanFwd"), ("lumi_sscan_bwd", "LumiSscanBwd")):
        addr = ctypes.cast(getattr(lib, sym), ctypes.c_void_p).value
        jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(addr), platform="ROCM")
    _registered = True


def is_available():
    try:
        _ensure()
        return True
    except Exception:
        return False


def _attrs(has_d, has_db, has_z, softplus):
    return dict(has_d=np.int32(has_d), has_db=np.int32(has_db), has_z=np.int32(has_z),
                softplus=np.int32(softplus))


def _fwd_call(u, delta, A, Bv, Cv, D, db, z, has_d, has_db, has_z, softplus):
    Bsz, dim, L = u.shape
    N = A.shape[1]
    n_chunks = -(-L // _CHUNK)
    rets = (jax.ShapeDtypeStruct(u.shape, u.dtype),                       # out
            jax.ShapeDtypeStruct((Bsz, dim, n_chunks, 2 * N), jnp.float32),  # x residual
            jax.ShapeDtypeStruct(u.shape, u.dtype))                       # out_z
    return jax.ffi.ffi_call("lumi_sscan_fwd", rets)(
        u, delta, A, Bv, Cv, D, db, z, **_attrs(has_d, has_db, has_z, softplus))


def _bwd_call(u, delta, A, Bv, Cv, D, db, z, out, x, dout, has_d, has_db, has_z, softplus):
    Bsz, dim, L = u.shape
    G, N = Bv.shape[1], Bv.shape[2]
    rets = (jax.ShapeDtypeStruct(u.shape, u.dtype),            # du
            jax.ShapeDtypeStruct(u.shape, u.dtype),            # ddelta
            jax.ShapeDtypeStruct((dim, N), jnp.float32),       # dA
            jax.ShapeDtypeStruct(Bv.shape, jnp.float32),       # dB (f32 accum)
            jax.ShapeDtypeStruct(Cv.shape, jnp.float32),       # dC
            jax.ShapeDtypeStruct((dim,), jnp.float32),         # dD
            jax.ShapeDtypeStruct((dim,), jnp.float32),         # ddelta_bias
            jax.ShapeDtypeStruct(u.shape, u.dtype))            # dz
    return jax.ffi.ffi_call("lumi_sscan_bwd", rets)(
        u, delta, A, Bv, Cv, D, db, z, out, x, dout,
        **_attrs(has_d, has_db, has_z, softplus))


@functools.partial(jax.custom_vjp, nondiff_argnums=(8, 9, 10, 11))
def _sscan(u, delta, A, Bv, Cv, D, db, z, has_d, has_db, has_z, softplus):
    _ensure()
    out, _, out_z = _fwd_call(u, delta, A, Bv, Cv, D, db, z, has_d, has_db, has_z, softplus)
    return out_z if has_z else out


def _vjp_fwd(u, delta, A, Bv, Cv, D, db, z, has_d, has_db, has_z, softplus):
    _ensure()
    out, x, out_z = _fwd_call(u, delta, A, Bv, Cv, D, db, z, has_d, has_db, has_z, softplus)
    return (out_z if has_z else out), (u, delta, A, Bv, Cv, D, db, z, out, x)


def _vjp_bwd(has_d, has_db, has_z, softplus, res, dout):
    u, delta, A, Bv, Cv, D, db, z, out, x = res
    du, ddelta, dA, dB, dC, dD, ddb, dz = _bwd_call(
        u, delta, A, Bv, Cv, D, db, z, out, x, dout, has_d, has_db, has_z, softplus)
    return (du, ddelta, dA, dB.astype(Bv.dtype), dC.astype(Cv.dtype), dD, ddb,
            dz if has_z else jnp.zeros_like(z))


_sscan.defvjp(_vjp_fwd, _vjp_bwd)


def selective_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                   delta_softplus=False):
    """Mamba selective scan (jittable fwd+bwd, torch-free FFI), matching
    mamba_ssm.ops.selective_scan_fn and differentiable in all array args.
    u/delta (B, dim, L) fp32/fp16/bf16; A (dim, N) fp32, N <= 256; B/C (B, G, N, L) in
    u.dtype (dim % G == 0); D/delta_bias (dim,) fp32 optional; z optional -> silu(z) * y."""
    Bsz, dim, L = u.shape
    has_d, has_db, has_z = D is not None, delta_bias is not None, z is not None
    one = np.zeros((dim,), np.float32)
    return _sscan(u, delta, A.astype(jnp.float32), B, C,
                  D.astype(jnp.float32) if has_d else one,
                  delta_bias.astype(jnp.float32) if has_db else one,
                  z if has_z else jnp.zeros_like(u),
                  has_d, has_db, has_z, bool(delta_softplus))
