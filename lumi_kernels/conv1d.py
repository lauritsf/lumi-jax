"""JAX binding for the container's causal_conv1d (Mamba/SSM) gfx90a kernel, plus a pure-JAX
reference. x (B, dim, L); weight (dim, width); bias (dim,); optional SiLU. `causal_conv1d`
is the differentiable, fully native path (fwd + bwd both XLA FFI, jittable);
`causal_conv1d_ref` is the portable pure-JAX oracle.
"""
from __future__ import annotations

import ctypes
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

from lumi_kernels._lib import so_path as _so_path, have as _have

_SO_NAME = "liblumi_conv1d.so"
_registered = False
_DUMMY = None


def _ensure():
    global _registered, _DUMMY
    if _registered:
        return
    import torch  # noqa: F401  (pull libc10 before dlopen)
    lib = ctypes.CDLL(_so_path(_SO_NAME))
    for target, sym in (("lumi_conv1d_fwd", "LumiConv1dFwd"), ("lumi_conv1d_bwd", "LumiConv1dBwd")):
        addr = ctypes.cast(getattr(lib, sym), ctypes.c_void_p).value
        jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(addr), platform="ROCM")
    _DUMMY = np.zeros((1,), np.float32)
    _registered = True


def is_available():
    try:
        _ensure()
        return _have(_SO_NAME)
    except Exception:
        return False


def causal_conv1d_ffi(x, weight, bias=None, silu=True):
    """Fused gfx90a causal_conv1d. x (B,dim,L), weight (dim,width), bias (dim,)."""
    _ensure()
    out_t = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.ffi.ffi_call("lumi_conv1d_fwd", out_t)(
        x, weight, bias if bias is not None else _DUMMY,
        has_bias=np.int32(0 if bias is None else 1), silu=np.int32(1 if silu else 0))


def causal_conv1d_bwd_ffi(dout, x, weight, bias, silu=True):
    """Native gfx90a causal_conv1d backward (non-deterministic path). Returns
    (dx, dweight, dbias). dout/x (B,dim,L); weight (dim,width); bias (dim,).
    dweight/dbias are FP32 accumulators in the kernel, cast back to weight dtype."""
    _ensure()
    dim, width = weight.shape
    dx_t = jax.ShapeDtypeStruct(x.shape, x.dtype)
    dw_t = jax.ShapeDtypeStruct(weight.shape, jnp.float32)  # kernel accumulates in fp32
    db_t = jax.ShapeDtypeStruct((dim,), jnp.float32)
    dx, dw, db = jax.ffi.ffi_call("lumi_conv1d_bwd", (dx_t, dw_t, db_t))(
        dout, x, weight, bias, has_bias=np.int32(1), silu=np.int32(1 if silu else 0))
    return dx, dw.astype(weight.dtype), db.astype(bias.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _ccd(x, weight, bias, silu):
    return causal_conv1d_ffi(x, weight, bias, silu)


def _ccd_fwd(x, weight, bias, silu):
    return causal_conv1d_ffi(x, weight, bias, silu), (x, weight, bias)


def _ccd_bwd(silu, res, g):
    x, weight, bias = res
    return causal_conv1d_bwd_ffi(g, x, weight, bias, silu)  # native bwd kernel


_ccd.defvjp(_ccd_fwd, _ccd_bwd)


def causal_conv1d(x, weight, bias=None, silu=True):
    """Differentiable causal_conv1d, fully native gfx90a (fwd + bwd FFI), jittable.
    (B,dim,L) x; (dim,width) weight; (dim,) bias."""
    if bias is None:
        bias = jnp.zeros((weight.shape[0],), x.dtype)  # bias=0 is a no-op
    return _ccd(x, weight, bias, silu)


def causal_conv1d_ref(x, weight, bias=None, silu=True):
    """Pure-JAX depthwise causal conv1d (jittable, differentiable)."""
    B, dim, L = x.shape
    width = weight.shape[1]
    xp = jnp.pad(x.astype(jnp.float32), ((0, 0), (0, 0), (width - 1, 0)))
    w = weight.astype(jnp.float32)
    out = sum(w[:, k][None, :, None] * jax.lax.slice_in_dim(xp, k, k + L, axis=2)
              for k in range(width))
    if bias is not None:
        out = out + bias.astype(jnp.float32)[None, :, None]
    if silu:
        out = out * jax.nn.sigmoid(out)
    return out.astype(x.dtype)
