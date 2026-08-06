"""libtorch-in-FFI: call vLLM torch-ABI ops from a JITTABLE XLA custom call, with no dlpack
bridge (no clones, no host sync) and torch out of the JAX graph.

vLLM ships only torch-ABI ops and no csrc, so a fully torch-free FFI is unreachable; the
handler instead wraps XLA device buffers as `at::Tensor` via `at::from_blob` (zero-copy)
and runs the op on the FFI's hipStream. `silu_and_mul` is the proof-of-mechanism; see
lumi_vllm_ffi.hip.
"""
from __future__ import annotations

import ctypes
import os

import jax
import jax.numpy as jnp

from lumi_kernels._lib import so_path as _so_path, have as _have

_SO_NAME = "liblumi_vllm_ffi.so"
_registered = False


def _ensure():
    global _registered
    if _registered:
        return
    import torch  # noqa: F401  (pull libc10/libtorch before dlopen)
    lib = ctypes.CDLL(_so_path(_SO_NAME))
    addr = ctypes.cast(lib.LumiSiluAndMul, ctypes.c_void_p).value
    jax.ffi.register_ffi_target("lumi_silu_and_mul", jax.ffi.pycapsule(addr), platform="ROCM")
    _registered = True


def is_available():
    try:
        _ensure()
        return _have(_SO_NAME)
    except Exception:
        return False


def silu_and_mul(x):
    """SwiGLU activation via vLLM's kernel inside a jittable FFI call (no bridge).
    x (N, 2d) -> (N, d): out = silu(x[:, :d]) * x[:, d:]."""
    _ensure()
    N, twod = x.shape
    return jax.ffi.ffi_call(
        "lumi_silu_and_mul", jax.ShapeDtypeStruct((N, twod // 2), x.dtype))(x)


def silu_and_mul_ref(x):
    """Pure-JAX SwiGLU oracle (jittable)."""
    d = x.shape[1] // 2
    a, b = x[:, :d], x[:, d:]
    return (a * jax.nn.sigmoid(a) * b).astype(x.dtype)
