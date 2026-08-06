"""Fused Mixture-of-Experts for JAX on gfx90a. `moe` bridges vLLM's `fused_experts` via
zero-copy dlpack and is EAGER, not jittable: `fused_experts` is a multi-stage Python
pipeline with no stable C++ entry point, so there is nothing to wrap in an FFI. `moe_ref`
is the pure-JAX oracle and fallback.

Shapes (vLLM): hidden (T, H); w1 (E, 2*I, H) [gate|up]; w2 (E, H, I); topk_weights/
topk_ids (T, k). Activation SwiGLU = silu(gate) * up.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_torch = None
_fe = None


def _lazy():
    global _torch, _fe
    if _torch is None:
        import torch
        from vllm.model_executor.layers.fused_moe import fused_experts
        _torch = torch
        _fe = fused_experts


def is_available():
    try:
        _lazy()
        return True
    except Exception:
        return False


def _to_torch(x):
    jax.block_until_ready(x)
    return _torch.utils.dlpack.from_dlpack(x.__dlpack__())


def moe(hidden, w1, w2, topk_weights, topk_ids):
    """Bridged vLLM fused_experts (eager). Returns (T, H) output."""
    _lazy()
    ht, w1t, w2t = _to_torch(hidden), _to_torch(w1), _to_torch(w2)
    twt = _to_torch(topk_weights).to(_torch.float32)
    tit = _to_torch(topk_ids).to(_torch.int32)
    out = _fe(ht, w1t, w2t, twt, tit, inplace=False)
    _torch.cuda.synchronize()
    return jax.dlpack.from_dlpack(out)


def moe_ref(hidden, w1, w2, topk_weights, topk_ids):
    """Pure-JAX MoE reference (per-token expert gather + SwiGLU). Differentiable."""
    T, H = hidden.shape
    k = topk_ids.shape[1]
    I = w1.shape[1] // 2
    x = hidden.astype(jnp.float32)
    out = jnp.zeros((T, H), jnp.float32)
    for j in range(k):
        e = topk_ids[:, j]                         # (T,)
        we = topk_weights[:, j].astype(jnp.float32)  # (T,)
        w1e = w1[e].astype(jnp.float32)            # (T, 2I, H)
        gate_up = jnp.einsum("th,tih->ti", x, w1e)  # (T, 2I)
        gate, up = gate_up[:, :I], gate_up[:, I:]
        act = (gate * jax.nn.sigmoid(gate)) * up    # SwiGLU (T, I)
        w2e = w2[e].astype(jnp.float32)             # (T, H, I)
        out = out + we[:, None] * jnp.einsum("ti,thi->th", act, w2e)
    return out.astype(hidden.dtype)


def router_topk(gate_logits, k):
    """Standard softmax router: returns (topk_weights, topk_ids) like vLLM expects."""
    probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1)
    topk_w, topk_id = jax.lax.top_k(probs, k)
    topk_w = topk_w / jnp.sum(topk_w, axis=-1, keepdims=True)  # renormalize
    return topk_w, topk_id.astype(jnp.int32)
