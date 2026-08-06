"""JAX paged-attention decode: pure-JAX write_kv_cache + bridged paged_attention_v1
+ jittable libtorch-in-FFI path, vs a full-attention reference. Needs the `torch` image."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from lumi_kernels import paged_attention as P

pytestmark = [pytest.mark.gpu, pytest.mark.torch, pytest.mark.so("liblumi_vllm_ffi.so")]


def build_cache(seq_lens, num_kv_heads, head_size, block_size, jdt, seed=0):
    g = np.random.default_rng(seed)
    total = sum(seq_lens)
    num_seqs = len(seq_lens)
    Kall = g.standard_normal((total, num_kv_heads, head_size)) * 0.5
    Vall = g.standard_normal((total, num_kv_heads, head_size)) * 0.5
    keys, vals, off = [], [], 0
    for L in seq_lens:
        keys.append(jnp.asarray(Kall[off:off + L]).astype(jdt))
        vals.append(jnp.asarray(Vall[off:off + L]).astype(jdt))
        off += L

    max_blocks = max((L + block_size - 1) // block_size for L in seq_lens)
    num_blocks = total // block_size + num_seqs + 4
    bt = np.zeros((num_seqs, max_blocks), np.int32)
    slot = np.zeros(total, np.int32)
    phys, off = 0, 0
    for i, L in enumerate(seq_lens):
        nb = (L + block_size - 1) // block_size
        blks = list(range(phys, phys + nb)); phys += nb
        bt[i, :nb] = blks
        for t in range(L):
            slot[off + t] = blks[t // block_size] * block_size + (t % block_size)
        off += L

    kc, vc = P.empty_kv_cache(num_blocks, num_kv_heads, head_size, block_size, jdt)
    kc, vc = jax.jit(P.write_kv_cache)(jnp.concatenate(keys, 0), jnp.concatenate(vals, 0),
                                       kc, vc, jnp.asarray(slot))
    return keys, vals, kc, vc, jnp.asarray(bt)


CASES = {
    "MHA-seqs-20-35-8": ([20, 35, 8], 8, 8, 128, 16),
    "GQA-16-4": ([33, 17], 16, 4, 128, 16),
    "MQA-16-1": ([40, 9], 16, 1, 128, 16),
    "headdim-64": ([50], 8, 8, 64, 16),
}


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("name", list(CASES))
def test_decode(name, dtype):
    seq_lens, num_heads, num_kv_heads, head_size, block_size = CASES[name]
    jdt = {"bf16": jnp.bfloat16, "fp16": jnp.float16}[dtype]
    scale = 1.0 / head_size ** 0.5
    g = np.random.default_rng(0)
    query = jnp.asarray(g.standard_normal((len(seq_lens), num_heads, head_size)) * 0.5).astype(jdt)

    keys, vals, kc, vc, btj = build_cache(seq_lens, num_kv_heads, head_size, block_size, jdt)
    slj = jnp.asarray(np.array(seq_lens, np.int32))
    ref = P.reference(query, keys, vals, seq_lens, num_kv_heads, scale)
    tol = 5e-2 if dtype == "bf16" else 2e-2

    out = P.paged_attention(query, kc, vc, btj, slj,
                            num_kv_heads, scale, block_size, max(seq_lens))
    e = float(np.max(np.abs(np.asarray(out, np.float32) - np.asarray(ref, np.float32))))
    assert e < tol, f"dlpack bridge err={e:.4f}"

    out_f = jax.jit(P.paged_attention_ffi, static_argnums=(5, 6, 7, 8))(
        query, kc, vc, btj, slj, num_kv_heads, scale, block_size, int(max(seq_lens)))
    ef = float(np.max(np.abs(np.asarray(out_f, np.float32) - np.asarray(ref, np.float32))))
    assert ef < tol, f"libtorch-in-FFI err={ef:.4f}"
