# Non-attention container kernels bound into JAX

The non-attention gfx90a kernels from LAIF's `full` layer that are worth exposing to JAX
(things XLA **can't** express well). They run only in the `torch` image, because they embed
libtorch.

| module | kernel | technique | speedup vs pure-JAX | test |
|--------|--------|-----------|--------------------:|------|
| `conv1d.py` | causal_conv1d (Mamba/SSM) | **XLA FFI** (raw `causal_conv1d_{fwd,bwd}_cuda<>` dispatchers) -- jittable native **fwd + bwd** (`custom_vjp` wires the native backward) | ~1.9–2.1× vs pure-JAX; faster than torch too (fwd+bwd) | `test_conv1d.py` |
| `paged_attention.py` | vLLM `paged_attention_v1` (KV-cache decode) | pure-JAX cache write (scatter) + kernel via **dlpack bridge** *or* **jittable libtorch-in-FFI** (`paged_attention_ffi`) | enables a decode path JAX lacked; FFI path is jittable + bridge-free (matches bridge exactly) | `test_paged_attention.py` |
| `moe.py` | vLLM `fused_experts` (grouped/expert GEMM) | **dlpack-bridged** (torch-ABI only) + pure-JAX `moe_ref` | **11.7×** | `test_moe.py` |

Build the FFI: `bash build_conv1d.sh` → `liblumi_conv1d.so`. The bridged ones (paged,
MoE) need no build (they call the torch op via dlpack).

**Why these and not RMSNorm/RoPE/SiLU?** Those are XLA-fusable -- pure-JAX RMSNorm
measured *faster* than the standalone vLLM kernel (0.44 vs 0.55 ms). Bind only what XLA
can't do: the flash algorithm, grouped GEMM, paged gather, depthwise conv.

**FFI vs dlpack bridge vs libtorch-in-FFI.** Kernels exposing a raw-pointer C++ dispatcher
(CK flash, AOTriton, causal_conv1d) get a clean torch-free XLA FFI (jittable). vLLM ships
only torch-ABI ops (`at::Tensor&`; raw launchers unexported, no csrc in the wheel), so a
torch-free FFI isn't reachable -- they're either **dlpack-bridged** (eager, clones + sync;
`paged_attention.py`, `moe.py`) or wrapped in a **libtorch-in-FFI** handler that takes torch
*out of the JAX graph* and removes the bridge overhead, staying jittable. `vllm_ffi.py` /
`lumi_vllm_ffi.hip` show the latter on `silu_and_mul` (build `bash build_vllm_ffi.sh`, test
`test_vllm_ffi.py`); paged attention follows the same pattern.
