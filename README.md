# JAX on LUMI + lumi_kernels

JAX with GPU support on LUMI (AMD MI250X, gfx90a, ROCm 7.0.2) — 0.11.0 by default, 0.9.1 on
request — plus **`lumi_kernels`**, custom XLA-FFI bindings to gfx90a kernels that XLA cannot
express on its own: CK flash attention, AOTriton, Mamba selective_scan, causal_conv1d, and
vLLM paged attention / MoE.

Runs in containers we build from the
[LUMI AI Factory environment](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/)
images. A stopgap until JAX is officially supported there
([laifs-container-recipes#31](https://github.com/lumi-ai-factory/laifs-container-recipes/issues/31)).

> **Use with caution.** These kernels were ported and bound to JAX with heavy use of AI and
> have not all been reviewed by hand. The test suite checks them against PyTorch and reference
> implementations where possible, but validate against your own reference before relying on it.

## Install

Point `LUMI_JAX_WORK` at the scratch directory holding the images. There is no default: a
guessed path resolves silently against another project's scratch, so everything here refuses
rather than guess.

```bash
export LUMI_JAX_WORK=/scratch/project_<id>/<name>-cache   # or: cp it into tools/site.local.sh
bash install.sh        # ./jax-venv        -- JAX + 4 prebuilt kernels, no torch
bash install-torch.sh  # ./jax-venv-torch  -- JAX + PyTorch + vLLM + a 5th kernel
```

`tools/site.local.sh` is sourced when present and is gitignored, so a checkout can carry its
own paths without them reaching the repo.

Both install pytest and `pip install -e .` this repo, then activate with `activate.sh` /
`activate-torch.sh`.

### Which container

`install.sh` gives you the **`lean`** image (6.9 GB): JAX, RCCL 2.27.7, Ray Tune, no shims of
any kind, and four XLA-FFI kernels prebuilt with **no PyTorch anywhere** — CK FMHA and AOTriton
fused attention, Mamba `selective_scan`, causal `conv1d`. Nothing to build, no `LD_PRELOAD`.

`install-torch.sh` gives you **`torch`** (~15 GB): the above plus PyTorch 2.10, vLLM, and a
fifth kernel. Use it when you want JAX and PyTorch in one interpreter.

```bash
bash install.sh                               # lean (default)
LUMI_JAX_VARIANT=stage-net bash install.sh    # same base, no kernel layers (6.6 GB)
SIF=/path/to/other.sif bash install.sh        # anything else, incl. a stock LAIF image
```

The one thing only `torch` has is `csrc/vllm` — `paged_attention_v1` and `silu_and_mul` — which
genuinely embed libtorch. (Mixture of Experts is *not* in that kernel; it is a Python bridge in
`lumi_kernels/moe.py`, which needs the vLLM package and so the `torch` image too.)

Both the jax line and the variant are recorded in the venv, so an existing install keeps the
image it was built against when a default moves, and every `sbatch` here resolves the same one.

### Which JAX

Both installers default to 0.11.0:

```bash
bash install.sh                        # jax 0.11.0
LUMI_JAX_LINE=0.9.1 bash install.sh    # jax 0.9.1  (JAXVER also works, as an alias)
```

`tools/resolve_image.sh` picks the container *and* the pip pins from that one knob, so a venv
and its image can never disagree about jax — they layer over the container's `/opt/venv`, and a
mismatch there fails in ways that look like a code bug.

Every release from 0.9.2 references `hipblaslt_ext::GemmInstance::setMaxWorkspaceBytes`, which
ROCm 7.0.2 does not define; our containers supply it as a loud no-op stub, so 0.11.0 needs one
of our images rather than a stock LAIF one. Its one cost is compile-time, not correctness: a
very large first GEMM compile (~23 min at 32768², under 10 s at 8192² and below), erased by
`jax_compilation_cache_dir`.

### Ray Tune on a GPU node

Both images ship `ray[tune]`, but **Ray will not start in a Slurm GPU job without one extra
step**. Source this before importing ray:

```bash
source tools/ray_visible_devices.sh
```

Ray raises `RuntimeError: Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES` when
`ROCR_VISIBLE_DEVICES` is set and `HIP_VISIBLE_DEVICES` is not — exactly what LUMI's Slurm gives
you. The check fires only once Ray has detected AMD GPUs, so a login-node `import ray` succeeds
and tells you nothing.

**Do not simply copy the value across**: the two masks compose, so `ROCR=6,7` means two GCDs
addressed `0,1`. The helper does that remap, leaves `HIP_VISIBLE_DEVICES` alone if you set it,
and is a no-op when `ROCR_VISIBLE_DEVICES` is unset.

## Use

```python
from lumi_kernels.attention import flash_ffi
out = flash_ffi.flash_attention_ffi(q, k, v, causal=True)         # jittable fwd+bwd
out = flash_ffi.flash_attention_varlen_traced(q, k, v, cu_q, cu_k, Sq, Sk)  # dynamic cu_seqlens
```

### Why these kernels

`jax.nn.dot_product_attention` runs on LUMI only through the XLA math lowering, which
materialises the full `S×S` score matrix per head: the forward OOMs a 64 GB GCD at S≈8192, the
backward at S≈4096, and no flag or dtype avoids it. The fix is a fused O(S)-memory attention
kernel, which XLA cannot express, so we bind prebuilt gfx90a kernels via XLA FFI. The same
reasoning covers the other bindings: grouped-expert GEMM, paged-KV gather, depthwise conv, and
selective scan. Things XLA fuses well on its own (RMSNorm, SiLU/SwiGLU, Adam, RoPE) measure as
fast or faster in pure JAX and are deliberately not bound.

### What's bound

All jittable and validated against the reference unless noted.

| binding | kernel | technique | fwd/bwd | notes |
|---|---|---|---|---|
| `flash_attention_ffi` | CK flash, dense | raw FFI (`fmha_fwd`/`fmha_bwd`) | yes / yes | full port: GQA/MQA, cross-attn, window/local, ALiBi, dropout, fp16/bf16, D≤256 |
| `flash_attention_varlen_ffi` | CK flash, packed varlen | raw FFI | yes / yes | `cu_seqlens` static (`nondiff`); same feature set |
| `flash_attention_varlen_traced` | CK flash, varlen | raw FFI | yes / yes | dynamic `cu_seqlens` (traced device operands), compiles once, composes with `jax.checkpoint` |
| `flash_attention_aotriton` | AOTriton flash | raw FFI (`attn_fwd`) | yes / no | forward only; 1.18–1.26x faster than CK forward through JAX |
| `flash_attention_hybrid` | AOTriton fwd + CK bwd | raw FFI | yes / yes | faster than PyTorch's best; works because AOTriton's `softmax_lse` matches CK's |
| `flash_attn_kvcache_ffi` | CK split-KV decode | raw FFI (`fmha_fwd_splitkv`) | yes / no | contiguous cache, attend only; split-KV heuristic up to ~8x on small-batch decode |
| `flash_attn_kvcache_step` | CK full decode step | raw FFI (`appendkv`+`splitkv`) | yes / no | in-place KV-append, fused rotary, paged `block_table`; caches via `input_output_aliases` |
| `conv1d` | causal_conv1d (Mamba/SSM) | raw FFI | yes / yes | ~1.9–2.1x vs pure-JAX, faster than torch too |
| `selective_scan` | Mamba selective_scan | raw FFI, self-compiled | yes / yes | torch-free; built from vendored mamba v2.3.2; completes the Mamba1 block |
| `paged_attention` | vLLM paged attention (decode) | dlpack bridge or libtorch-in-FFI | yes / no | the FFI path (`paged_attention_ffi`) is jittable and bridge-free |
| `moe` | vLLM fused experts (grouped GEMM) | dlpack bridge | yes / no | 11.7x vs pure-JAX; stays eager (Python-orchestrated pipeline) |
| `vllm_ffi.silu_and_mul` | vLLM SwiGLU | libtorch-in-FFI | yes / no | example of the libtorch-in-FFI mechanism |

Pure-JAX fallbacks that need no container: `attention.chunked` (online-softmax flash, O(S)
via internal remat) and `attention.reference`. For RoPE, use the pure-JAX `rope_ref`; the vLLM
kernel is faster in isolation but loses through the bridge.

Performance in one paragraph, measured on the 0.9.1 line (B=2, H=16, D=128, bf16): the FFI
attention paths hold peak HBM flat where the XLA lowering grows quadratically (1.50 GiB forward
/ 3.26 GiB gradient at S=32768, vs OOM at S=8192), and the freed memory buys batch worth more
than the kernel speedup itself. On the same kernel, JAX matches native PyTorch throughput
(0.98–1.00x). CK is the training backend, AOTriton the inference/prefill forward;
`flash_attention_hybrid` combines them.

Three binding techniques, in order of preference: **raw-pointer FFI** (stable dispatcher, no
torch ABI, jittable — all attention and conv), **dlpack bridge** (no linked symbols but eager —
MoE), and **libtorch-in-FFI** (jittable but pinned to the container's torch ABI — paged
attention, silu_and_mul). "Torch-free" means no torch ABI coupling: the CK-flash and conv1d
`.so` still load libtorch at process start, which is why `import torch` must precede
`ctypes.CDLL` for them. Only AOTriton and selective_scan are libtorch-free on disk.

### Gotchas

- ScratchAllocator alignment is capped at 16 on this ROCm FFI build — request 16, the pointer
  is page-aligned anyway.
- Varlen padded tails are unwritten: forward `out` rows past `cu_q[-1]` and backward dk/dv rows
  past `cu_k[-1]` are uninitialized, so mask them in the consumer.
- Every per-sequence `seqlen_{q,k}` must be `≤ max_seqlen_{q,k}`; it cannot be checked
  host-side because `cu_seqlens` are traced, and violating it reads out of bounds.
- Dropout (seed, offset) is threaded as a device operand so per-step dropout works under jit
  with no recompile; full 64-bit seeds need `jax_enable_x64`.
- `import torch` before `ctypes.CDLL(our.so)` for any binding whose `.so` transitively needs
  libtorch (CK-flash, conv1d).

## Build the kernels

Not needed on either install — `lean` ships four of the five prebuilt and `torch` all five.
Build only when changing a kernel; all emit into `lumi_kernels/lib/`. Four build in either
image — `csrc/flash/build_ffi.sh`, `csrc/aotriton/build_aotriton.sh`,
`csrc/conv1d/build_conv1d.sh`, `csrc/sscan/build_sscan.sh` — and print which source they
resolved. Only `csrc/vllm/build_vllm_ffi.sh` requires the `torch` image.

`lumi_kernels` looks for each library in `$LUMI_KERNELS_LIB`, then `lumi_kernels/lib/`, then the
image's `/opt/lumi-jax/lumi_kernels/lib` — **per library**, so a local build of one kernel takes
precedence for that kernel while the others still come from the image.

## Test

```bash
sbatch tests/slurm/run-tests-base.sbatch   # lean image, 8 GPUs: pure-JAX + torch-free kernels
sbatch tests/slurm/run-tests-full.sbatch   # torch image, 1 GPU: all kernel bindings
```

Markers auto-skip what the environment lacks (no GPU, no torch, missing `.so`), so each runner
also asserts that something actually ran — "0 passed" is a pytest success and a non-result here.
Both runners resolve their image through `jax_resolve_image`, so they follow the venv.

## Building the images

Images live in `$LUMI_JAX_WORK/images/`, one def + sbatch pair per
variant in `containers/`, ABI baselines in `tools/baselines/`. Two **products** — what the
installers give you — and five **stages** that exist so the products can be built. Both products
exist on both jax lines, as `lumi-jax-<variant>-j{091,0110}-latest.sif`:

| variant | size | what it adds |
|---|---|---|
| `stage-mpich` | 6.45 GB | jax on LAIF's mpich layer; `libdw` and `librocm_smi64` shims gone |
| `stage-net` | 6.6 | RCCL 2.27.7 over the base's 2.26.6 — **no shims at all** |
| `stage-aotriton` | 6.7 | standalone AOTriton 0.11.1b, unpacked from the release tarballs |
| `stage-fmha` | 6.8 | `libck_fmha.so` + CK headers — fused attention with no torch |
| **`lean`** | **6.9** | the four torch-free FFI kernels prebuilt + Ray Tune |
| `stage-torch` | 15.0 | jax layered over LAIF's full layer, in `/opt/jax-venv` |
| `torch` | 15.1 | all five kernels, including `csrc/vllm` — the only image with vllm |

### The build tree

Every image is one `singularity build` on top of another; each layer adds one thing. **Two
roots, both from LAIF**, at
`/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/`:

```
LAIF lumi-multitorch-mpich-….sif                       (the lean root)
 └─ build_jax.sbatch               jax.def          ─► stage-mpich
     └─ build_jax_rccl.sbatch      jax-rccl.def     ─► stage-net
         └─ build_jax_aotriton.sbatch  jax-aotriton.def ─► stage-aotriton
             └─ build_jax_ck_fmha.sbatch  jax-ck-fmha.def ─► stage-fmha
                 └─ build_jax_kernels_lean.sbatch         ─► lean        ★ install.sh
LAIF lumi-multitorch-full-….sif                        (the torch root)
 └─ build_jax_torch_base.sbatch    jax-torch-base.def    ─► stage-torch
     └─ build_jax_torch_kernels.sbatch jax-torch-kernels.def ─► torch    ★ install-torch.sh
```

**The two roots never meet.** The lean chain has no torch at any point; the torch chain must
start from LAIF's full layer because `csrc/vllm` links libtorch. Apptainer bootstrap chains are
linear — one parent SIF, no composition — so RCCL, AOTriton and CK FMHA are duplicated between
the chains by necessity. Ray Tune enters at the product leaves only: adding it at a root
invalidates every rung above it.

**Both jax lines.** `jax-version.def` is a sideways step, not a rung: it takes a variant at
0.9.1 and produces the same variant at another jax, swapping the four pins and adding the
hipBLASLt stub and the all-reduce `.pth` hook. Only `stage-net` is retargeted — everything above
is an ordinary build that resolves a 0.11.0 parent. Retargeting higher would swap jax underneath
kernels already compiled against it; `build_jax_version.sbatch` refuses `VARIANT=torch` for that
reason. Mind `VENVDIR`: the lean chain keeps jax in `/opt/venv`, the torch chain keeps *torch*
there and layers jax in `/opt/jax-venv` — getting it wrong leaves two jax trees in one
interpreter, and it looks like it worked.

**Rebuilding.** Run `tools/check_defs.sh` before submitting — it parses all eight defs in about
three seconds. Singularity strips leading whitespace before scanning for section headers, so an
indented line of prose beginning with `%post` or `%help` makes a def unparseable.

Every def takes `{{ BASE }}` as a `--build-arg`; each build script resolves its
parent through the `-latest` symlink for that variant and line, so rebuilding a lower layer
needs no edit above it. Two side artifacts (`tools/build_ck_fmha.sh`,
`tools/build_causal_conv1d.sh`) are built by separate CPU jobs and copied in via `%files`.
Builds need no GPU. Set `SBATCH_ACCOUNT`; the CPU partitions need a project with CPU billing
units, and a GPU-only project fails there with a bare `AssocMaxSubmitJobLimit` that reads
like a queue-depth problem. `dev-g` is meant for
debugging and quick testing rather than production runs, and caps a job at 3 h walltime and
16 node-hours, with at most 2 jobs at once. Filesystem access follows unix group membership,
not the job account.

### Upgrading onto a new LAIF base

The base release is two variables, `LAIF_PINS` and `LAIF_STAMP` at the top of
`tools/image_name.sh` — bump both together, never one; that is the only place either appears.
Both are environment-overridable, which is how to dry-run a candidate without editing anything.

Assess before rebuilding — a bad base should cost minutes, not a day of builds:

```bash
tools/probe_image.sh <new-laif-base.sif>          # login node, seconds
#   diff against tools/baselines/
tools/compat_matrix.sh <new-laif-base.sif>        # login node: probe + rebuild all csrc targets
sbatch containers/run_compat_matrix.sbatch <sif>  # adds the pytest suite, needs GPUs
```

A stage that cannot run reports `unknown`, never a pass; a target whose inputs the image does
not ship reports `n/a`. Only `FAIL` is a problem. Then rebuild bottom-up along the tree, gate,
and re-baseline into `tools/baselines/`. The known ABI risk on a base bump is `csrc/vllm`:
vLLM moves `silu_and_mul` between extension modules and signatures.

### Naming

```
ours    lumi-jax-<variant>-<pins>-<stamp>.sif
LAIF    lumi-multitorch-<layer>-<pins>-<stamp>.sif

lumi-jax-lean-u24r70f21m50t210_c227j0110-20260805_180448.sif
                └────────────┘ └───────┘ └─────────────┘
                LAIF's, verbatim   ours    our build time
```

Everything before the underscore is the LAIF base release copied verbatim — the one fact you
need when a base bump breaks something — so `t210` appears in images with no torch in them.
The `-latest` symlink is keyed `<variant, jax line>`, exactly two axes; a second LAIF base
would make it ambiguous, and only `jax_image_{name,resolve,publish}` change when one arrives.

### Constraints worth knowing

**Build route.** `singularity build` needs root or fakeroot and neither is available on LUMI
(`max_user_namespaces` is 0, no `/etc/subuid` entry). PRoot works — it emulates root via
`ptrace`, roughly 4x slower, and apt/pip work but expect surprises around chown/setuid:

```bash
export PATH=/appl/lumi/SW/CrayEnv/EB/systools/25.03/bin:$PATH   # provides proot
export SINGULARITY_CACHEDIR=<somewhere with space, not $HOME>
singularity build jax.sif jax.def     # singularity auto-detects proot
```

**The three shims.** Our images need none of them; `install.sh` probes for each and skips it.
The machinery is there for stock LAIF images, which still do.

| shim | root cause on a stock base | what our images do instead |
|---|---|---|
| `libdw.so.1` from Cray PE | no `libdw*` in the image | `apt install libdw1t64` at build time |
| `librocm_smi64.so.1` symlink | image ships only `.so.7` | symlink + `ldconfig` at build time |
| `librccl_window_stub.so` + `LD_PRELOAD` | the base RCCL exports zero `ncclCommWindow*` | RCCL 2.27+ in the leaf image |

Multi-node works today because JAX's collectives happen never to call the window API — an
assumption with no test protecting it, which will fail confusingly the day JAX adopts symmetric
memory.

**jax 0.11.0 takes two fixes.** The hipBLASLt `setMaxWorkspaceBytes` stub, and: single-process
collectives over 2+ GCDs hang, because XLA's Triton-generated intra-node all-reduce kernel —
enabled for every ROCm device with no arch test upstream — never completes on gfx90a. The image
turns it off at python startup via a `.pth` hook (`tools/install_allreduce_hook.sh`).

**Gates must not be able to pass while testing nothing.** Every marker in `tests/` auto-skips on
a missing prerequisite, so a wrong image skips the whole suite and still exits 0. Each gate
greps its own output for `[1-9][0-9]* passed`, and asserts only that *something* passed, never a
fixed count. Keep the guard when touching a gate.

**Verify on a compute node.** `singularity exec` on a login node is not evidence about GPU
behaviour.

**The repo is bind-mounted live into the gates.** Editing `tests/` or `lumi_kernels/` while a
gate runs means it is testing a moving target; `containers/test_lean_kernels_image.sbatch`
copies what it needs to a per-job staging dir for this reason.

**Load kernels through `_lib.so_path()`, never `_lib.lib_path()`.** `lib_path()` names a build
*output* and is always the checkout's `lib/`, so loading through it hides the image's prebuilt
kernels; `so_path()` does the per-library search described under Build the kernels.

**`%files` copies `lumi_kernels/lib/` into the image**, so every def that copies the package
clears `lib/*.so` in `%post` before building — otherwise a kernel that failed to build keeps the
submitter's stale copy and the `test -s` assert passes on it.

## Minimum recipe (if you already use the LAIF containers)

1. `pip install 'jax==0.9.1' 'jaxlib==0.9.1' 'jax-rocm7-plugin==0.9.1.post5' 'jax-rocm7-pjrt==0.9.1.post5'`
   — 0.9.1 rather than 0.11.0, because a stock LAIF container has no hipBLASLt stub.
2. Provide three libs (see `install.sh`): a `librocm_smi64.so.1` symlink, `libdw.so.1` from
   Cray PE, and `librccl_window_stub.so` from `csrc/rccl_window_stub.c`.
3. `LD_LIBRARY_PATH=$SHIM:...`, `LD_PRELOAD=$SHIM/librccl_window_stub.so`, `NCCL_WIN_ENABLE=0`.
4. On 0.10.2+ only: set `XLA_FLAGS=--xla_gpu_unsupported_use_all_reduce_one_shot_kernel=false`,
   or any `pmap`/`shard_map` collective over 2+ GCDs in one process hangs forever. Our containers
   do this for you (`tools/install_allreduce_hook.sh`); a stock one does not.

**mpi4jax** is not included: it has lowering rules only for cpu/cuda, not rocm.

## License

MIT (see `LICENSE`).
