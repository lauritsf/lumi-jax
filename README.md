# JAX and CK attention on LUMI

JAX 0.11.2 and standalone CK attention for MI250X (`gfx90a`), on ROCm 7.2.4.
The base is LAIF `u24r72f21m50t211-20260929_104918`.
Validate your application's outputs against its reference before changing image pins.

## Install and run

```bash
export LUMI_JAX_WORK=/scratch/project_<id>/<name>-cache
bash install.sh
```

Installation creates `$LUMI_JAX_WORK/jax-venv` over the image's `/opt/venv`, adding
pytest, Ray Tune and an editable checkout. Set `VENV_DIR` for another destination;
installation clears that directory. `tools/site.local.sh` can supply local paths
and is gitignored.

```bash
VENV_DIR=/path/to/new-env SIF=/path/to/immutable.sif bash install.sh
```

Inside the matching container, run `source activate.sh`, setting the same
`VENV_DIR` for custom installations. Bind the resolved environment path into the
container; `slurm/example.sbatch` provides a job template. Existing environments
retain their recorded immutable image, JAX line and variant. The resolver accepts
0.9.1 and 0.11.0 for existing environments. We test this release with JAX 0.11.2.
`LUMI_JAX_LINE` selects the line, with `JAXVER` accepted as an alias.

Before importing Ray on GPU nodes, run `source tools/ray_visible_devices.sh`.
The helper maps Slurm's physical GPU mask to relative HIP indices.

## Attention

```python
from lumi_kernels.attention import flash_ffi
out = flash_ffi.flash_attention_ffi(q, k, v, causal=True)
out = flash_ffi.flash_attention_varlen_traced(q, k, v, cu_q, cu_k, Sq, Sk)
```

| API | Constraints |
| --- | --- |
| `flash_attention_ffi` | Dense forward/backward; fp16/bf16, D≤256, GQA/MQA, windows, ALiBi, dropout |
| `flash_attention_varlen_ffi` | Packed forward/backward; static cumulative lengths |
| `flash_attention_varlen_traced` | Packed forward/backward; device cumulative lengths, checkpointing |
| `flash_attn_kvcache_ffi` | Forward-only contiguous-cache decode |
| `flash_attn_kvcache_step` | Forward-only cache append/decode, rotary and paged block tables |

`attention.chunked`, `attention.reference` and `rope.rope_ref` provide pure-JAX
implementations. JAX collectives use RCCL; mpi4jax has no ROCm lowering.

- Mask varlen output rows beyond `cu_q[-1]` and gradient rows beyond `cu_k[-1]`;
  the kernel does not write those tails.
- Keep sequence lengths within the declared maxima; traced cumulative lengths
  cannot be checked on the host.
- Pass dropout seeds as device operands. Enable `jax_enable_x64` for full-width
  64-bit seeds.
- Keep the startup hook's collective and command-buffer settings until replacement
  settings pass application checks on compute nodes.

## Build and validate

Set `SBATCH_ACCOUNT` before submitting from the repo root. Builds use a CPU-billing
account. `BUILD_SCRATCH` overrides temporary storage; `/tmp` consumes job memory.

| Script in `containers/` | Output | Input |
| --- | --- | --- |
| `build_jax.sbatch` | stage-mpich, JAX 0.9.1 | LAIF MPICH |
| `build_jax_rccl.sbatch` | stage-net | stage-mpich |
| `build_jax_version.sbatch` | selected JAX stage | 0.9.1 stage |
| `build_ck_fmha.sbatch` | standalone CK library and headers | pinned upstream source |
| `build_jax_kernels_lean.sbatch` | lean attention image | stage-net and CK artifacts |
| `build_jax_rocr.sbatch` | immutable repaired runtime image | compatible lean image |

```bash
sbatch --export=ALL,VARIANT=stage-net,LUMI_JAX_LINE=0.11.2 containers/build_jax_version.sbatch
# After the stage and CK artifacts are available:
sbatch --export=ALL,LUMI_JAX_LINE=0.11.2 containers/build_jax_kernels_lean.sbatch
```

The attention builder accepts `BASE` and `OUT` overrides and leaves release links
unchanged. FFI output is specific to its JAX headers; retarget stages before building
attention. `csrc/flash/build_ffi.sh` rebuilds the adapter against `CK_FMHA_ROOT`.
Libraries resolve through `$LUMI_KERNELS_LIB`, checkout `lumi_kernels/lib`, then the
image. Load through `_lib.so_path()`; `_lib.lib_path()` names build output.

To build the ROCr queue repairs, export `BASE`, an unused `OUT`, and a scratch
`RUN` directory, then submit `containers/build_jax_rocr.sbatch`. The image records
source, patch and runtime hashes under `/opt/lumi-jax-rocr`.
A Python-only refresh uses `build_jax_python.sbatch` and checks unchanged kernel hashes.

```bash
python3 -m unittest discover -s tests/host
bash tools/check_defs.sh
sbatch --no-requeue containers/test_lean_kernels_image.sbatch /path/to/candidate.sif
SIF=/path/to/candidate.sif sbatch --no-requeue tests/slurm/run-tests-base.sbatch
SIF=/path/to/candidate.sif sbatch --no-requeue containers/test_jax011.sbatch
SIF=/path/to/candidate.sif sbatch --no-requeue containers/test_jax_rccl_multinode.sbatch
```

Use a candidate-specific `VENV_DIR` for the base suite. The image gate verifies the
packaged CK library and numerical attention/gradient tests against JAX references.
The native gate runs GEMM autotuning cold; multi-node acceptance requires NET/OFI CXI.
Every pytest gate must find `[1-9][0-9]* passed`; all-skipped runs are not acceptance.
GPU claims require compute nodes.

Export `RUN` and `SIF` for `containers/test_rocr_profiler.sbatch`. Run
`containers/test_rocr_logic.sh "$RUN"` inside the image. ICICLE's gate additionally
accepts `BATCH=1|32` and runs three full scans. Compare saved trajectories with
`tools/compare_icicle_runs.py` before changing application pins.

## Change the base

Change `LAIF_PINS` and `LAIF_STAMP` together, using a tagged LAIF release. Run
`tools/probe_image.sh IMAGE` and `tools/compat_matrix.sh IMAGE`; submit
`containers/run_compat_matrix.sbatch IMAGE` for GPU compatibility checks.
Missing prerequisites or an unexecuted stage do not establish a pass. Keep
immutable rollback images and shared compilation caches outside the checkout.

Rebuild CK and its adapter after ABI changes. Container recipes must discard copied
`.so` files before rebuilding. The publication helper checks 512-byte SIF partition
alignment and mounts the image before changing a release link.

## License

MIT; see `LICENSE`.
