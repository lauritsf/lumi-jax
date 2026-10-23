# AI Agent Guidelines

Adopt the persona of a staff-level software engineer: operational, lean, strictly
professional. Optimize for the reader's time.

## Repo map

```
install.sh / activate.sh        install and enter $LUMI_JAX_WORK/jax-venv over /opt/venv
lumi_kernels/                    the Python package users import; lib/ holds built .so files (gitignored)
csrc/                            CK attention HIP sources and build script
tests/                           pytest suite (markers gpu/multigpu/so(name) auto-skip);
                                 slurm/ holds the test gates
containers/                      .def + build_*.sbatch per image variant, test_*.sbatch acceptance gates
tools/                           resolve_image.sh picks image + pip pins from one knob;
                                 baselines/ holds ABI probe outputs to diff a new base against
slurm/example.sbatch             template for user jobs in the container
```

Images live outside the repo, under `$LUMI_JAX_WORK/images/`.

## Verifying changes

- The gates, submitted from the repo root: `sbatch tests/slurm/run-tests-base.sbatch`
  (lean image, 8 GCDs). Set `SBATCH_ACCOUNT`; success ends with `BASE SUITE OK`.
- After a container rebuild, run the matching `containers/test_*.sbatch` acceptance gate.
- Off-GPU, `pytest tests -q` in the venv collects cleanly and runs host tests; device
  tests skip missing prerequisites. A GPU gate must check its devices explicitly.
- Builds and gates outlive a session, so a clean tree is not a verified one. Start with
  `git log -1` and `squeue -u $USER`: a session that left work running lists the job
  numbers and what each must show in the newest commit body.

## Repo invariants

- Gates must not be able to pass while testing nothing: every pytest marker auto-skips on a
  missing prerequisite, so each gate greps its output for `[1-9][0-9]* passed`. Keep the
  guard when touching a gate; never assert a fixed test count.
- GPU claims require a compute node. `singularity exec` on a login node proves nothing
  about GPU behavior.
- Load kernels through `_lib.so_path()` (per-library search), never `_lib.lib_path()`
  (always the checkout's build output).

## Documentation

- Use plain language: say "tests" instead of "qualification".

- Every sentence must change what the reader does next; delete it otherwise.
- `README.md` and `AGENTS.md` are the only top-level documents. Never create additional Markdown files
  (STATUS.md, CONTAINER.md, NOTES.md, TODO.md, design docs, handoffs). Kernel-local build
  notes may live in `csrc/<kernel>/README.md`.
- Document the present, not the past. No history, dead ends, deprecated paths, retractions,
  phase/ticket numbers, or recreated historical logs.
- Zero defensive justification. State what the software does, its requirements, and its
  constraints. Explain *why* only when it affects a configuration choice the reader must
  make today.
- Traps get the fix first: the exact command or code, then at most one sentence of root
  cause.
- Separate user from maintainer concerns. Install/use instructions never mix with
  build-tree internals, CI, or billing accounts.
- Do not document what is self-evident from the code or quick to discover.

**Anti-pattern:**

> Historically, attention on LUMI ran through the XLA math lowering, which materializes the
> full score matrix and OOMs at S≈8192. We tried several dead ends, like kvax and Pallas,
> but they didn't work for various reasons. Therefore, we chose to bind prebuilt gfx90a
> kernels via XLA FFI to achieve O(S) memory instead.

**Target:**

> The native XLA math lowering scales quadratically and OOMs at S=8192. We bind prebuilt
> gfx90a kernels via XLA FFI, which hold peak HBM flat.

## Context placement

- Context lives at the point of use: a build constraint in that build script's header, a
  kernel gotcha next to the code that implements it — one place, not duplicated into
  README.
- No cross-reference sprawl. Do not write "see FILE.md" pointers from code to docs or
  between docs; they rot. If a fact is needed in two places, put it in the one closer to
  where it bites.

## Code comments

- Comments state constraints the code cannot show — the *what* and *how* of complex logic
  only. Never narrate the next line, justify a change to a reviewer, or document standard
  library behavior.
- No provenance: no phase numbers, dates, or pointers to documents.

**Anti-pattern:**

```bash
# Check if the image resolves the library. We do this by launching singularity and
# trying to load it via ctypes, instead of ldconfig, because ldconfig uses the SONAME.
if loads_in_image librocm_smi64.so.1; then
    rm -f "$SHIM_DIR/librocm_smi64.so.1"   # succeeded, so remove the shim
```

**Target:**

```bash
# Probe the image loader directly; `ldconfig -p` caches by SONAME, which hides symlinks.
if loads_in_image librocm_smi64.so.1; then
    rm -f "$SHIM_DIR/librocm_smi64.so.1"
```

## Commits and PRs

- Summary line: imperative, states the change, not the journey.
- Body: bullets of what changed and any behavioral consequence. Never "In this PR",
  never a narration of attempts, never a restatement of the diff.
