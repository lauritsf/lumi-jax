#!/bin/bash
# Install the XLA all-reduce workaround into one site-packages directory.
#
#   tools/install_allreduce_hook.sh <site-packages-dir> <python>
#
# Turns off XLA's Triton-generated intra-node all-reduce kernel, enabled upstream for
# every ROCm device with no arch test; it never completes on gfx90a, so pmap/shard_map
# collectives over >=2 GCDs in ONE process never return.
# Multi-node is unaffected -- XLA only takes that path when every rank is local.
#
# Two installers share it: the image's /opt/venv, and the venv install.sh layers on top
# -- which descends from the BASE interpreter and never sees the image's site-packages,
# so a hook only in the image is a hook that is not there.
#
# A .pth is the only mechanism that runs late enough and cannot be shadowed:
# site.addsitedir executes its `import` line after singularity injects the caller's
# environment and before jax reads XLA_FLAGS. %environment loses to the caller's own
# XLA_FLAGS, and Ubuntu's python3.12 ships a sitecustomize earlier on sys.path. The
# logic lives in a normal module so it can be read; the .pth just imports it.
set -euo pipefail

SITEDIR="${1:?usage: install_allreduce_hook.sh <site-packages-dir> <python>}"
PYTHON="${2:?usage: install_allreduce_hook.sh <site-packages-dir> <python>}"

test -d "$SITEDIR" || { echo "FATAL: no such site-packages dir: $SITEDIR"; exit 1; }

# The pair has to match. A hook written into a directory the interpreter never reads is
# the exact failure being fixed, so refuse rather than write a file and report success.
if ! "$PYTHON" -c "import sys; sys.exit(0 if '$SITEDIR' in sys.path else 1)"; then
    echo "FATAL: $PYTHON does not have $SITEDIR on sys.path -- the hook would never run."
    "$PYTHON" -c "import sys; [print('  ', p) for p in sys.path]"
    exit 1
fi

cat > "$SITEDIR/_lumi_xla_flags.py" <<'PY'
"""Keep XLA's broken intra-node all-reduce kernel off on gfx90a.

The kernel is enabled for every ROCm device upstream with no arch test and never
completes on gfx90a: without this, jax.pmap / shard_map collectives over two or more
GCDs in one process never return.
The flag is PREPENDED, so a value the caller also sets wins -- XLA takes the last
occurrence of a repeated flag -- and setting LUMI_JAX_NO_ALLREDUCE_WORKAROUND=1 removes
this behaviour entirely.
"""
import os

_FLAG = "--xla_gpu_unsupported_use_all_reduce_one_shot_kernel=false"

if os.environ.get("LUMI_JAX_NO_ALLREDUCE_WORKAROUND", "") != "1":
    _cur = os.environ.get("XLA_FLAGS", "")
    # Substring test on the flag NAME, not the whole assignment: a caller who wrote
    # `=true` has made a deliberate choice and must not have it prepended away.
    if "xla_gpu_unsupported_use_all_reduce_one_shot_kernel" not in _cur:
        os.environ["XLA_FLAGS"] = (_FLAG + " " + _cur).strip()
PY

# zz- so it sorts last among .pth files: env edits after whatever else runs.
echo "import _lumi_xla_flags" > "$SITEDIR/zz-lumi-xla-flags.pth"

# Verify it actually RUNS, in the interpreter that matters, under both ways a caller can
# set the variable: a written file proves nothing about whether the hook executes.
"$PYTHON" -c "
import os
assert '--xla_gpu_unsupported_use_all_reduce_one_shot_kernel=false' in os.environ.get('XLA_FLAGS',''), os.environ.get('XLA_FLAGS')
print('pth hook: clean env  ->', os.environ['XLA_FLAGS'])"
XLA_FLAGS="--xla_gpu_autotune_level=4" "$PYTHON" -c "
import os
v = os.environ['XLA_FLAGS']
assert 'one_shot_kernel=false' in v and 'autotune_level=4' in v, v
print('pth hook: caller set ->', v)"
LUMI_JAX_NO_ALLREDUCE_WORKAROUND=1 "$PYTHON" -c "
import os
assert 'one_shot' not in os.environ.get('XLA_FLAGS',''), os.environ.get('XLA_FLAGS')
print('pth hook: opt-out    -> respected')"
