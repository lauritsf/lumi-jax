#!/bin/bash
# Install the XLA flag workarounds into one site-packages directory.
#
#   tools/install_xla_flags_hook.sh <site-packages-dir> <python>
#
# Two flags, both prepended to XLA_FLAGS at python startup:
#
# 1. --xla_gpu_unsupported_use_all_reduce_one_shot_kernel=false
#    XLA's Triton-generated intra-node all-reduce kernel is enabled upstream for every
#    ROCm device with no arch test; it never completes on gfx90a, so pmap/shard_map
#    collectives over >=2 GCDs in ONE process never return. Multi-node is unaffected --
#    XLA only takes that path when every rank is local.
#
# 2. --xla_gpu_enable_command_buffer=   (empty: every command-buffer type off)
#    Command buffers are unreliable on this ROCm stack. Our FFI kernels are not the
#    trigger -- custom calls are excluded from capture unless
#    --legacy_command_buffer_custom_call_targets names them -- so this cannot be scoped
#    to the kernels and has to be off process-wide.
#
# Install in both /opt/venv and the writable environment: the latter descends
# from the base interpreter and does not inherit the image startup hook.
#
# A .pth is the only mechanism that runs late enough and cannot be shadowed:
# site.addsitedir executes its `import` line after singularity injects the caller's
# environment and before jax reads XLA_FLAGS. %environment loses to the caller's own
# XLA_FLAGS, and Ubuntu's python3.12 ships a sitecustomize earlier on sys.path. The
# logic lives in a normal module so it can be read; the .pth just imports it.
set -euo pipefail

SITEDIR="${1:?usage: install_xla_flags_hook.sh <site-packages-dir> <python>}"
PYTHON="${2:?usage: install_xla_flags_hook.sh <site-packages-dir> <python>}"

test -d "$SITEDIR" || { echo "FATAL: no such site-packages dir: $SITEDIR"; exit 1; }

# The pair has to match. A hook written into a directory the interpreter never reads is
# the exact failure being fixed, so refuse rather than write a file and report success.
if ! "$PYTHON" -c "import sys; sys.exit(0 if '$SITEDIR' in sys.path else 1)"; then
    echo "FATAL: $PYTHON does not have $SITEDIR on sys.path -- the hook would never run."
    "$PYTHON" -c "import sys; [print('  ', p) for p in sys.path]"
    exit 1
fi

cat > "$SITEDIR/_lumi_xla_flags.py" <<'PY'
"""Keep the XLA settings gfx90a needs on, whatever the caller's XLA_FLAGS.

Each flag is PREPENDED, so a value the caller also sets wins -- XLA takes the last
occurrence of a repeated flag -- and each has its own opt-out variable.

all-reduce: XLA's one-shot kernel is enabled for every ROCm device upstream with no arch
test and never completes on gfx90a, so jax.pmap / shard_map collectives over two or more
GCDs in one process never return.

command buffers: unreliable on this ROCm stack. Custom calls are excluded from capture
unless --legacy_command_buffer_custom_call_targets names them, so the FFI kernels are not
the trigger and this cannot be narrowed to them.
"""
import os
import importlib.metadata as _metadata

# AMD SMI and ROCm SMI export incompatible private amd::smi C++ symbols.
# JAX 0.11.2 loads both; each must resolve its internals in its own lookup scope.
try:
    _needs_smi_isolation = _metadata.version("jax-rocm7-plugin") == "0.11.2"
except _metadata.PackageNotFoundError:
    _needs_smi_isolation = False
if _needs_smi_isolation:
    import ctypes as _ctypes
    _smi_binding_mode = os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_DEEPBIND
    _smi_handles = [
        _ctypes.CDLL("/opt/rocm/lib/" + name, mode=_smi_binding_mode)
        for name in ("librocm_smi64.so.1", "libamd_smi.so.26")
    ]

# (flag, the flag NAME to test for, opt-out variable)
_FLAGS = [
    ("--xla_gpu_unsupported_use_all_reduce_one_shot_kernel=false",
     "xla_gpu_unsupported_use_all_reduce_one_shot_kernel",
     "LUMI_JAX_NO_ALLREDUCE_WORKAROUND"),
    ("--xla_gpu_enable_command_buffer=",
     "xla_gpu_enable_command_buffer",
     "LUMI_JAX_NO_COMMAND_BUFFER_WORKAROUND"),
]

for _flag, _name, _optout in _FLAGS:
    if os.environ.get(_optout, "") == "1":
        continue
    _cur = os.environ.get("XLA_FLAGS", "")
    # Substring test on the flag NAME, not the whole assignment: a caller who set their
    # own value has made a deliberate choice and must not have it prepended away.
    if _name not in _cur:
        os.environ["XLA_FLAGS"] = (_flag + " " + _cur).strip()
PY

# zz- so it sorts last among .pth files: env edits after whatever else runs.
echo "import _lumi_xla_flags" > "$SITEDIR/zz-lumi-xla-flags.pth"

# Verify it actually RUNS, in the interpreter that matters, under every way a caller can
# set the variable: a written file proves nothing about whether the hook executes.
"$PYTHON" -c "
import os
v = os.environ.get('XLA_FLAGS', '')
assert '--xla_gpu_unsupported_use_all_reduce_one_shot_kernel=false' in v, v
assert '--xla_gpu_enable_command_buffer=' in v, v
print('pth hook: clean env  ->', v)"
"$PYTHON" -c "
import _lumi_xla_flags as hook
if hook._needs_smi_isolation:
    assert len(hook._smi_handles) == 2
    print('pth hook: native SMI libraries bound locally')"
XLA_FLAGS="--xla_gpu_autotune_level=4" "$PYTHON" -c "
import os
v = os.environ['XLA_FLAGS']
assert 'one_shot_kernel=false' in v and 'enable_command_buffer=' in v \
   and 'autotune_level=4' in v, v
print('pth hook: caller set ->', v)"
# A caller's own value for either flag survives: ours is prepended, XLA takes the last.
XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION" "$PYTHON" -c "
import os
v = os.environ['XLA_FLAGS']
assert v.count('enable_command_buffer') == 1 and 'FUSION' in v, v
print('pth hook: caller CB  -> respected')"
LUMI_JAX_NO_ALLREDUCE_WORKAROUND=1 LUMI_JAX_NO_COMMAND_BUFFER_WORKAROUND=1 "$PYTHON" -c "
import os
v = os.environ.get('XLA_FLAGS', '')
assert 'one_shot' not in v and 'command_buffer' not in v, v
print('pth hook: opt-out    -> respected')"
