# Source from inside the LAIF container. Sets the shim env and activates the venv
# created by install.sh.

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_VENV_DIR="${VENV_DIR:-$_SCRIPT_DIR/jax-venv}"
_SHIM="$_SCRIPT_DIR/shim"

# shim/ first on LD_LIBRARY_PATH so its sonames win over the container's; the RCCL
# window stub via LD_PRELOAD so its no-op ncclCommWindow* symbols satisfy the loader.
# NCCL_WIN_ENABLE=0 disables RCCL's symmetric-memory window path -- the only thing that
# would call into the stub. Standard collectives never use windows, so this is safe.
export LD_LIBRARY_PATH="$_SHIM:${LD_LIBRARY_PATH:-}"
# install.sh drops the stub when the image's RCCL exports ncclCommWindow*; preloading a
# missing file makes every exec print a loader error.
[ -e "$_SHIM/librccl_window_stub.so" ] && \
    export LD_PRELOAD="$_SHIM/librccl_window_stub.so${LD_PRELOAD:+:$LD_PRELOAD}"
export NCCL_WIN_ENABLE=0

# Drop the lean image's PYTHONPATH=/opt/lumi-jax: PYTHONPATH precedes site-packages, so
# the image's baked lumi_kernels shadows this venv's editable checkout -- edits and
# locally built kernels are silently ignored. The image's prebuilt .so files still load;
# lumi_kernels/_lib.py searches /opt/lumi-jax/lumi_kernels/lib by path, not by import.
_PP=""
IFS=: read -ra _PP_PARTS <<< "${PYTHONPATH:-}"
for _p in "${_PP_PARTS[@]}"; do
    case "$_p" in ""|/opt/lumi-jax) continue ;; esac
    _PP="${_PP:+$_PP:}$_p"
done
if [ -n "$_PP" ]; then export PYTHONPATH="$_PP"; else unset PYTHONPATH; fi
unset _PP _PP_PARTS _p

# shellcheck source=/dev/null
source "$_VENV_DIR/bin/activate"
