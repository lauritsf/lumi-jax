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
export LD_PRELOAD="$_SHIM/librccl_window_stub.so${LD_PRELOAD:+:$LD_PRELOAD}"
export NCCL_WIN_ENABLE=0
# shellcheck source=/dev/null
source "$_VENV_DIR/bin/activate"
