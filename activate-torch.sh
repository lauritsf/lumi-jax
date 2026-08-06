# Source from inside the LAIF *full* container. Activates jax-venv-torch (JAX layered
# over the container's /opt/venv) and sets the same shim env as activate.sh.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_REPO_DIR="$_SCRIPT_DIR"
_VENV_DIR="${VENV_DIR:-$_REPO_DIR/jax-venv-torch}"
_SHIM="$_REPO_DIR/shim"

export LD_LIBRARY_PATH="$_SHIM:${LD_LIBRARY_PATH:-}"
# install-torch.sh drops the stub when the image's RCCL exports ncclCommWindow*;
# preloading a missing file makes every exec print a loader error.
[ -e "$_SHIM/librccl_window_stub.so" ] && \
    export LD_PRELOAD="$_SHIM/librccl_window_stub.so${LD_PRELOAD:+:$LD_PRELOAD}"
export NCCL_WIN_ENABLE=0
# shellcheck source=/dev/null
source "$_VENV_DIR/bin/activate"
