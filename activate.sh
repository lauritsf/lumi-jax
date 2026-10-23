# Source inside the matching image. VENV_DIR overrides the product default.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_SCRIPT_DIR/tools/resolve_image.sh"
_VENV_DIR="$(_jax_venv_dir "lean")"
if [ -f "$_VENV_DIR/lumi-jax-native-rocm" ]; then
    # Exclude compatibility libraries left by a venv built on an older base.
    _LD=""
    IFS=: read -ra _LD_PARTS <<< "${LD_LIBRARY_PATH:-}"
    for _p in "${_LD_PARTS[@]}"; do
        case "$_p" in ""|"$_SCRIPT_DIR/shim") continue ;; esac
        _LD="${_LD:+$_LD:}$_p"
    done
    export LD_LIBRARY_PATH="$_LD"
    _PRELOAD=""
    _PRELOAD_PARTS="${LD_PRELOAD:-}"
    _PRELOAD_PARTS="${_PRELOAD_PARTS//:/ }"
    for _p in $_PRELOAD_PARTS; do
        case "$_p" in "$_SCRIPT_DIR/shim/librccl_window_stub.so") continue ;; esac
        _PRELOAD="${_PRELOAD:+$_PRELOAD }$_p"
    done
    if [ -n "$_PRELOAD" ]; then export LD_PRELOAD="$_PRELOAD"; else unset LD_PRELOAD; fi
    unset _LD _LD_PARTS _PRELOAD _PRELOAD_PARTS _p

else
    export LD_LIBRARY_PATH="$_SCRIPT_DIR/shim:${LD_LIBRARY_PATH:-}"
    [ ! -e "$_SCRIPT_DIR/shim/librccl_window_stub.so" ] || \
        export LD_PRELOAD="$_SCRIPT_DIR/shim/librccl_window_stub.so${LD_PRELOAD:+:$LD_PRELOAD}"
    export NCCL_WIN_ENABLE=0
fi

_PP=""
IFS=: read -ra _PP_PARTS <<< "${PYTHONPATH:-}"
for _p in "${_PP_PARTS[@]}"; do
    case "$_p" in ""|/opt/lumi-jax|/opt/venv/lib/python3.12/site-packages) continue ;; esac
    _PP="${_PP:+$_PP:}$_p"
done
if [ -n "$_PP" ]; then export PYTHONPATH="$_PP"; else unset PYTHONPATH; fi
unset _PP _PP_PARTS _p

# shellcheck source=/dev/null
source "$_VENV_DIR/bin/activate"
