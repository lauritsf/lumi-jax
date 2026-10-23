#!/bin/bash
# Image and package selection: explicit overrides, recorded venv, then defaults.

_JAX_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_JAX_REPO_DIR/tools/image_name.sh"
_JAX_IMG_DIR="$IMAGES_DIR"
_JAX_LAIF_DIR="$LAIF_DIR"

_JAX_DEFAULT_LINE_lean=0.11.2

_JAX_DEFAULT_VARIANT_lean=lean

_jax_venv_dir() {
    case "$1" in
        lean) echo "${VENV_DIR:-${LUMI_JAX_WORK:?set LUMI_JAX_WORK or VENV_DIR}/jax-venv}" ;;
        *) echo "Unsupported product: $1" >&2; return 2 ;;
    esac
}

# Read dist-info without executing a venv interpreter that points inside the image.
jax_line() {
    local kind="${1:?usage: jax_line <lean>}"
    local dflt recorded
    eval "dflt=\${_JAX_DEFAULT_LINE_$kind:-}"
    [ -n "$dflt" ] || { echo "jax_line: unknown kind '$kind'" >&2; return 2; }

    local _line="${LUMI_JAX_LINE:-${JAXVER:-}}"
    if [ -n "$_line" ]; then
        case "$_line" in
            091)  echo 0.9.1 ;;
            0110) echo 0.11.0 ;;
            0112) echo 0.11.2 ;;
            *)    echo "$_line" ;;
        esac
        return 0
    fi

    local venv; venv="$(_jax_venv_dir "$kind")"
    recorded="$venv/lumi-jax-line"
    if [ -f "$recorded" ]; then
        read -r recorded < "$recorded" || recorded=""
        case "$recorded" in
            0.9.1|0.11.0|0.11.2) echo "$recorded"; return 0 ;;
        esac
    fi

    if [ -d "$venv" ]; then
        local d
        for d in "$venv"/lib/python3.*/site-packages/jax-*.dist-info; do
            case "${d##*/jax-}" in
                0.9.1.dist-info)  echo 0.9.1;  return 0 ;;
                0.11.0.dist-info) echo 0.11.0; return 0 ;;
                0.11.2.dist-info) echo 0.11.2; return 0 ;;
            esac
        done
    fi

    echo "$dflt"
}

jax_variant() {
    local kind="${1:?usage: jax_variant <lean>}"
    local dflt recorded venv
    eval "dflt=\${_JAX_DEFAULT_VARIANT_$kind:-}"
    [ -n "$dflt" ] || { echo "jax_variant: unknown kind '$kind'" >&2; return 2; }


    if [ -n "${LUMI_JAX_VARIANT:-}" ]; then
        case "$LUMI_JAX_VARIANT" in
            lean|stage-net) echo "$LUMI_JAX_VARIANT"; return 0 ;;
            *) echo "jax_variant: unknown variant '$LUMI_JAX_VARIANT' (lean ships lean and stage-net)" >&2
               return 2 ;;
        esac
    fi

    venv="$(_jax_venv_dir "$kind")"
    if [ -f "$venv/lumi-jax-variant" ]; then
        read -r recorded < "$venv/lumi-jax-variant" || recorded=""
        case "$recorded" in
            lean|stage-net) echo "$recorded"; return 0 ;;
            kernels)        echo lean;       return 0 ;;
            rccl)           echo stage-net;  return 0 ;;
        esac
    fi

    if [ -d "$venv" ]; then echo stage-net; return 0; fi

    echo "$dflt"
}

jax_line_tag() {
    case "${1:?usage: jax_line_tag <line>}" in
        0.9.1)  echo 091 ;;
        0.11.0) echo 0110 ;;
        0.11.2) echo 0112 ;;
        091)    echo 091 ;;
        0110)   echo 0110 ;;
        0112)   echo 0112 ;;
        *) echo "jax_line_tag: unknown jax line '$1' (supported: 0.9.1, 0.11.0, 0.11.2)" >&2; return 2 ;;
    esac
}

# JAX 0.9.1 plugins use .post5; jaxlib must stay pinned with the Python package.
jax_pip_pins() {
    case "${1:?usage: jax_pip_pins <line>}" in
        0.9.1)  printf '%s\n' 'jax==0.9.1' 'jaxlib==0.9.1' \
                              'jax-rocm7-plugin==0.9.1.post5' 'jax-rocm7-pjrt==0.9.1.post5' ;;
        0.11.0) printf '%s\n' 'jax==0.11.0' 'jaxlib==0.11.0' \
                              'jax-rocm7-plugin==0.11.0' 'jax-rocm7-pjrt==0.11.0' ;;
        0.11.2) printf '%s\n' 'jax==0.11.2' 'jaxlib==0.11.2' \
                              'jax-rocm7-plugin==0.11.2' 'jax-rocm7-pjrt==0.11.2' ;;
        *) echo "jax_pip_pins: unknown jax line '$1' (supported: 0.9.1, 0.11.0, 0.11.2)" >&2; return 2 ;;
    esac
}

jax_resolve_image() {
    local kind="${1:?usage: jax_resolve_image <lean> [line]}"
    local line="${2:-}"
    local jaxver ours stock why

    case "$kind" in
        lean) ;;
        *) echo "jax_resolve_image: unknown kind '$kind'" >&2; return 2 ;;
    esac
    [ -n "$line" ] || line="$(jax_line "$kind")" || return 2
    jaxver="$(jax_line_tag "$line")" || return 2

    local variant; variant="$(jax_variant "$kind")" || return 2

    case "$kind" in
        lean) ours="$_JAX_IMG_DIR/lumi-jax-${variant}-j${jaxver}-latest.sif"
              stock="$_JAX_LAIF_DIR/lumi-multitorch-mpich-${LAIF_PINS}-${LAIF_STAMP}.sif"
              case "$variant" in
                  lean) why="ours: RCCL 2.27.7, no shims, CK attention prebuilt; jax $line" ;;
                  stage-net) why="ours: RCCL 2.27.7, no shims, no kernels; jax $line" ;;
              esac ;;
    esac

    if [ -n "${SIF:-}" ]; then
        echo "image: $SIF (SIF override)" >&2
        echo "$SIF"; return 0
    fi
    # Installation may choose a new image; execution must honor a matching venv record.
    if [ "${3:-run}" = run ]; then
        local venv recorded_line recorded_image recorded_variant
        venv="$(_jax_venv_dir "$kind")"
        if [ -f "$venv/lumi-jax-line" ] && [ -f "$venv/lumi-jax-image" ]; then
            read -r recorded_line < "$venv/lumi-jax-line"
            read -r recorded_image < "$venv/lumi-jax-image"
            recorded_variant="$kind"
            if [ -f "$venv/lumi-jax-variant" ]; then
                read -r recorded_variant < "$venv/lumi-jax-variant"
            fi
            if [ "$recorded_line" = "$line" ] && [ "$recorded_variant" = "$variant" ]; then
                [ -f "$recorded_image" ] || {
                    echo "FATAL: venv image missing: $recorded_image" >&2; return 1; }
                echo "image: $recorded_image (recorded by this venv)" >&2
                echo "$recorded_image"; return 0
            fi
        elif [ -d "$venv" ] && [ -n "$LUMI_JAX_WORK" ]; then
            local previous="$LUMI_JAX_WORK/images/lumi-jax-${variant}-j${jaxver}-latest.sif"
            if [ -f "$previous" ]; then
                echo "image: $previous (installed venv without a release record)" >&2
                echo "$previous"; return 0
            fi
        fi
    fi
    if [ -f "$ours" ]; then
        echo "image: $ours" >&2
        echo "       ($why)" >&2
        echo "$ours"; return 0
    fi
    echo "image: $stock" >&2
    if [ -z "$_JAX_IMG_DIR" ]; then
        echo "       (stock LAIF -- LUMI_JAX_WORK is unset, so our images cannot be located;" >&2
        echo "        export it or create tools/site.local.sh, see tools/image_name.sh)" >&2
    else
        echo "       (stock LAIF -- ours not found at $ours; build the matching product for prebuilt kernels)" >&2
    fi
    [ "$kind" = lean ] && [ "$variant" = lean ] && \
        echo "       (no prebuilt kernels there: build CK attention from csrc/flash)" >&2
    echo "$stock"
}
