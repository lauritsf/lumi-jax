#!/bin/bash
# Which container does this repo run in, and at which jax? One answer, in one place.
#
#   source tools/resolve_image.sh
#   SIF="$(jax_resolve_image lean)"     # JAX + the four torch-free kernels (install.sh)
#   SIF="$(jax_resolve_image torch)"     # torch + all five kernels          (install-torch.sh)
#   jax_pip_pins "$(jax_line lean)"     # the matching pip specs
#
# Two products, `lean` and `torch`; kind == variant for both. `lean` can also resolve
# to `stage-net`, the same base without the kernel layers (170 MB smaller):
# LUMI_JAX_VARIANT=stage-net ./install.sh.
#
# Two jax lines ship, default 0.11.0; LUMI_JAX_LINE=0.9.1 (or JAXVER) picks the other,
# here and in any script that sources this file. Both lines are first-class; every
# published number in this repo was taken on 0.9.1.
#
# THE VENV DECIDES, ONCE IT EXISTS: the line recorded there wins over the default,
# otherwise flipping the default would put every installed venv on the wrong image at
# the next sbatch. JAXVER still overrides -- an explicit request is not a mistake.
#
# Every caller shares this file because a venv is layered over the image's own /opt/venv:
# running it inside a different image mixes two site-packages trees and two RCCLs, and
# fails in ways that look like a code bug.
#
# Prefers OUR image, falls back to the stock LAIF one, printing which and why on
# stderr; `SIF` in the environment overrides both. The fallback is required: the image
# can be absent in a fresh clone or another project.

_JAX_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# LAIF_PINS/LAIF_STAMP/IMAGES_DIR live only in image_name.sh, so a base bump is one edit.
source "$_JAX_REPO_DIR/tools/image_name.sh"
_JAX_IMG_DIR="$IMAGES_DIR"
_JAX_LAIF_DIR="$LAIF_DIR"

_JAX_DEFAULT_LINE_lean=0.11.0
# Same default on both kinds: a default that differs per kind would hand two jax
# versions out of the same checkout.
_JAX_DEFAULT_LINE_torch=0.11.0

# Which image variant each kind means. Only `lean` has a choice.
_JAX_DEFAULT_VARIANT_lean=lean
_JAX_DEFAULT_VARIANT_torch=torch

# Where each kind's venv lives, and therefore where its recorded line is read from.
# VENV_DIR is honoured because install.sh honours it.
_jax_venv_dir() {
    case "$1" in
        lean) echo "${VENV_DIR:-$_JAX_REPO_DIR/jax-venv}" ;;
        torch) echo "${VENV_DIR:-$_JAX_REPO_DIR/jax-venv-torch}" ;;
    esac
}

# jax_line <lean|torch>
# Which jax line applies: JAXVER, else what the existing venv was built with, else the
# default for this kind. Dotted form, e.g. 0.11.0.
jax_line() {
    local kind="${1:?usage: jax_line <lean|torch>}"
    local dflt recorded
    eval "dflt=\${_JAX_DEFAULT_LINE_$kind:-}"
    [ -n "$dflt" ] || { echo "jax_line: unknown kind '$kind'" >&2; return 2; }

    # LUMI_JAX_LINE is the knob; JAXVER is an accepted alias. The tag spelling is
    # accepted too, because build scripts and image names use it -- a JAXVER that
    # resolved an image but not a pip pin would be a knob that half works.
    local _line="${LUMI_JAX_LINE:-${JAXVER:-}}"
    if [ -n "$_line" ]; then
        case "$_line" in
            091)  echo 0.9.1 ;;
            0110) echo 0.11.0 ;;
            *)    echo "$_line" ;;
        esac
        return 0
    fi

    local venv; venv="$(_jax_venv_dir "$kind")"
    recorded="$venv/lumi-jax-line"
    if [ -f "$recorded" ]; then
        # Only trust a line we can serve: a hand-edited or truncated file should not
        # silently select an image name that does not exist.
        read -r recorded < "$recorded" || recorded=""
        case "$recorded" in
            0.9.1|0.11.0) echo "$recorded"; return 0 ;;
        esac
    fi

    # A venv with no recorded line still has to resolve to the image it holds, or this
    # file creates the exact mismatch it prevents. Read the version off pip's dist-info
    # directory name, NOT by running the interpreter: the venv's python is a symlink
    # chain into the image and does not execute on the host.
    if [ -d "$venv" ]; then
        local d
        for d in "$venv"/lib/python3.*/site-packages/jax-*.dist-info; do
            case "${d##*/jax-}" in
                0.9.1.dist-info)  echo 0.9.1;  return 0 ;;
                0.11.0.dist-info) echo 0.11.0; return 0 ;;
            esac
        done
    fi

    echo "$dflt"
}

# jax_variant <lean|torch>
# Which image variant applies: LUMI_JAX_VARIANT, else what the existing venv was built
# against (sbatch scripts pass no argument, so without this a user who installed one
# variant would silently get the other), else the default.
jax_variant() {
    local kind="${1:?usage: jax_variant <lean|torch>}"
    local dflt recorded venv
    eval "dflt=\${_JAX_DEFAULT_VARIANT_$kind:-}"
    [ -n "$dflt" ] || { echo "jax_variant: unknown kind '$kind'" >&2; return 2; }

    # `torch` is one image; honouring the knob here would resolve a name nobody built.
    [ "$kind" = torch ] && { echo "$dflt"; return 0; }

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
            # Legacy variant names, still written in venvs on disk.
            kernels)        echo lean;       return 0 ;;
            rccl)           echo stage-net;  return 0 ;;
        esac
    fi

    # A venv with no recorded variant was built on what is now `stage-net`. Unlike the
    # line, this cannot be read off pip metadata: the two variants install identical
    # packages -- the difference is in the image.
    if [ -d "$venv" ]; then echo stage-net; return 0; fi

    echo "$dflt"
}

# jax_line_tag <line> -- the tag used in image names: 0.9.1 -> j091, 0.11.0 -> j0110.
jax_line_tag() {
    case "${1:?usage: jax_line_tag <line>}" in
        0.9.1)  echo 091 ;;
        0.11.0) echo 0110 ;;
        # Accept the tag spelling too, so `JAXVER=0110` does something sane.
        091)    echo 091 ;;
        0110)   echo 0110 ;;
        *) echo "jax_line_tag: unknown jax line '$1' (we ship 0.9.1 and 0.11.0)" >&2; return 2 ;;
    esac
}

# jax_pip_pins <line> -- the four pins, one per line, for pip.
#
# Not string interpolation over one version: 0.9.1's plugin and pjrt are `.post5` while
# 0.11.0's are not. jaxlib is pinned on both lines because flax/orbax resolve jax
# UPWARD given the chance.
jax_pip_pins() {
    case "${1:?usage: jax_pip_pins <line>}" in
        0.9.1)  printf '%s\n' 'jax==0.9.1' 'jaxlib==0.9.1' \
                              'jax-rocm7-plugin==0.9.1.post5' 'jax-rocm7-pjrt==0.9.1.post5' ;;
        0.11.0) printf '%s\n' 'jax==0.11.0' 'jaxlib==0.11.0' \
                              'jax-rocm7-plugin==0.11.0' 'jax-rocm7-pjrt==0.11.0' ;;
        *) echo "jax_pip_pins: unknown jax line '$1' (we ship 0.9.1 and 0.11.0)" >&2; return 2 ;;
    esac
}

# jax_resolve_image <lean|torch> [line]
# Prints the image path on stdout, and a one-line explanation on stderr.
jax_resolve_image() {
    local kind="${1:?usage: jax_resolve_image <lean|torch> [line]}"
    local line="${2:-}"
    local jaxver ours stock why

    case "$kind" in
        lean|torch) ;;
        *) echo "jax_resolve_image: unknown kind '$kind'" >&2; return 2 ;;
    esac
    [ -n "$line" ] || line="$(jax_line "$kind")" || return 2
    jaxver="$(jax_line_tag "$line")" || return 2

    local variant; variant="$(jax_variant "$kind")" || return 2

    case "$kind" in
        lean) ours="$_JAX_IMG_DIR/lumi-jax-${variant}-j${jaxver}-latest.sif"
              stock="$_JAX_LAIF_DIR/lumi-multitorch-mpich-${LAIF_PINS}-${LAIF_STAMP}.sif"
              case "$variant" in
                  lean) why="ours: RCCL 2.27.7, no shims, 4 torch-free kernels prebuilt; jax $line" ;;
                  stage-net) why="ours: RCCL 2.27.7, no shims, no kernels; jax $line" ;;
              esac ;;
        torch) ours="$_JAX_IMG_DIR/lumi-jax-torch-j${jaxver}-latest.sif"
              stock="$_JAX_LAIF_DIR/lumi-multitorch-full-${LAIF_PINS}-${LAIF_STAMP}.sif"
              why="ours: no shims, kernels prebuilt; jax $line" ;;
    esac

    if [ -n "${SIF:-}" ]; then
        echo "image: $SIF (SIF override)" >&2
        echo "$SIF"; return 0
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
        echo "       (stock LAIF -- ours not found at $ours; run the build scripts? shims apply)" >&2
    fi
    [ "$kind" = lean ] && [ "$variant" = lean ] && \
        echo "       (no prebuilt kernels there: build them from csrc/, or use the torch image)" >&2
    # The stock image carries no hipBLASLt stub. install.sh puts the all-reduce hook
    # into the venv, so the hang itself is covered.
    [ "$jaxver" = 091 ] || \
        echo "       (WARNING: jax $line on a stock LAIF image is UNGATED -- no hipBLASLt stub;" >&2
    [ "$jaxver" = 091 ] || \
        echo "        the first call into setMaxWorkspaceBytes would die. JAXVER=0.9.1 is the tested pair.)" >&2
    echo "$stock"
}
