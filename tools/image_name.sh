#!/bin/bash
# The image naming scheme, in one place.
#
#   source tools/image_name.sh          # from containers/build_*.sbatch
#   tools/image_name.sh fmha 227 0.11.0 # or run it to print a name
#
# THE SCHEME mirrors LAIF's grammar (lumi-multitorch-<layer>-<pins>-<stamp>.sif):
#
#   lumi-jax-<variant>-<pins>-<stamp>.sif
#
# <pins> keeps the LAIF base release's pin string VERBATIM and appends our deltas:
#
#   u24   ubuntu 24.04     ]
#   r70   rocm 7.0.2       ] verbatim from the LAIF base release,
#   f21   libfabric 2.1    ] never edited by us -- if any change,
#   m50   mpich 5.0.1      ] we are on a new base
#   t210  torch 2.10.0     ]
#   c227  rccl 2.27.7      } our leaf upgrade (c226 = the base's stock RCCL)
#   j091  jax 0.9.1        } our pin (j0110 = 0.11.0)
#
# The pin string names the base RELEASE, not the layer's contents -- t210 stays in the
# name of torch-free images, per LAIF's own convention; reading it as a manifest is the
# mistake. AOTriton/CK FMHA versions go in %labels, not the filename.
#
# THE -latest SYMLINK: build scripts resolve their BASE through
# lumi-jax-<variant>-j<jax>-latest.sif, so a rebuilt base is picked up without editing
# the layers above. Keyed by variant AND jax line, since both lines ship and an unkeyed
# name would silently mean one of them. A second LAIF base would make the key
# <variant, base, jax line>; only jax_image_{name,resolve,publish} below change then.

# The LAIF base release everything below is built on. Bump both together, never one.
LAIF_PINS="${LAIF_PINS:-u24r70f21m50t210}"
LAIF_STAMP="${LAIF_STAMP:-20260513_121430}"
LAIF_DIR="${LAIF_DIR:-/appl/local/laifs/containers/lumi-multitorch-${LAIF_PINS}-${LAIF_STAMP}}"

# THE ONE SITE KNOB: where built images and build caches live -- a scratch directory this
# project can write, holding images/, ck-fmha/ and causal-conv1d/. No default, because a
# guessed path resolves silently against someone else's scratch.
#
#   export LUMI_JAX_WORK=/scratch/project_<id>/<name>-cache
#
# tools/site.local.sh is sourced when present and is gitignored, so a checkout can carry
# its own value without it reaching the repo. Empty rather than fatal here: this file is
# sourced from interactive shells, where `:?` would kill the shell.
_JAX_SITE="$(dirname "${BASH_SOURCE[0]}")/site.local.sh"
if [ -f "$_JAX_SITE" ]; then . "$_JAX_SITE"; fi
LUMI_JAX_WORK="${LUMI_JAX_WORK:-}"
IMAGES_DIR="${IMAGES_DIR:-${LUMI_JAX_WORK:+$LUMI_JAX_WORK/images}}"

# PRODUCTS -- what a user installs (the names in install.sh and README):
#   lean       RCCL 2.27.7, no shims, standalone AOTriton + CK FMHA,
#              the four torch-free FFI kernels prebuilt          -> install.sh
#   torch      the above plus PyTorch, vLLM and the fifth kernel -> install-torch.sh
# STAGES -- build intermediates; the `stage-` prefix says "scaffolding":
#   stage-mpich      JAX on the LAIF mpich layer, base's stock RCCL 2.26.6
#   stage-net        + RCCL 2.27.7, no shims (installable: LUMI_JAX_VARIANT=stage-net)
#   stage-aotriton   + standalone AOTriton, no torch
#   stage-fmha       + standalone CK FMHA
#   stage-torch      JAX on the LAIF full layer, kernels not yet built
# stage-mpich and stage-net must stay separate variants: jax-rccl.def takes the first as
# its base and produces the second, so one shared name makes its `-latest` ambiguous.
jax_image_variants() { echo "stage-mpich stage-net stage-aotriton stage-fmha lean stage-torch torch"; }

# jax_image_pins <rccl 226|227> <jax 0.9.1|0.11.0>
# The underscore separates LAIF's verbatim base pins from our deltas.
jax_image_pins() {
    local rccl="${1:?rccl, e.g. 227}" jax="${2:?jax version, e.g. 0.11.0}"
    printf '%s_c%sj%s' "$LAIF_PINS" "$rccl" "${jax//./}"
}

# jax_image_name <variant> <rccl> <jax> [stamp] -- stamp defaults to now; pass one to
# name an image after its actual build time.
jax_image_name() {
    local variant="${1:?variant}" rccl="${2:?rccl}" jax="${3:?jax}" stamp="${4:-}"
    [ -n "$stamp" ] || stamp="$(date +%Y%m%d_%H%M%S)"
    printf 'lumi-jax-%s-%s-%s.sif' "$variant" "$(jax_image_pins "$rccl" "$jax")" "$stamp"
}

# jax_image_latest <variant> <jax> -- the stable path a build script resolves its BASE
# through.
jax_image_latest() {
    [ -n "$IMAGES_DIR" ] || {
        echo "set LUMI_JAX_WORK, or IMAGES_DIR (see tools/image_name.sh)" >&2; return 2; }
    printf '%s/lumi-jax-%s-j%s-latest.sif' \
        "$IMAGES_DIR" "${1:?variant}" "${2:?jax version}"
}

# jax_image_publish <built_sif> <variant> <jax> -- point the matching -latest at what
# was just built. Relative target, so the symlink survives the images directory being
# moved or bind-mounted elsewhere inside a container.
jax_image_publish() {
    local built="${1:?built sif}" variant="${2:?variant}" jax="${3:?jax version}"
    test -f "$built" || { echo "FATAL: cannot publish, no such file: $built" >&2; return 1; }
    local link; link="$(jax_image_latest "$variant" "${jax//./}")"
    ln -sfn "$(basename "$built")" "$link"
    echo "published  $link -> $(basename "$built")"
}

# jax_image_resolve <variant> <jax> -- the BASE a build script passes to --build-arg.
# Asserted here: a missing base under `singularity build` surfaces as a confusing
# bootstrap failure minutes into a reservation.
jax_image_resolve() {
    local variant="${1:?variant}" jax="${2:?jax version}"
    local link; link="$(jax_image_latest "$variant" "${jax//./}")"
    if [ ! -e "$link" ]; then
        echo "FATAL: no base for variant '$variant' at jax $jax" >&2
        echo "       expected: $link" >&2
        echo "       have:" >&2
        ls -1 "$IMAGES_DIR"/lumi-jax-*-latest.sif 2>/dev/null | sed 's/^/         /' >&2
        return 1
    fi
    readlink -f "$link"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case "${1:-}" in
        ""|-h|--help)
            echo "usage: tools/image_name.sh <variant> <rccl> <jax> [stamp]"
            echo "       tools/image_name.sh fmha 227 0.11.0"
            echo
            echo "variants: $(jax_image_variants)"
            echo "base:     lumi-multitorch-${LAIF_PINS}-${LAIF_STAMP}"
            echo
            echo "current -latest links:"
            ls -1 "$IMAGES_DIR"/lumi-jax-*-latest.sif 2>/dev/null | while read -r l; do
                printf '  %-34s -> %s\n' "$(basename "$l")" "$(basename "$(readlink "$l")")"
            done
            exit 0 ;;
    esac
    jax_image_name "$@"
    echo
fi
