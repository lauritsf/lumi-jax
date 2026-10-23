#!/bin/bash
# Image names and build publication for the selected LAIF release.

# Change the LAIF pin string and release stamp together.
LAIF_PINS="${LAIF_PINS:-u24r72f21m50t211}"
LAIF_STAMP="${LAIF_STAMP:-20260929_104918}"
LAIF_DIR="${LAIF_DIR:-/appl/local/laifs/containers/lumi-multitorch-${LAIF_PINS}-${LAIF_STAMP}}"

_JAX_SITE="$(dirname "${BASH_SOURCE[0]}")/site.local.sh"
_JAX_WORK_OVERRIDE="${LUMI_JAX_WORK:-}"
if [ -f "$_JAX_SITE" ]; then . "$_JAX_SITE"; fi
[ -z "$_JAX_WORK_OVERRIDE" ] || LUMI_JAX_WORK="$_JAX_WORK_OVERRIDE"
unset _JAX_WORK_OVERRIDE
LUMI_JAX_WORK="${LUMI_JAX_WORK:-}"
IMAGES_DIR="${IMAGES_DIR:-${LUMI_JAX_WORK:+$LUMI_JAX_WORK/images/${LAIF_PINS}-${LAIF_STAMP}}}"
BUILD_ARTIFACT_DIR="${BUILD_ARTIFACT_DIR:-${LUMI_JAX_WORK:+$LUMI_JAX_WORK/build/${LAIF_PINS}-${LAIF_STAMP}}}"

jax_image_variants() { echo "stage-mpich stage-net lean"; }

jax_image_pins() {
    local rccl="${1:?rccl, e.g. 227}" jax="${2:?jax version, e.g. 0.11.0}"
    printf '%s_c%sj%s' "$LAIF_PINS" "$rccl" "${jax//./}"
}

jax_image_name() {
    local variant="${1:?variant}" rccl="${2:?rccl}" jax="${3:?jax}" stamp="${4:-}"
    [ -n "$stamp" ] || stamp="$(date +%Y%m%d_%H%M%S)"
    printf 'lumi-jax-%s-%s-%s.sif' "$variant" "$(jax_image_pins "$rccl" "$jax")" "$stamp"
}

jax_image_latest() {
    [ -n "$IMAGES_DIR" ] || {
        echo "set LUMI_JAX_WORK, or IMAGES_DIR (see tools/image_name.sh)" >&2; return 2; }
    printf '%s/lumi-jax-%s-j%s-latest.sif' \
        "$IMAGES_DIR" "${1:?variant}" "${2:?jax version}"
}

# Loop mounts truncate to 512-byte boundaries; pad the filesystem partition, not the SIF.
_jax_sif_align_partition() {
    local sif="${1:?sif}" row id range start end size pad tmp rc
    row="$(singularity sif list "$sif" | awk -F'|' '$5 ~ /FS/ {print; exit}')"
    [ -n "$row" ] || { echo "align: no FS partition in $sif" >&2; return 1; }
    id="$(echo "$row" | awk -F'|' '{gsub(/ /,"",$1); print $1}')"
    range="$(echo "$row" | awk -F'|' '{gsub(/ /,"",$4); print $4}')"
    start="${range%-*}"; end="${range#*-}"
    size=$((end - start))
    pad=$(( (512 - size % 512) % 512 ))
    [ "$pad" -eq 0 ] && return 0
    tmp="$sif.part.$$"
    singularity sif dump "$id" "$sif" > "$tmp" || { rm -f "$tmp"; return 1; }
    # Verify the dump before replacing the original partition.
    [ "$(stat -c %s "$tmp")" -eq "$size" ] || { rm -f "$tmp"; return 1; }
    truncate -s $((size + pad)) "$tmp" || { rm -f "$tmp"; return 1; }
    singularity sif del "$id" "$sif" &&
        singularity sif add "$sif" "$tmp" \
            --datatype 4 --partfs 1 --parttype 2 --partarch 2 --groupid 1
    rc=$?
    rm -f "$tmp"
    [ "$rc" -eq 0 ] && echo "realigned  FS partition of $(basename "$sif") (+$pad pad bytes)"
    return "$rc"
}

jax_image_publish() {
    local built="${1:?built sif}" variant="${2:?variant}" jax="${3:?jax version}"
    test -f "$built" || { echo "FATAL: cannot publish, no such file: $built" >&2; return 1; }
    _jax_sif_align_partition "$built"
    singularity exec --no-home "$built" true 2>/dev/null || {
        echo "FATAL: $built does not mount; refusing to publish" >&2
        return 1
    }
    local link; link="$(jax_image_latest "$variant" "${jax//./}")"
    ln -sfn "$(basename "$built")" "$link"
    echo "published  $link -> $(basename "$built")"
}

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
