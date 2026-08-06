#!/bin/bash
# check_defs.sh [def ...] -- reject container definitions singularity cannot parse; all of
# containers/*.def by default. Run it before submitting a build:
#
#   tools/check_defs.sh
#
# Login-node safe and ~50 ms per def. Nothing is bootstrapped: the build aborts while
# resolving {{ BASE }}, which happens after the definition is parsed.
#
# The failure this exists to catch is invisible on inspection. Singularity strips leading
# whitespace before scanning for section headers, so an indented line of prose that begins
# with %post, %help, %environment and so on is read as a section declaration, and the whole
# file fails to parse with `invalid section(s) specified:` and an empty list.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

command -v singularity >/dev/null 2>&1 \
    || { echo "check_defs.sh: singularity is not on PATH" >&2; exit 2; }

defs=("$@")
[ ${#defs[@]} -gt 0 ] || defs=("$REPO"/containers/*.def)

status=0
for def in "${defs[@]}"; do
    [ -f "$def" ] || { echo "MISSING  $def"; status=1; continue; }
    out="$(singularity build --notest --force \
             --build-arg BASE=/nonexistent/check_defs-probe.sif \
             "$(mktemp -u "${TMPDIR:-/tmp}/check_defs-XXXXXX.sif")" "$def" 2>&1)"
    if printf '%s' "$out" | grep -q 'while parsing definition'; then
        echo "FAIL     $def"
        printf '%s\n' "$out" | grep -E '^FATAL' | sed 's/^/         /'
        grep -nE '^[[:space:]]+%[a-zA-Z]' "$def" \
            | sed 's/^/         reads as a section header, reflow it: /'
        status=1
    else
        echo "ok       $def"
    fi
done
exit $status
