#!/bin/bash
# probe_image.sh <SIF> [...] -- dump the ABI surface for comparison with tools/baselines/.
# GPU behavior requires the compute-node acceptance gates.

set -uo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $(basename "$0") <image.sif> [more.sif ...]" >&2
    exit 2
fi

# Bump when a field's *meaning* changes, so a stale baseline diffs as a header line.
PROBE_VERSION=4

# The in-container probe. Quoted heredoc: nothing here is expanded by the host.
read -r -d '' PROBE_BODY <<'PROBE' || true
# No `pipefail`: grep -q SIGPIPEs its producer, which would turn "present" into "absent".
set -u

SP=/opt/venv/lib/python3.12/site-packages   # LAIF's venv; absent in mpich/rocm layers

# Fixed-width keys so a diff shows changed values, not reflowed columns.
kv()   { printf '%-34s %s\n' "$1" "$2"; }
kv_or() { printf '%-34s %s\n' "$1" "${2:-ABSENT}"; }
sec()  { printf '\n[%s]\n' "$1"; }

sonames_of() {
    local out
    out=$(ls -1 $1 2>/dev/null | xargs -r -n1 basename | sort | tr '\n' ' ')
    echo "${out% }"
}

# Via ldconfig, not guessed dirs: a guessed path makes a missing file look like a missing symbol.
lib_any() {
    ldconfig -p 2>/dev/null | awk -v b="$1.so" \
        'index($1, b) == 1 {print $NF; exit}'
}

# Cache first, then dlopen: a compat symlink is indexed under its target's SONAME.
lib_has() {
    local n
    n=$(ldconfig -p 2>/dev/null | awk -v s="$1" '$1 == s' | grep -c .)
    if [ "${n:-0}" -gt 0 ]; then echo yes; return; fi
    if python3 -c "import ctypes,sys; ctypes.CDLL('$1')" 2>/dev/null; then
        echo yes; return
    fi
    echo no
}

# BASE.so* beside the resolved lib; the versioned suffix encodes the ROCm version.
lib_files() {
    local p
    p=$(lib_any "$1") || true
    [ -n "$p" ] || return 0
    sonames_of "$(dirname "$p")/$1.so*"
}

dpkg_ver() { dpkg-query -W -f='${Version}' "$1" 2>/dev/null; }

pip_ver() {
    "$PY" - "$1" <<'EOF' 2>/dev/null
import importlib.metadata as m, sys
try: print(m.version(sys.argv[1]))
except Exception: pass
EOF
}

# Cached: some libraries are ~1 GB and queried several times.
SYMCACHE=$(mktemp -d)
trap 'rm -rf "$SYMCACHE"' EXIT
symdump() {
    local key="$SYMCACHE/$(echo "$1" | tr -c 'A-Za-z0-9' '_')"
    if [ ! -e "$key" ]; then
        nm -D --defined-only "$1" 2>/dev/null \
            | c++filt \
            | sed 's/^[0-9a-f]* [A-Za-z] //' \
            | sort -u > "$key"
    fi
    echo "$key"
}

# Signatures verbatim: the signature *is* the ABI contract for symbols we link.
syms() {
    [ -n "${1:-}" ] && [ -e "${1:-}" ] || return 0
    grep -E "$2" "$(symdump "$1")"
}

sym_count() {
    [ -e "$1" ] || { echo 0; return; }
    syms "$1" "$2" | grep -c . || true
}

first_match() { ls -1 $1 2>/dev/null | head -1; }

# Counts rather than short-circuiting (pipefail note above).
has_string() {
    [ -n "${1:-}" ] && [ -e "${1:-}" ] || { echo ABSENT; return; }
    local n
    n=$(strings "$1" 2>/dev/null | grep -cw -- "$2")
    [ "${n:-0}" -gt 0 ] && echo present || echo ABSENT
}

has_sym() {
    [ "$(sym_count "${1:-}" "$2")" -gt 0 ] && echo present || echo ABSENT
}

# /opt/jax-venv first: where jax is layered over /opt/venv, /opt/venv's python misses it.
for c in /opt/jax-venv/bin/python /opt/venv/bin/python; do
    [ -x "$c" ] && { PY="$c"; break; }
done
PY="${PY:-$(command -v python3)}"

sec os
kv os.pretty      "$(. /etc/os-release; echo "$PRETTY_NAME")"
kv os.version_id  "$(. /etc/os-release; echo "$VERSION_ID")"
kv glibc.version  "$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
kv python.path    "$PY"
kv python.version "$("$PY" -V 2>&1 | awk '{print $2}')"
kv gcc.version    "$(gcc -dumpfullversion 2>/dev/null)"

sec rocm
kv_or rocm.version      "$(cat /opt/rocm/.info/version 2>/dev/null)"
kv_or rocm.core_pkg     "$(dpkg_ver rocm-core)"
kv_or hip.pkg           "$(dpkg_ver hip-runtime-amd)"
kv_or hsa.sonames       "$(lib_files libhsa-runtime64)"
kv_or hipblaslt.sonames "$(lib_files libhipblaslt)"

# The three runtime shims: is each still needed on this image?
sec shims
# 1. librocm_smi64.so.1 -- the jax plugin links the old soname.
kv_or rocm_smi.sonames  "$(lib_files librocm_smi64)"
kv    rocm_smi.has_so1  "$(lib_has librocm_smi64.so.1)"
# 2. libdw.so.1 -- pulled in by the plugin via elfutils; copied from Cray PE today.
kv_or libdw.sonames     "$(lib_files libdw)"
kv    libdw.has_so1     "$(lib_has libdw.so.1)"
kv_or libdw.pkg         "$(dpkg_ver libdw1t64)"
kv_or libdw.apt_candidate "$(apt-cache policy libdw1t64 2>/dev/null | awk '/Candidate:/{print $2}')"
# 3. librccl_window_stub.so -- needed while RCCL lacks the ncclCommWindow* API.
RCCL_SO=$(lib_any librccl)
# dpkg can lag the file on disk (in-place replacement); trust rccl.lib_version.
kv_or rccl.pkg          "$(dpkg_ver rccl)"
kv_or rccl.lib_version  "$(basename "$(readlink -f "${RCCL_SO:-/nonexistent}")" 2>/dev/null \
                            | sed -n 's/^librccl\.so\.//p')"
kv_or rccl.sonames      "$(lib_files librccl)"
kv_or rccl.path         "$RCCL_SO"
kv    rccl.window_syms  "$(sym_count "$RCCL_SO" '^ncclCommWindow')"
syms "$RCCL_SO" '^ncclCommWindow' | while read -r s; do kv rccl.window_sym "$s"; done
# "yes" only when librccl exists and lacks the API; no librccl is a different problem.
if [ -z "$RCCL_SO" ]; then
    kv rccl.stub_needed unknown
else
    kv rccl.stub_needed \
       "$([ "$(sym_count "$RCCL_SO" '^ncclCommWindow')" -eq 0 ] && echo yes || echo no)"
fi

# Newer JAX plugins require the native hipBLASLt workspace API.
sec plugin-gate
HIPBLASLT=$(lib_any libhipblaslt)
kv_or hipblaslt.path "$HIPBLASLT"
if [ -z "$HIPBLASLT" ]; then
    kv hipblaslt.setMaxWorkspaceBytes unknown
else
    kv hipblaslt.setMaxWorkspaceBytes "$(has_sym "$HIPBLASLT" 'GemmInstance::setMaxWorkspaceBytes')"
fi

sec net
kv_or mpich.pkg        "$(dpkg_ver mpich)"
# Values can contain colons (ch4:ofi, 18:1:6), so strip labels with sed, not cut.
kv_or mpich.version    "$(mpichversion 2>/dev/null | sed -n 's/^MPICH Version: *//p')"
kv_or mpich.device     "$(mpichversion 2>/dev/null | sed -n 's/^MPICH Device: *//p')"
kv_or mpich.abi        "$(mpichversion 2>/dev/null | sed -n 's/^MPICH ABI: *//p')"
kv_or libfabric.pkg    "$(dpkg_ver libfabric)"
kv_or libfabric.sonames "$(lib_files libfabric)"
# Prefer the distro binary: the full image's venv shadows fi_info with a broken wrapper.
FI_INFO=/usr/bin/fi_info
[ -x "$FI_INFO" ] || FI_INFO=$(command -v fi_info 2>/dev/null)
kv_or libfabric.fi_info    "$FI_INFO"
kv_or libfabric.providers "$([ -x "${FI_INFO:-}" ] && "$FI_INFO" -l 2>/dev/null \
      | sed -n 's/^\([a-z0-9_]*\):$/\1/p' | sort -u | tr '\n' ' ' | sed 's/ $//')"
# Do NOT detect kdreg2 by grepping "kdreg2": the -FI_ENOSYS stub has the same names.
# "/dev/kdreg2" arrives only via linux/kdreg2.h, the header whose absence selects the stub.
FAB_SO=$(lib_any libfabric)
kv    libfabric.kdreg2_compiled "$(has_string "$FAB_SO" /dev/kdreg2)"
# Three-valued; reflects the host's kdreg2 module too, so not a pure image property.
kdreg2_runtime() {
    [ -x "${FI_INFO:-}" ] || { echo unknown; return; }
    local out
    out=$(FI_MR_CACHE_MONITOR=kdreg2 FI_LOG_LEVEL=warn "$FI_INFO" -p shm 2>&1)
    case "$out" in
        *"kdreg2 monitor not available"*) echo rejected ;;
        *"provider:"*)                    echo accepted ;;
        *)                                echo unknown  ;;
    esac
}
kv    libfabric.kdreg2_runtime "$(kdreg2_runtime)"
kv_or aws_ofi_nccl.pkg "$(dpkg_ver aws-ofi-nccl)"
# RCCL <=2.26 loads librccl-net.so, 2.27+ libnccl-net.so; a missing name falls back to
# NET/Socket -- correct over TCP, invisible to a psum test. Looked up by filename, as RCCL
# does, since ldconfig indexes a compat symlink under its target's SONAME.
plugin_path() {
    local n="$1" d p
    p=$(ldconfig -p 2>/dev/null | awk -v s="$n" '$1 == s {print $NF; exit}')
    [ -n "$p" ] && { echo "$p"; return; }
    for d in /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib /opt/rocm/lib; do
        [ -e "$d/$n" ] && { echo "$d/$n"; return; }
    done
}
kv_or ofi_plugin.rccl_name "$(plugin_path librccl-net.so)"
kv_or ofi_plugin.nccl_name "$(plugin_path libnccl-net.so)"

sec python-pins
for dist in jax jaxlib jax-rocm7-plugin jax-rocm7-pjrt; do
    kv_or "pin.$dist" "$(pip_ver "$dist")"
done

# Kernel binaries csrc/ links: a changed signature here is a broken csrc/ target.
sec kernel-abi

FA_SO=$(first_match "${CK_FMHA_ROOT:-/opt/ck-fmha}/lib/libck_fmha.so")
kv_or flash.path "$FA_SO"
# The two dispatchers csrc/flash calls; bwd is a template, so it demangles with a return type.
syms "$FA_SO" '^(float )?fmha_(fwd|bwd)(<[0-9]+>)?\(' | while read -r s; do kv flash.sym "$s"; done
# A drop here means the base pruned kernel configs, which no signature check sees.
kv    flash.fwd_kernel_instances "$(sym_count "$FA_SO" '^(float )?fmha_fwd_<')"
kv    flash.bwd_kernel_instances "$(sym_count "$FA_SO" '^(float )?fmha_bwd_<')"

PROBE

status=0
for SIF in "$@"; do
    if [ ! -e "$SIF" ]; then
        echo "probe_image.sh: no such image: $SIF" >&2
        status=1
        continue
    fi

    # Host-side identity: read the sha from LAIF's sidecar, never rehash a multi-GB file.
    name=$(basename "$SIF")
    sha=$(awk -v n="$name" '$2 == n || $2 == "*"n {print $1}' \
            "$(dirname "$SIF")"/*.sha256 2>/dev/null | head -1)

    echo "# lumi-jax image ABI probe v$PROBE_VERSION"
    echo "# generated by tools/probe_image.sh; diff this against tools/baselines/"
    printf '\n[image]\n'
    printf '%-34s %s\n' image.name       "$name"
    # -L: images are reached via a -latest symlink; bare stat reports the link's size.
    printf '%-34s %s\n' image.size_bytes "$(stat -Lc %s "$SIF" 2>/dev/null || echo unknown)"
    printf '%-34s %s\n' image.sha256     "${sha:-not-recorded}"

    singularity exec --no-home "$SIF" bash -s <<<"$PROBE_BODY" || status=1
    echo
done
exit $status
