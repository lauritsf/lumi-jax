#!/bin/bash
# Verify native ROCm 7.2 libraries through their loader paths.
set -euo pipefail
for name in librccl libhipblaslt; do
    path="$(readlink -f "/opt/rocm/lib/$name.so.1")"
    case "$path" in
        /opt/rocm-7.2.4/lib/*.70204) ;;
        *) echo "FATAL: $name is not native ROCm 7.2.4: $path"; exit 1 ;;
    esac
    if readelf -d "$path" | grep -E 'NEEDED.*lumi_.*stub'; then
        echo "FATAL: $path depends on a compatibility stub"; exit 1
    fi
done
python3 - <<'CHECK'
import ctypes
for name in ("libdw.so.1", "librocm_smi64.so.1"):
    ctypes.CDLL(name)
    print("native loader OK:", name)
rccl = ctypes.CDLL("librccl.so.1")
for name in ("ncclCommWindowRegister", "ncclCommWindowDeregister"):
    getattr(rccl, name)
    print("native RCCL OK:", name)
hb = ctypes.CDLL("libhipblaslt.so.1")
getattr(hb, "_ZN13hipblaslt_ext12GemmInstance20setMaxWorkspaceBytesEm")
print("native hipBLASLt workspace API OK")
CHECK
