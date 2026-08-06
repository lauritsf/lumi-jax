#!/bin/bash
# Make Ray usable in a LUMI GPU job. Source this before `import ray`:
#
#   source tools/ray_visible_devices.sh
#
# Ray refuses to start on AMD when ROCR_VISIBLE_DEVICES is set and HIP_VISIBLE_DEVICES
# is not, and LUMI's Slurm sets only the former, so `import ray` raises in any GPU job.
# The check fires only once Ray has DETECTED AMD GPUs -- so a login-node `import ray`
# succeeding proves nothing.
#
# Do NOT just copy the value across: the two masks COMPOSE. ROCR selects which GCDs
# exist for the process, then HIP indexes into that already-masked set -- ROCR=6,7
# means 2 GCDs addressed 0,1, so HIP must be 0,1, not 6,7. Copying would hide every GPU
# under exactly the per-GCD pinning multi-node runs use; the remap is 0..N-1, where N
# is how many entries ROCR lists.
#
# Deliberately NOT baked into the image's %environment: an env default set inside the
# image is invisible to whoever debugs it, and this failure names its own fix already.

if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
    _lumi_ray_n=$(awk -F, '{print NF}' <<<"$ROCR_VISIBLE_DEVICES")
    HIP_VISIBLE_DEVICES=$(seq -s, 0 $((_lumi_ray_n - 1)))
    export HIP_VISIBLE_DEVICES
    echo "ray: ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES -> HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES" >&2
    unset _lumi_ray_n
fi
