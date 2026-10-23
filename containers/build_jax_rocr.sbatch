#!/bin/bash
#SBATCH --job-name=jax-rocr-build
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=build-rocr-%j.out
# The overlay replaces the ROCm 7.2.4 runtime and keeps its register/thunk ABI.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source tools/image_name.sh
BASE=${BASE:-$(jax_image_resolve lean 0.11.2)}
RUN=${RUN:?qualification directory}
OUT=${OUT:?immutable output path}
test ! -e "$OUT"
singularity exec "$BASE" bash -c '
    test "$(dpkg-query -W -f='"'"'${Version}'"'"' hsa-rocr)" = "1.18.0.70204-93~24.04"
'

mkdir -p "$RUN/build-input/tools" "$RUN/payload"
curl -fLsS https://github.com/ROCm/ROCR-Runtime/archive/refs/tags/rocm-7.2.4.tar.gz -o "$RUN/build-input/source.tar.gz"
echo "60532cd86edce5a603aa18df406ec1f5a3d18f0663d79b3b7822ff721c5a04ec  $RUN/build-input/source.tar.gz" | sha256sum -c -
rm -rf "$RUN/build-input/ROCR-Runtime-rocm-7.2.4"
tar xf "$RUN/build-input/source.tar.gz" -C "$RUN/build-input"
SRC="$RUN/build-input/ROCR-Runtime-rocm-7.2.4"
sha256sum -c <<'HASHES'
50171dfc74cc68f77d384bab724adbfe961c605fef3c70fbeb793aa989511ab8  containers/patches/rocr-4066.patch
9b6b902dc209291b9397a950b909aceae8dcad76661e2ffb7df706c816d7f2c5  containers/patches/rocr-7874.patch
HASHES
patch --batch --fuzz=0 -d "$SRC" -p3 < containers/patches/rocr-4066.patch
patch --batch --fuzz=0 -d "$SRC" -p3 < containers/patches/rocr-7874.patch
cp /usr/bin/xxd "$RUN/build-input/tools/"
singularity exec -B "$RUN:$RUN" "$BASE" env PATH="$RUN/build-input/tools:/opt/rocm/bin:/opt/venv/bin:/usr/bin:/bin" \
    bash "$PWD/tools/build_rocr.sh" "$SRC" "$RUN/rocr-build"
cp "$RUN/rocr-build/rocr/lib/libhsa-runtime64.so.1.18.0" "$RUN/payload/"
python3 - "$BASE" "$RUN" <<'PY'
import hashlib,json,pathlib,sys
base,run=sys.argv[1:]; p=pathlib.Path(run)
def sha(f):
 h=hashlib.sha256()
 with open(str(f),'rb') as s:
  for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
d={'base':base,'base_sha256':sha(base),'source':'ROCR-Runtime rocm-7.2.4','source_sha256':sha(p/'build-input/source.tar.gz'),'patches':{str(f):sha(f) for f in pathlib.Path('containers/patches').glob('rocr-*.patch')},'runtime_sha256':sha(p/'payload/libhsa-runtime64.so.1.18.0')}
(p/'payload/manifest.json').write_text(json.dumps(d,indent=2)+'\n')
PY
export PATH=/appl/lumi/SW/CrayEnv/EB/systools/25.03/bin:$PATH
export SINGULARITY_CACHEDIR="$LUMI_JAX_WORK/sing-cache"
export SINGULARITY_TMPDIR="/tmp/lumi-jax-rocr-$SLURM_JOB_ID"
mkdir -p "$SINGULARITY_TMPDIR"
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT
singularity build --notest --build-arg BASE="$BASE" --build-arg PAYLOAD="$RUN/payload" "$OUT" containers/jax-rocr.def
_jax_sif_align_partition "$OUT"
singularity exec "$OUT" cat /opt/lumi-jax-rocr/manifest.json
sha256sum "$OUT" | tee "$RUN/image.sha256"
printf '%s\n' "$OUT" > "$RUN/image.path"
echo 'ROCR IMAGE BUILD OK'
