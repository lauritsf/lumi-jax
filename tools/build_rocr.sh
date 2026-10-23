#!/bin/bash
# Build against the image's ROCm headers, register library, and tagged static thunk.
set -euo pipefail
SRC=${1:?source directory}
OUT=${2:?build directory}
mkdir -p "$OUT/cmake-tools"
cat > "$OUT/cmake-tools/ClangConfig.cmake" <<'CMAKE'
if(NOT TARGET clang)
add_executable(clang IMPORTED GLOBAL)
set_target_properties(clang PROPERTIES IMPORTED_LOCATION /opt/rocm/llvm/bin/clang)
endif()
CMAKE
cat > "$OUT/cmake-tools/LLVMConfig.cmake" <<'CMAKE'
if(NOT TARGET llvm-objcopy)
add_executable(llvm-objcopy IMPORTED GLOBAL)
set_target_properties(llvm-objcopy PROPERTIES IMPORTED_LOCATION /opt/rocm/llvm/bin/llvm-objcopy)
endif()
CMAKE
export CC=/usr/bin/gcc CXX=/usr/bin/g++
cmake -S "$SRC" -B "$OUT" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_PREFIX_PATH=/opt/rocm -DClang_DIR="$OUT/cmake-tools" \
    -DLLVM_DIR="$OUT/cmake-tools"
cmake --build "$OUT" --parallel "${BUILD_JOBS:-8}"
