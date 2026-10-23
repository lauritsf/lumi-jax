#!/bin/bash
set -euo pipefail
RUN=${1:?qualification directory}
SRC="$RUN/build-input/ROCR-Runtime-rocm-7.2.4"
curl -fLsS https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz -o "$RUN/gtest.tar.gz"
echo "7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926  $RUN/gtest.tar.gz" | sha256sum -c -
tar xf "$RUN/gtest.tar.gz" -C "$RUN"
GTEST="$RUN/googletest-1.15.2/googletest"
g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -pthread \
    -I"$SRC/runtime/hsa-runtime" -I"$GTEST/include" -I"$GTEST" \
    "$SRC/rocrtst/suites/functional/intercept_queue/intercept_queue_logic_test.cc" \
    "$GTEST/src/gtest-all.cc" "$GTEST/src/gtest_main.cc" -o "$RUN/intercept-queue-test"
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 "$RUN/intercept-queue-test" | tee "$RUN/queue-logic.log"
grep -qE '\[ +PASSED +\] [1-9][0-9]* tests?' "$RUN/queue-logic.log"
