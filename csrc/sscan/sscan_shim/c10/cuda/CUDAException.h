// Shim for c10/cuda/CUDAException.h: the real header expands C10_CUDA_CHECK into a
// call into libc10 (c10_cuda_check_implementation), pulling a torch link dependency.
// We only need the two macros, so keep the build torch-free.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#define C10_CUDA_CHECK(expr)                                                  \
  do {                                                                        \
    hipError_t e_ = (expr);                                                   \
    if (e_ != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),       \
              __FILE__, __LINE__);                                            \
      abort();                                                                \
    }                                                                         \
  } while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(hipGetLastError())
