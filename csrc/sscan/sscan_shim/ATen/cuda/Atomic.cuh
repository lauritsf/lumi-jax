// Shim for ATen/cuda/Atomic.cuh (the real one includes <cuda.h>). The bwd kernel
// only does gpuAtomicAdd on float (real-weight build).
#pragma once
#include <hip/hip_runtime.h>

template <typename T>
__device__ inline T gpuAtomicAdd(T *addr, T val) { return atomicAdd(addr, val); }
