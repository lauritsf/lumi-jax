// Force-included (-include) CUDA->HIP aliases mamba normally gets from torch hipify.
#pragma once
#include <hip/hip_runtime.h>

#define cudaStream_t hipStream_t
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
