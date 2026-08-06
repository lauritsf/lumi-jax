// Shim: mamba includes <cuda_fp16.h> unconditionally (torch hipify normally maps it).
#pragma once
#include <hip/hip_fp16.h>
