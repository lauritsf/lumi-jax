// Explicit real-weight instantiations of mamba's selective_scan kernels (gfx90a).
// Complex (weight_t = c10::complex<float>) is deliberately NOT instantiated -- Mamba1
// uses real A and the FFI handler dispatches input dtype only.
#include "selective_scan_fwd_kernel.cuh"
#include "selective_scan_bwd_kernel.cuh"

template void selective_scan_fwd_cuda<float, float>(SSMParamsBase &params, hipStream_t stream);
template void selective_scan_fwd_cuda<at::Half, float>(SSMParamsBase &params, hipStream_t stream);
template void selective_scan_fwd_cuda<at::BFloat16, float>(SSMParamsBase &params, hipStream_t stream);

template void selective_scan_bwd_cuda<float, float>(SSMParamsBwd &params, hipStream_t stream);
template void selective_scan_bwd_cuda<at::Half, float>(SSMParamsBwd &params, hipStream_t stream);
template void selective_scan_bwd_cuda<at::BFloat16, float>(SSMParamsBwd &params, hipStream_t stream);
