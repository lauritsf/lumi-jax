// XLA FFI binding for the container's causal_conv1d (Dao-AILab, Mamba/SSM) gfx90a
// kernel: depthwise causal 1D conv (+ optional SiLU). Forward.
//
// Calls the raw template dispatcher causal_conv1d_fwd_cuda<in_t,w_t>(ConvParamsBase&,
// stream) from causal_conv1d_cuda.so (all 9 dtype combos of {fp32,fp16,bf16} are
// instantiated). Built against the causal-conv1d @ 0d2252d header (ConvParamsBase)
// and torch's c10 half/bf16 headers (header-only). No torch in the JAX graph.
//
// Layout: x (B, dim, L) contiguous; weight (dim, width); bias (dim,); out (B, dim, L).

#include <cstdint>
#include <hip/hip_runtime.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include "causal_conv1d.h"  // ConvParamsBase, ConvParamsBwd
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// raw kernel (defined in causal_conv1d_cuda.so)
template <typename in_t, typename w_t>
void causal_conv1d_fwd_cuda(ConvParamsBase& params, hipStream_t stream);

template <typename in_t, typename w_t>
static void launch(ConvParamsBase& p, hipStream_t s) { causal_conv1d_fwd_cuda<in_t, w_t>(p, s); }

static ffi::Error dispatch(ffi::DataType xt, ffi::DataType wt, ConvParamsBase& p, hipStream_t s) {
  using ffi::DataType;
  auto F32 = DataType::F32, F16 = DataType::F16, BF16 = DataType::BF16;
#define CC(XT, WT, XTYPE, WTYPE) \
  if (xt == XT && wt == WT) { launch<XTYPE, WTYPE>(p, s); return ffi::Error::Success(); }
  CC(F32, F32, float, float) CC(F32, F16, float, c10::Half) CC(F32, BF16, float, c10::BFloat16)
  CC(F16, F32, c10::Half, float) CC(F16, F16, c10::Half, c10::Half) CC(F16, BF16, c10::Half, c10::BFloat16)
  CC(BF16, F32, c10::BFloat16, float) CC(BF16, F16, c10::BFloat16, c10::Half) CC(BF16, BF16, c10::BFloat16, c10::BFloat16)
#undef CC
  return ffi::Error::InvalidArgument("causal_conv1d: unsupported (x,weight) dtype combo");
}

static ffi::Error FwdImpl(hipStream_t stream,
                          ffi::AnyBuffer x, ffi::AnyBuffer weight, ffi::AnyBuffer bias,
                          int32_t has_bias, int32_t silu,
                          ffi::Result<ffi::AnyBuffer> out) {
  auto dx = x.dimensions();
  if (dx.size() != 3) return ffi::Error::InvalidArgument("x must be (B, dim, L)");
  const int B = dx[0], dim = dx[1], L = dx[2];
  const int width = weight.dimensions()[1];

  ConvParamsBase p{};  // zero-init (matches set_conv_params_fwd's memset)
  p.batch = B; p.dim = dim; p.seqlen = L; p.width = width;
  p.silu_activation = silu != 0;
  p.x_ptr = x.untyped_data();
  p.weight_ptr = weight.untyped_data();
  p.bias_ptr = has_bias ? bias.untyped_data() : nullptr;
  p.out_ptr = out->untyped_data();
  p.x_batch_stride = (uint32_t)(dim * L); p.x_c_stride = (uint32_t)L; p.x_l_stride = 1;
  p.weight_c_stride = (uint32_t)width; p.weight_width_stride = 1;
  p.out_batch_stride = (uint32_t)(dim * L); p.out_c_stride = (uint32_t)L; p.out_l_stride = 1;

  return dispatch(x.element_type(), weight.element_type(), p, stream);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LumiConv1dFwd, FwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()  // x (B,dim,L)
        .Arg<ffi::AnyBuffer>()  // weight (dim,width)
        .Arg<ffi::AnyBuffer>()  // bias (dim,) or dummy
        .Attr<int32_t>("has_bias")
        .Attr<int32_t>("silu")
        .Ret<ffi::AnyBuffer>()); // out (B,dim,L)

// ---------------------------------------------------------------------------
// Backward. Default (non-deterministic, non-channel-last) path -> NO workspace:
// the kernel atomic-adds into a zeroed dweight/dbias and writes dx directly.
// (The chunked dweight_workspace is only for deterministic / channel-last.)
// ---------------------------------------------------------------------------
template <typename in_t, typename w_t>
void causal_conv1d_bwd_cuda(ConvParamsBwd& params, hipStream_t stream);

static ffi::Error bdispatch(ffi::DataType xt, ffi::DataType wt, ConvParamsBwd& p, hipStream_t s) {
  using ffi::DataType;
  auto F32 = DataType::F32, F16 = DataType::F16, BF16 = DataType::BF16;
#define BB(XT, WT, XTYPE, WTYPE) \
  if (xt == XT && wt == WT) { causal_conv1d_bwd_cuda<XTYPE, WTYPE>(p, s); return ffi::Error::Success(); }
  BB(F32, F32, float, float) BB(F32, F16, float, c10::Half) BB(F32, BF16, float, c10::BFloat16)
  BB(F16, F32, c10::Half, float) BB(F16, F16, c10::Half, c10::Half) BB(F16, BF16, c10::Half, c10::BFloat16)
  BB(BF16, F32, c10::BFloat16, float) BB(BF16, F16, c10::BFloat16, c10::Half) BB(BF16, BF16, c10::BFloat16, c10::BFloat16)
#undef BB
  return ffi::Error::InvalidArgument("causal_conv1d_bwd: unsupported (x,weight) dtype combo");
}

static ffi::Error BwdImpl(hipStream_t stream,
                          ffi::AnyBuffer dout, ffi::AnyBuffer x, ffi::AnyBuffer weight,
                          ffi::AnyBuffer bias, int32_t has_bias, int32_t silu,
                          ffi::Result<ffi::AnyBuffer> dx,
                          ffi::Result<ffi::AnyBuffer> dweight,
                          ffi::Result<ffi::AnyBuffer> dbias) {
  auto dxs = x.dimensions();
  if (dxs.size() != 3) return ffi::Error::InvalidArgument("x must be (B, dim, L)");
  const int B = dxs[0], dim = dxs[1], L = dxs[2];
  const int width = weight.dimensions()[1];

  // dweight/dbias are FP32 accumulators (the kernel always reads dweight_ptr/dbias_ptr
  // as float*, regardless of weight dtype -- caller casts back). They accumulate via
  // atomics -> must start zeroed. dx (x dtype) is written fully, no zeroing needed.
  if (hipMemsetAsync(dweight->untyped_data(), 0, (size_t)dim * width * sizeof(float),
                     stream) != hipSuccess)
    return ffi::Error::Internal("memset dweight failed");
  if (has_bias && hipMemsetAsync(dbias->untyped_data(), 0, (size_t)dim * sizeof(float),
                     stream) != hipSuccess)
    return ffi::Error::Internal("memset dbias failed");

  ConvParamsBwd p{};
  p.batch = B; p.dim = dim; p.seqlen = L; p.width = width;
  p.silu_activation = silu != 0;
  p.x_ptr = x.untyped_data();
  p.weight_ptr = weight.untyped_data();
  p.bias_ptr = has_bias ? bias.untyped_data() : nullptr;
  p.x_batch_stride = (uint32_t)(dim * L); p.x_c_stride = (uint32_t)L; p.x_l_stride = 1;
  p.weight_c_stride = (uint32_t)width; p.weight_width_stride = 1;
  p.dout_ptr = dout.untyped_data();
  p.dx_ptr = dx->untyped_data();
  p.dweight_ptr = dweight->untyped_data();
  p.dbias_ptr = has_bias ? dbias->untyped_data() : nullptr;
  p.dout_batch_stride = (uint32_t)(dim * L); p.dout_c_stride = (uint32_t)L; p.dout_l_stride = 1;
  p.dweight_c_stride = (uint32_t)width; p.dweight_width_stride = 1;
  p.dx_batch_stride = (uint32_t)(dim * L); p.dx_c_stride = (uint32_t)L; p.dx_l_stride = 1;
  p.deterministic = false;  // default path: no workspace

  return bdispatch(x.element_type(), weight.element_type(), p, stream);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LumiConv1dBwd, BwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()  // dout (B,dim,L)
        .Arg<ffi::AnyBuffer>()  // x
        .Arg<ffi::AnyBuffer>()  // weight (dim,width)
        .Arg<ffi::AnyBuffer>()  // bias or dummy
        .Attr<int32_t>("has_bias")
        .Attr<int32_t>("silu")
        .Ret<ffi::AnyBuffer>()   // dx (B,dim,L)
        .Ret<ffi::AnyBuffer>()   // dweight (dim,width)
        .Ret<ffi::AnyBuffer>()); // dbias (dim,)
