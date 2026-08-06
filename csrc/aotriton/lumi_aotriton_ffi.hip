// AOTriton forward FFI (inference/prefill path). AOTriton is faster than CK on the
// forward on gfx90a, but its backward is weak there and is not bound.
// Forward only; bind CK for training. Calls aotriton::v3::flash::attn_fwd.
//
// AOTriton uses (B, H, S, D) layout; JAX passes (B, S, H, D) -- handled via strides
// (no transpose). Links libaotriton_v2.so: the standalone install where the image has
// one, torch's bundled copy otherwise.

#include <array>
#include <cstdint>

#include "aotriton/dtypes.h"
#include "aotriton/util.h"
#include "aotriton/runtime.h"
#include "aotriton/flash.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace aot = aotriton;
namespace flash = aotriton::v3::flash;

static ffi::Error AotFwdImpl(hipStream_t stream, ffi::ScratchAllocator scratch,
                             ffi::AnyBuffer q, ffi::AnyBuffer k, ffi::AnyBuffer v,
                             double scale, int32_t is_bf16, int32_t is_causal,
                             int32_t window_left, int32_t window_right,
                             ffi::Result<ffi::AnyBuffer> out,
                             ffi::Result<ffi::AnyBuffer> lse) {
  auto d = q.dimensions();
  if (d.size() != 4)
    return ffi::Error::InvalidArgument("q must be rank 4 (B,S,H,D)");
  const uint64_t B = d[0], S = d[1], H = d[2], Dh = d[3];
  const uint64_t Hk = k.dimensions()[2];
  const aot::DType dt = is_bf16 ? aot::DType::kBFloat16 : aot::DType::kFloat16;

  // (B,S,H,D) contiguous data presented as logical (B, h, S, D)
  auto t4 = [&](void* p, uint64_t h) -> flash::T4 {
    return flash::T4(reinterpret_cast<intptr_t>(p),
                     std::array<uint64_t, 4>{B, h, S, Dh},
                     std::array<uint64_t, 4>{S * h * Dh, Dh, h * Dh, 1ULL}, dt);
  };

  flash::attn_fwd_params p;  // default-ctor sets null tensors / defaults
  p.Q = t4(q.untyped_data(), H);
  p.K = t4(k.untyped_data(), Hk);
  p.V = t4(v.untyped_data(), Hk);
  p.Out = t4(out->untyped_data(), H);
  p.Sm_scale = static_cast<float>(scale);
  // softmax_lse (B*H, S) f32
  p.L = flash::T2(reinterpret_cast<intptr_t>(lse->untyped_data()),
                  std::array<uint64_t, 2>{B * H, S},
                  std::array<uint64_t, 2>{S, 1ULL}, aot::DType::kFloat32);
  p.dropout_p = 0.f;
  p.varlen_type = flash::VarlenType::None;
  // The persistent kernel variant load-balances via a global atomic counter; it must
  // be a real, zeroed int32 in device memory (a null tensor faults).
  auto ctr = scratch.Allocate(sizeof(int32_t), 16);
  if (!ctr) return ffi::Error::Internal("scratch alloc (counter) failed");
  if (hipMemsetAsync(*ctr, 0, sizeof(int32_t), stream) != hipSuccess)
    return ffi::Error::Internal("hipMemsetAsync(counter) failed");
  p.persistent_atomic_counter = flash::T0(reinterpret_cast<intptr_t>(*ctr),
                                          aot::DType::kInt32);
  // AOTriton expresses causal/local via "WindowedAttention" (top-left/bottom-right
  // causal types are not compiled into the binary kernels). Causal = the magic
  // BottomRightAligned window value (matches flash_attn's alignment for S_q != S_k);
  // explicit local window = the (left,right) counts; otherwise full attention.
  if (is_causal) {
    p.causal_type = flash::CausalType::WindowedAttention;
    p.window_left = flash::WindowValue::BottomRightAligned;
    p.window_right = 0;
  } else if (window_left >= 0 || window_right >= 0) {
    p.causal_type = flash::CausalType::WindowedAttention;
    p.window_left = window_left;
    p.window_right = window_right;
  } else {
    p.causal_type = flash::CausalType::None;
    p.window_left = 0;
    p.window_right = 0;
  }

  flash::attn_options opts;
  aot::Stream s(stream);
  hipError_t err = flash::attn_fwd(p, flash::attn_fwd_params::kVersion, s, &opts);
  if (err != hipSuccess)
    return ffi::Error::Internal("aotriton attn_fwd returned a non-success hipError_t");
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LumiAotFwd, AotFwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::AnyBuffer>()  // q (B,S,H,D)
        .Arg<ffi::AnyBuffer>()  // k (B,S,Hk,D)
        .Arg<ffi::AnyBuffer>()  // v
        .Attr<double>("scale")
        .Attr<int32_t>("is_bf16")
        .Attr<int32_t>("is_causal")
        .Attr<int32_t>("window_left")
        .Attr<int32_t>("window_right")
        .Ret<ffi::AnyBuffer>()   // out (B,S,H,D)
        .Ret<ffi::AnyBuffer>()); // lse (B*H, S)
