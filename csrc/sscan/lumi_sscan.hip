// Mamba selective_scan fwd+bwd as XLA FFI custom calls (gfx90a) -- self-compiled
// from state-spaces/mamba @ v2.3.2 csrc (real weights only), NO torch link: c10
// half/bf16 types are header-only and C10_CUDA_CHECK is shimmed (sscan_shim/).
//
// Layouts (contiguous, strides in elements -- mirrors selective_scan.cpp):
//   u/delta/z/out/out_z (B, dim, L) input dtype (fp32/fp16/bf16)
//   A (dim, N) f32, N <= 256; variable B/C (B, G, N, L) input dtype
//   D/delta_bias (dim,) f32; x residual (B, dim, n_chunks, 2N) f32, n_chunks=ceil(L/2048)
// Backward accumulates dA/dB/dC/dD/ddelta_bias atomically in f32 -> zero them first
// (hipMemsetAsync, same pattern as conv1d). dB/dC come back f32; wrapper casts.

#include <cstdint>

#include <hip/hip_runtime.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include "selective_scan.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

template <typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, hipStream_t stream);
template <typename input_t, typename weight_t>
void selective_scan_bwd_cuda(SSMParamsBwd &params, hipStream_t stream);

static constexpr int kChunk = 2048;

static void set_common(SSMParamsBase &p, int64_t B, int64_t dim, int64_t L, int64_t N,
                       int64_t G, bool has_z, bool softplus,
                       void *u, void *delta, void *A, void *Bv, void *Cv,
                       void *D, void *db, void *z, void *out, void *out_z, void *x) {
  p.batch = (int)B; p.dim = (int)dim; p.seqlen = (int)L; p.dstate = (int)N;
  p.n_groups = (int)G; p.n_chunks = (int)((L + kChunk - 1) / kChunk);
  p.dim_ngroups_ratio = (int)(dim / G);
  p.is_variable_B = true; p.is_variable_C = true;
  p.delta_softplus = softplus;
  p.u_ptr = u; p.delta_ptr = delta; p.A_ptr = A; p.B_ptr = Bv; p.C_ptr = Cv;
  p.D_ptr = D; p.delta_bias_ptr = db; p.out_ptr = out; p.x_ptr = x;
  p.z_ptr = has_z ? z : nullptr;
  p.out_z_ptr = has_z ? out_z : nullptr;
  p.A_d_stride = (uint32_t)N; p.A_dstate_stride = 1;
  p.B_batch_stride = (uint32_t)(G * N * L); p.B_group_stride = (uint32_t)(N * L);
  p.B_dstate_stride = (uint32_t)L;
  p.C_batch_stride = (uint32_t)(G * N * L); p.C_group_stride = (uint32_t)(N * L);
  p.C_dstate_stride = (uint32_t)L;
  p.u_batch_stride = (uint32_t)(dim * L); p.u_d_stride = (uint32_t)L;
  p.delta_batch_stride = (uint32_t)(dim * L); p.delta_d_stride = (uint32_t)L;
  p.z_batch_stride = (uint32_t)(dim * L); p.z_d_stride = (uint32_t)L;
  p.out_batch_stride = (uint32_t)(dim * L); p.out_d_stride = (uint32_t)L;
  p.out_z_batch_stride = (uint32_t)(dim * L); p.out_z_d_stride = (uint32_t)L;
}

template <typename F>
static ffi::Error dispatch_input(ffi::DataType dt, F &&f) {
  switch (dt) {
    case ffi::DataType::F32:  f.template operator()<float>(); return ffi::Error::Success();
    case ffi::DataType::F16:  f.template operator()<at::Half>(); return ffi::Error::Success();
    case ffi::DataType::BF16: f.template operator()<at::BFloat16>(); return ffi::Error::Success();
    default: return ffi::Error::InvalidArgument("selective_scan: dtype must be f32/f16/bf16");
  }
}

static ffi::Error FwdImpl(hipStream_t stream,
                          ffi::AnyBuffer u, ffi::AnyBuffer delta, ffi::AnyBuffer A,
                          ffi::AnyBuffer Bv, ffi::AnyBuffer Cv,
                          ffi::AnyBuffer D, ffi::AnyBuffer db, ffi::AnyBuffer z,
                          int32_t has_d, int32_t has_db, int32_t has_z, int32_t softplus,
                          ffi::Result<ffi::AnyBuffer> out,
                          ffi::Result<ffi::AnyBuffer> x,
                          ffi::Result<ffi::AnyBuffer> out_z) {
  auto du = u.dimensions();
  if (du.size() != 3 || Bv.dimensions().size() != 4)
    return ffi::Error::InvalidArgument("u must be (B,dim,L); B/C must be (B,G,N,L)");
  const int64_t B = du[0], dim = du[1], L = du[2];
  const int64_t G = Bv.dimensions()[1], N = Bv.dimensions()[2];
  if (N > 256) return ffi::Error::InvalidArgument("dstate must be <= 256");
  if (dim % G) return ffi::Error::InvalidArgument("dim must be a multiple of n_groups");

  SSMParamsBase p;
  memset(&p, 0, sizeof(p));
  set_common(p, B, dim, L, N, G, has_z, softplus,
             u.untyped_data(), delta.untyped_data(), A.untyped_data(),
             Bv.untyped_data(), Cv.untyped_data(),
             has_d ? D.untyped_data() : nullptr, has_db ? db.untyped_data() : nullptr,
             z.untyped_data(), out->untyped_data(), out_z->untyped_data(),
             x->untyped_data());
  return dispatch_input(u.element_type(), [&]<typename T>() {
    selective_scan_fwd_cuda<T, float>(p, stream);
  });
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LumiSscanFwd, FwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()  // u (B,dim,L)
        .Arg<ffi::AnyBuffer>()  // delta (B,dim,L)
        .Arg<ffi::AnyBuffer>()  // A (dim,N) f32
        .Arg<ffi::AnyBuffer>()  // B (B,G,N,L)
        .Arg<ffi::AnyBuffer>()  // C (B,G,N,L)
        .Arg<ffi::AnyBuffer>()  // D (dim,) f32 | dummy
        .Arg<ffi::AnyBuffer>()  // delta_bias (dim,) f32 | dummy
        .Arg<ffi::AnyBuffer>()  // z (B,dim,L) | dummy
        .Attr<int32_t>("has_d")
        .Attr<int32_t>("has_db")
        .Attr<int32_t>("has_z")
        .Attr<int32_t>("softplus")
        .Ret<ffi::AnyBuffer>()   // out (B,dim,L)
        .Ret<ffi::AnyBuffer>()   // x (B,dim,n_chunks,2N) f32
        .Ret<ffi::AnyBuffer>()); // out_z (B,dim,L) (= silu(z)*out; garbage if !has_z)

static ffi::Error BwdImpl(hipStream_t stream,
                          ffi::AnyBuffer u, ffi::AnyBuffer delta, ffi::AnyBuffer A,
                          ffi::AnyBuffer Bv, ffi::AnyBuffer Cv,
                          ffi::AnyBuffer D, ffi::AnyBuffer db, ffi::AnyBuffer z,
                          ffi::AnyBuffer out, ffi::AnyBuffer x, ffi::AnyBuffer dout,
                          int32_t has_d, int32_t has_db, int32_t has_z, int32_t softplus,
                          ffi::Result<ffi::AnyBuffer> du, ffi::Result<ffi::AnyBuffer> ddelta,
                          ffi::Result<ffi::AnyBuffer> dA, ffi::Result<ffi::AnyBuffer> dB,
                          ffi::Result<ffi::AnyBuffer> dC, ffi::Result<ffi::AnyBuffer> dD,
                          ffi::Result<ffi::AnyBuffer> ddb, ffi::Result<ffi::AnyBuffer> dz) {
  auto dd = u.dimensions();
  const int64_t B = dd[0], dim = dd[1], L = dd[2];
  const int64_t G = Bv.dimensions()[1], N = Bv.dimensions()[2];

  // dA/dB/dC/dD/ddelta_bias are atomic f32 accumulators -> zero before launch.
  if (hipMemsetAsync(dA->untyped_data(), 0, dim * N * sizeof(float), stream) != hipSuccess ||
      hipMemsetAsync(dB->untyped_data(), 0, B * G * N * L * sizeof(float), stream) != hipSuccess ||
      hipMemsetAsync(dC->untyped_data(), 0, B * G * N * L * sizeof(float), stream) != hipSuccess ||
      hipMemsetAsync(dD->untyped_data(), 0, dim * sizeof(float), stream) != hipSuccess ||
      hipMemsetAsync(ddb->untyped_data(), 0, dim * sizeof(float), stream) != hipSuccess)
    return ffi::Error::Internal("hipMemsetAsync(grad accumulators) failed");

  SSMParamsBwd p;
  memset(&p, 0, sizeof(p));
  // !has_z: out is unused by the kernel; upstream passes dout in its place.
  set_common(p, B, dim, L, N, G, has_z, softplus,
             u.untyped_data(), delta.untyped_data(), A.untyped_data(),
             Bv.untyped_data(), Cv.untyped_data(),
             has_d ? D.untyped_data() : nullptr, has_db ? db.untyped_data() : nullptr,
             z.untyped_data(),
             has_z ? out.untyped_data() : dout.untyped_data(), nullptr,  // no recompute_out_z
             x.untyped_data());
  p.dout_ptr = dout.untyped_data();
  p.du_ptr = du->untyped_data();
  p.ddelta_ptr = ddelta->untyped_data();
  p.dA_ptr = dA->untyped_data();
  p.dB_ptr = dB->untyped_data();
  p.dC_ptr = dC->untyped_data();
  p.dD_ptr = has_d ? dD->untyped_data() : nullptr;
  p.ddelta_bias_ptr = has_db ? ddb->untyped_data() : nullptr;
  p.dz_ptr = has_z ? dz->untyped_data() : nullptr;
  p.dout_batch_stride = (uint32_t)(dim * L); p.dout_d_stride = (uint32_t)L;
  p.du_batch_stride = (uint32_t)(dim * L); p.du_d_stride = (uint32_t)L;
  p.ddelta_batch_stride = (uint32_t)(dim * L); p.ddelta_d_stride = (uint32_t)L;
  p.dz_batch_stride = (uint32_t)(dim * L); p.dz_d_stride = (uint32_t)L;
  p.dA_d_stride = (uint32_t)N; p.dA_dstate_stride = 1;
  p.dB_batch_stride = (uint32_t)(G * N * L); p.dB_group_stride = (uint32_t)(N * L);
  p.dB_dstate_stride = (uint32_t)L;
  p.dC_batch_stride = (uint32_t)(G * N * L); p.dC_group_stride = (uint32_t)(N * L);
  p.dC_dstate_stride = (uint32_t)L;
  return dispatch_input(u.element_type(), [&]<typename T>() {
    selective_scan_bwd_cuda<T, float>(p, stream);
  });
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LumiSscanBwd, BwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()  // u
        .Arg<ffi::AnyBuffer>()  // delta
        .Arg<ffi::AnyBuffer>()  // A (dim,N) f32
        .Arg<ffi::AnyBuffer>()  // B (B,G,N,L)
        .Arg<ffi::AnyBuffer>()  // C (B,G,N,L)
        .Arg<ffi::AnyBuffer>()  // D | dummy
        .Arg<ffi::AnyBuffer>()  // delta_bias | dummy
        .Arg<ffi::AnyBuffer>()  // z | dummy
        .Arg<ffi::AnyBuffer>()  // out (fwd, pre-z) | dummy
        .Arg<ffi::AnyBuffer>()  // x residual (B,dim,n_chunks,2N) f32
        .Arg<ffi::AnyBuffer>()  // dout
        .Attr<int32_t>("has_d")
        .Attr<int32_t>("has_db")
        .Attr<int32_t>("has_z")
        .Attr<int32_t>("softplus")
        .Ret<ffi::AnyBuffer>()   // du (B,dim,L)
        .Ret<ffi::AnyBuffer>()   // ddelta (B,dim,L)
        .Ret<ffi::AnyBuffer>()   // dA (dim,N) f32
        .Ret<ffi::AnyBuffer>()   // dB (B,G,N,L) f32
        .Ret<ffi::AnyBuffer>()   // dC (B,G,N,L) f32
        .Ret<ffi::AnyBuffer>()   // dD (dim,) f32
        .Ret<ffi::AnyBuffer>()   // ddelta_bias (dim,) f32
        .Ret<ffi::AnyBuffer>()); // dz (B,dim,L)
