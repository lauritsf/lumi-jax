// libtorch-in-FFI proof: call a vLLM torch-ABI op from a JITTABLE XLA custom call,
// with NO dlpack bridge (no clones, no host sync) and torch OUT of the JAX graph.
//
// vLLM ships only torch-ABI ops (`silu_and_mul(at::Tensor&, at::Tensor&)`, etc.) -- the
// raw-pointer launchers are internal/unexported and the wheel has no csrc, so a fully
// torch-free FFI (like CK/AOTriton/causal_conv1d) isn't reachable. Instead we wrap the
// XLA device buffers as at::Tensor via at::from_blob (zero-copy), run the op on the FFI's
// hipStream, and return. libtorch is linked into THIS .so but invisible to JAX; the call
// is jittable. This is the general mechanism for paged_attention_v1 / fused MoE too.
//
// PoC op: silu_and_mul(out, input): out[..., :d] = silu(in[..., :d]) * in[..., d:].

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <hip/hip_runtime.h>

#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPGuard.h>

#if LUMI_VLLM_STABLE_ABI
// vLLM v0.22.1+ moved the activation ops to torch's stable ABI, in a second extension
// module. See the shim below.
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#endif

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// silu_and_mul moved between vLLM v0.20.1 and v0.22.1: different module AND different
// signature. Both halves of the link move at once.
//
//   v0.20.1  vllm/_C.abi3.so                 void silu_and_mul(at::Tensor&, at::Tensor&)
//            _Z12silu_and_mulRN2at6TensorES1_
//   v0.22.1  vllm/_C_stable_libtorch.abi3.so void silu_and_mul(torch::stable::Tensor&,
//                                                              torch::stable::Tensor&)
//            _Z12silu_and_mulRN5torch6stable6TensorES2_
//
// build_vllm_ffi.sh probes which module is present and defines LUMI_VLLM_STABLE_ABI, so
// one source serves both bases. paged_attention_v1 is NOT affected -- still in _C, still
// at::Tensor&, signature byte-identical across the two tags -- so it is left alone below.
//
// STATUS OF THE STABLE PATH: the declaration's mangled name is VERIFIED (compiled against
// torch 2.10's real headers, and it matches the name 5.2 predicted from upstream source).
// What is NOT verified is the link and the run, because no image with v0.22.1 exists yet.
// Do not read "it compiles" as "it works".
// ---------------------------------------------------------------------------
#if LUMI_VLLM_STABLE_ABI
void silu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input);
#else
void silu_and_mul(at::Tensor& out, at::Tensor& input);
#endif
void paged_attention_v1(
    at::Tensor& out, at::Tensor& query, at::Tensor& key_cache, at::Tensor& value_cache,
    int64_t num_kv_heads, double scale, at::Tensor& block_tables, at::Tensor& seq_lens,
    int64_t block_size, int64_t max_seq_len, const std::optional<at::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, at::Tensor& k_scale, at::Tensor& v_scale,
    int64_t tp_rank, int64_t blocksparse_local_blocks, int64_t blocksparse_vert_stride,
    int64_t blocksparse_block_size, int64_t blocksparse_head_sliding_step);

#if LUMI_VLLM_STABLE_ABI
static torch::headeronly::ScalarType to_stable(ffi::DataType t) {
  using S = torch::headeronly::ScalarType;
  switch (t) {
    case ffi::DataType::F32:  return S::Float;
    case ffi::DataType::F16:  return S::Half;
    case ffi::DataType::BF16: return S::BFloat16;
    case ffi::DataType::S32:  return S::Int;
    case ffi::DataType::S64:  return S::Long;
    default:                  return S::Float;
  }
}
#endif

static at::ScalarType to_aten(ffi::DataType t) {
  switch (t) {
    case ffi::DataType::F32:  return at::kFloat;
    case ffi::DataType::F16:  return at::kHalf;
    case ffi::DataType::BF16: return at::kBFloat16;
    case ffi::DataType::S32:  return at::kInt;
    case ffi::DataType::S64:  return at::kLong;
    default:                  return at::kFloat;
  }
}

// Zero-copy at::Tensor view over an XLA device buffer (no deleter -> torch won't free).
static at::Tensor view(void* p, std::vector<int64_t> sizes, ffi::DataType dt, int dev) {
  auto opts = at::TensorOptions().dtype(to_aten(dt)).device(at::kCUDA, dev);
  return at::from_blob(p, std::move(sizes), opts);
}

static ffi::Error SiluMulImpl(hipStream_t stream, ffi::AnyBuffer input,
                              ffi::Result<ffi::AnyBuffer> out) {
  auto dims = input.dimensions();
  if (dims.size() != 2) return ffi::Error::InvalidArgument("input must be (N, 2d)");
  const int64_t N = dims[0], twod = dims[1];
  if (twod % 2 != 0) return ffi::Error::InvalidArgument("last dim must be even");
  const int64_t d = twod / 2;

  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) return ffi::Error::Internal("hipGetDevice");

  // Run the torch op on XLA's stream so it orders against the rest of the graph.
  auto tstream = c10::hip::getStreamFromExternal(stream, dev);
  c10::hip::HIPStreamGuard guard(tstream);

#if LUMI_VLLM_STABLE_ABI
  // torch::stable::from_blob requires EXPLICIT strides -- unlike at::from_blob, which
  // infers contiguous ones. Passing the wrong strides here would not fail loudly; it
  // would read the right memory in the wrong order, so they are spelled out.
  namespace th = torch::headeronly;
  auto sdtype = to_stable(input.element_type());
  auto sdev   = torch::stable::Device(th::DeviceType::CUDA, dev);  // CUDA == HIP on ROCm
  int64_t in_sizes[2]  = {N, twod}, in_strides[2]  = {twod, 1};
  int64_t out_sizes[2] = {N, d},    out_strides[2] = {d, 1};

  torch::stable::Tensor in_t  = torch::stable::from_blob(
      input.untyped_data(), th::IntHeaderOnlyArrayRef(in_sizes, 2),
      th::IntHeaderOnlyArrayRef(in_strides, 2), sdev, sdtype);
  torch::stable::Tensor out_t = torch::stable::from_blob(
      out->untyped_data(), th::IntHeaderOnlyArrayRef(out_sizes, 2),
      th::IntHeaderOnlyArrayRef(out_strides, 2), sdev, sdtype);
#else
  auto opts = at::TensorOptions().dtype(to_aten(input.element_type()))
                  .device(at::kCUDA, dev);  // at::kCUDA == HIP on ROCm
  // Zero-copy views over XLA-owned device memory (no deleter -> torch won't free).
  at::Tensor in_t  = at::from_blob(input.untyped_data(), {N, twod}, opts);
  at::Tensor out_t = at::from_blob(out->untyped_data(), {N, d},   opts);
#endif

  silu_and_mul(out_t, in_t);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LumiSiluAndMul, SiluMulImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()   // input (N, 2d)
        .Ret<ffi::AnyBuffer>()); // out   (N, d)

// ---------------------------------------------------------------------------
// paged_attention_v1 (decode). The real application of the libtorch-in-FFI path:
// replaces the eager dlpack bridge (clones + host sync) with a jittable handler.
//   query       (num_seqs, num_heads, head_size)
//   key_cache   (num_blocks, num_kv_heads, head_size/X, block_size, X)
//   value_cache (num_blocks, num_kv_heads, head_size, block_size)
//   block_tables(num_seqs, max_blocks) int32 ; seq_lens (num_seqs,) int32
//   k_scale,v_scale (1,) f32 (==1 for non-fp8 kv cache) ; out (num_seqs, num_heads, head_size)
// ---------------------------------------------------------------------------
static ffi::Error PagedAttnImpl(
    hipStream_t stream, ffi::AnyBuffer query, ffi::AnyBuffer key_cache,
    ffi::AnyBuffer value_cache, ffi::AnyBuffer block_tables, ffi::AnyBuffer seq_lens,
    ffi::AnyBuffer k_scale, ffi::AnyBuffer v_scale, int32_t num_kv_heads, double scale,
    int32_t block_size, int32_t max_seq_len, ffi::Result<ffi::AnyBuffer> out) {
  auto qd = query.dimensions();
  if (qd.size() != 3) return ffi::Error::InvalidArgument("query must be (S, H, D)");
  auto kd = key_cache.dimensions(), vd = value_cache.dimensions();
  auto bd = block_tables.dimensions(), sd = seq_lens.dimensions();

  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) return ffi::Error::Internal("hipGetDevice");
  auto tstream = c10::hip::getStreamFromExternal(stream, dev);
  c10::hip::HIPStreamGuard guard(tstream);

  auto qt = query.element_type();
  at::Tensor t_q  = view(query.untyped_data(), {qd[0], qd[1], qd[2]}, qt, dev);
  at::Tensor t_o  = view(out->untyped_data(), {qd[0], qd[1], qd[2]}, qt, dev);
  at::Tensor t_kc = view(key_cache.untyped_data(), {kd[0], kd[1], kd[2], kd[3], kd[4]}, qt, dev);
  at::Tensor t_vc = view(value_cache.untyped_data(), {vd[0], vd[1], vd[2], vd[3]}, qt, dev);
  at::Tensor t_bt = view(block_tables.untyped_data(), {bd[0], bd[1]},
                         block_tables.element_type(), dev);
  at::Tensor t_sl = view(seq_lens.untyped_data(), {sd[0]}, seq_lens.element_type(), dev);
  at::Tensor t_ks = view(k_scale.untyped_data(), {1}, k_scale.element_type(), dev);
  at::Tensor t_vs = view(v_scale.untyped_data(), {1}, v_scale.element_type(), dev);

  paged_attention_v1(t_o, t_q, t_kc, t_vc, (int64_t)num_kv_heads, scale, t_bt, t_sl,
                     (int64_t)block_size, (int64_t)max_seq_len, std::nullopt, "auto",
                     t_ks, t_vs, /*tp_rank=*/0, 0, 0, 0, 0);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LumiPagedAttnV1, PagedAttnImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()   // query
        .Arg<ffi::AnyBuffer>()   // key_cache
        .Arg<ffi::AnyBuffer>()   // value_cache
        .Arg<ffi::AnyBuffer>()   // block_tables (int32)
        .Arg<ffi::AnyBuffer>()   // seq_lens (int32)
        .Arg<ffi::AnyBuffer>()   // k_scale (f32, 1)
        .Arg<ffi::AnyBuffer>()   // v_scale (f32, 1)
        .Attr<int32_t>("num_kv_heads")
        .Attr<double>("scale")
        .Attr<int32_t>("block_size")
        .Attr<int32_t>("max_seq_len")
        .Ret<ffi::AnyBuffer>()); // out
