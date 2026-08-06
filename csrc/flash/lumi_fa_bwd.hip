// Backward: dQ/dK/dV via the CK fmha_bwd<2> dispatcher (gfx90a).
//
// Full port of flash_attn's mha_bwd capability (this CK build): GQA/MQA, cross-
// attention (S_q != S_k), window/local (mask_info::decode), ALiBi, dropout
// (philox seed/offset passed in), deterministic option, fp16 + bf16, hdim <= 256.
//
// For GQA the kernel writes dk/dv at H_q heads (expanded); this handler returns
// them at H_q and the Python custom_vjp sums over the H_q/H_k groups -> H_k (the
// same reduction mha_bwd does with at::sum_out). Separate TU from fwd (their
// example headers can't co-include); both link into liblumi_fa_ffi.so.
//
// Scratch (dq_acc f32, zeroed + softmax_d f32) via XLA's ScratchAllocator.
// q,k,v,out,do = (B,S,H,D); dq,dk,dv returned at (B,S,H_q,D); lse,d = (B,Hq,Sq) f32.

#include <cstdint>
#include <string>

#include <hip/hip_runtime.h>
#include "fmha_bwd.hpp"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

static mask_info make_mask(bool is_causal, int win_l, int win_r, int Sq, int Sk) {
  if (win_l >= Sk) win_l = -1;
  if (win_r >= Sk) win_r = -1;
  if (is_causal)
    return mask_info::decode("b:" + std::to_string(win_l) + ",0", Sq, Sk);
  if (win_l == -1 && win_r == -1)
    return mask_info::decode("0", Sq, Sk);
  return mask_info::decode("b:" + std::to_string(win_l) + "," + std::to_string(win_r), Sq, Sk);
}

static int32_t ceil_div(int32_t a, int32_t b) { return (a + b - 1) / b; }

static ffi::Error BwdImpl(hipStream_t stream, ffi::ScratchAllocator scratch,
                          ffi::AnyBuffer dout, ffi::AnyBuffer q, ffi::AnyBuffer k,
                          ffi::AnyBuffer v, ffi::AnyBuffer out, ffi::AnyBuffer lse,
                          ffi::AnyBuffer alibi, ffi::AnyBuffer drop_rng,
                          double scale, int32_t is_bf16, int32_t is_causal,
                          int32_t window_left, int32_t window_right,
                          int32_t has_alibi, double p_drop, int32_t deterministic,
                          ffi::Result<ffi::AnyBuffer> dq,
                          ffi::Result<ffi::AnyBuffer> dk,    // expanded: (B,Sk,Hq,D)
                          ffi::Result<ffi::AnyBuffer> dv) {  // expanded: (B,Sk,Hq,D)
  auto dd = q.dimensions();
  if (dd.size() != 4)
    return ffi::Error::InvalidArgument("q must be rank 4 (B,S,H,D)");
  const int64_t B = dd[0], Sq = dd[1], Hq = dd[2], D = dd[3];
  const int64_t Sk = k.dimensions()[1], Hk = k.dimensions()[2];
  if (Hq % Hk != 0)
    return ffi::Error::InvalidArgument("H_q must be a multiple of H_k");

  const mask_info mask = make_mask(is_causal, window_left, window_right, (int)Sq, (int)Sk);
  const bool has_dropout = p_drop > 0.0;
  const bool det = deterministic != 0;

  // scratch: softmax_d (B,Hq,Sq) f32; dq_acc (nsplits,B,Sq,Hq,D) f32, zeroed.
  // nsplits must cover ceil(seqlen_k / <kernel's k-tile>): the deterministic dq grad is
  // reduced over one split per k-tile (convert_dq). The kernel's true k-tile is an
  // internal detail, so size with kN0=16 -- the conservative split granularity CK's own
  // fmha_bwd_runner.hpp uses ("<= minimal bn0 of all tiles"). A larger kN0 (e.g. 128)
  // UNDER-allocates splits whenever seqlen_k isn't a multiple of the real tile -> OOB
  // read in convert_dq -> non-reproducible/garbage dq (memory fault if seqlen_k>max).
  const int32_t kN0 = 16;
  const int32_t nsplits = det ? ceil_div((int32_t)Sk, kN0) : 1;
  const size_t d_bytes = (size_t)B * Hq * Sq * sizeof(float);
  const size_t dqacc_bytes = (size_t)nsplits * B * Sq * Hq * D * sizeof(float);
  auto d_mem = scratch.Allocate(d_bytes, 16);
  auto dqacc_mem = scratch.Allocate(dqacc_bytes, 16);
  if (!d_mem || !dqacc_mem)
    return ffi::Error::Internal("scratch allocation failed");
  if (hipMemsetAsync(*dqacc_mem, 0, dqacc_bytes, stream) != hipSuccess)
    return ffi::Error::Internal("hipMemsetAsync(dq_acc) failed");

  fmha_bwd_traits traits;
  traits.hdim_q = (int)D;
  traits.hdim_v = (int)D;
  traits.data_type = is_bf16 ? "bf16" : "fp16";
  traits.is_group_mode = false;
  traits.mask_type = mask.type;
  traits.bias_type = has_alibi ? bias_enum::alibi : bias_enum::no_bias;
  traits.has_dbias = false;
  traits.has_dropout = has_dropout;
  traits.is_store_randval = false;
  traits.is_deterministic = det;

  const int32_t nhs = (int32_t)D;
  const int32_t stride_q = (int32_t)(Hq * D), stride_kv = (int32_t)(Hk * D);
  const int32_t stride_exp = (int32_t)(Hq * D); // dk/dv expanded at Hq heads
  const int32_t bstride_q = (int32_t)(Sq * Hq * D), bstride_kv = (int32_t)(Sk * Hk * D);
  const int32_t bstride_dk_exp = (int32_t)(Sk * Hq * D);
  const int32_t nhs_lsed = (int32_t)Sq;
  const int32_t bs_lsed = (int32_t)(Hq * Sq);
  const int32_t stride_alibi = (has_alibi && alibi.dimensions().size() == 2) ? (int32_t)Hq : 0;

  fmha_bwd_args args{};
  args.q_ptr = q.untyped_data();
  args.k_ptr = k.untyped_data();
  args.v_ptr = v.untyped_data();
  args.bias_ptr = has_alibi ? alibi.untyped_data() : nullptr;
  args.o_ptr = out.untyped_data();
  args.lse_ptr = lse.untyped_data();
  args.do_ptr = dout.untyped_data();
  args.d_ptr = *d_mem;
  args.rand_val_ptr = nullptr;
  args.dq_ptr = dq->untyped_data();
  args.dk_ptr = dk->untyped_data();  // expanded Hq
  args.dv_ptr = dv->untyped_data();  // expanded Hq
  args.dbias_ptr = nullptr;
  args.dq_acc_ptr = *dqacc_mem;
  args.seqlen_q = (int32_t)Sq;
  args.seqlen_k = (int32_t)Sk;
  args.batch = (int32_t)B;
  args.max_seqlen_q = (int32_t)Sq;
  args.max_seqlen_k = (int32_t)Sk;
  args.hdim_q = (int32_t)D;
  args.hdim_v = (int32_t)D;
  args.nhead_q = (int32_t)Hq;
  args.nhead_k = (int32_t)Hk;
  args.scale = (float)scale;
  args.stride_q = stride_q;
  args.stride_k = args.stride_v = stride_kv;
  args.stride_bias = stride_alibi;
  args.stride_o = stride_q;
  args.stride_randval = 0;
  args.stride_do = stride_q;
  args.stride_dq_acc = stride_exp;
  args.stride_dq = stride_q;
  args.stride_dk = args.stride_dv = stride_exp;
  args.stride_dbias = 0;
  args.nhead_stride_q = args.nhead_stride_k = args.nhead_stride_v = nhs;
  args.nhead_stride_o = args.nhead_stride_do = nhs;
  args.nhead_stride_bias = 0;
  args.nhead_stride_randval = 0;
  args.nhead_stride_lsed = nhs_lsed;
  args.nhead_stride_dq_acc = nhs;
  args.nhead_stride_dq = args.nhead_stride_dk = args.nhead_stride_dv = nhs;
  args.nhead_stride_dbias = 0;
  args.batch_stride_q = bstride_q;
  args.batch_stride_k = args.batch_stride_v = bstride_kv;
  args.batch_stride_bias = 0;
  args.batch_stride_o = args.batch_stride_do = bstride_q;
  args.batch_stride_randval = 0;
  args.batch_stride_lsed = bs_lsed;
  args.batch_stride_dq_acc = (int32_t)(Sq * Hq * D);
  args.batch_stride_dq = (int32_t)(Sq * Hq * D);
  args.batch_stride_dk = args.batch_stride_dv = bstride_dk_exp;
  args.batch_stride_dbias = 0;
  args.split_stride_dq_acc = (int32_t)(B * Sq * Hq * D);
  args.window_size_left = mask.left;
  args.window_size_right = mask.right;
  args.mask_type = (int32_t)mask.type;
  args.p_drop = (float)p_drop;
  args.p_undrop = 1.0f - (float)p_drop;
  // (seed, offset) as DEVICE pointers -- same buffer the fwd used (threaded residual).
  const uint64_t* rng = static_cast<const uint64_t*>(drop_rng.untyped_data());
  args.drop_seed_offset = std::make_pair((const void*)rng, (const void*)(rng + 1));

  ck_tile::stream_config cfg;
  cfg.stream_id_ = stream;
  float t = fmha_bwd(traits, args, cfg);
  if (t < 0.f)
    return ffi::Error::Internal(
        "fmha_bwd: no CK kernel instance for this (head dim/dtype/mask) config");
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LumiFaBwd, BwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::AnyBuffer>()  // dout
        .Arg<ffi::AnyBuffer>()  // q
        .Arg<ffi::AnyBuffer>()  // k
        .Arg<ffi::AnyBuffer>()  // v
        .Arg<ffi::AnyBuffer>()  // out
        .Arg<ffi::AnyBuffer>()  // lse
        .Arg<ffi::AnyBuffer>()  // alibi (dummy if unused)
        .Arg<ffi::AnyBuffer>()  // drop_rng (2,) u64 device (seed, offset)
        .Attr<double>("scale")
        .Attr<int32_t>("is_bf16")
        .Attr<int32_t>("is_causal")
        .Attr<int32_t>("window_left")
        .Attr<int32_t>("window_right")
        .Attr<int32_t>("has_alibi")
        .Attr<double>("p_drop")
        .Attr<int32_t>("deterministic")
        .Ret<ffi::AnyBuffer>()   // dq    (B,Sq,Hq,D)
        .Ret<ffi::AnyBuffer>()   // dk_exp(B,Sk,Hq,D)
        .Ret<ffi::AnyBuffer>()); // dv_exp(B,Sk,Hq,D)

// ---------------------------------------------------------------------------
// Varlen / packed backward. q/k/v/out/do packed (total,H,D); lse (Hq,total_q);
// cu_seqlens_q/k (B+1,i32). Group mode. dq (total_q,Hq,D), dk/dv expanded to
// (total_k,Hq,D) (Python sums groups -> Hk). Scratch: softmax_d (B,Hq,max_sq) f32,
// dq_acc (nsplits,total_q,Hq,D) f32 zeroed. Mirrors mha_varlen_bwd.
// ---------------------------------------------------------------------------
static ffi::Error VarlenBwdImpl(hipStream_t stream, ffi::ScratchAllocator scratch,
                                ffi::AnyBuffer dout, ffi::AnyBuffer q, ffi::AnyBuffer k,
                                ffi::AnyBuffer v, ffi::AnyBuffer out, ffi::AnyBuffer lse,
                                ffi::AnyBuffer cu_q, ffi::AnyBuffer cu_k,
                                ffi::AnyBuffer alibi, ffi::AnyBuffer drop_rng,
                                double scale, int32_t is_bf16, int32_t is_causal,
                                int32_t window_left, int32_t window_right,
                                int32_t has_alibi, double p_drop, int32_t deterministic,
                                int32_t max_seqlen_q, int32_t max_seqlen_k,
                                ffi::Result<ffi::AnyBuffer> dq,
                                ffi::Result<ffi::AnyBuffer> dk,
                                ffi::Result<ffi::AnyBuffer> dv) {
  auto dd = q.dimensions();
  if (dd.size() != 3)
    return ffi::Error::InvalidArgument("varlen q must be rank 3 (total,H,D)");
  const int64_t total_q = dd[0], Hq = dd[1], D = dd[2];
  const int64_t total_k = k.dimensions()[0], Hk = k.dimensions()[1];
  const int32_t B = (int32_t)(cu_q.dimensions()[0] - 1);
  if (Hq % Hk != 0)
    return ffi::Error::InvalidArgument("H_q must be a multiple of H_k");

  const mask_info mask = make_mask(is_causal, window_left, window_right,
                                   max_seqlen_q, max_seqlen_k);
  const bool has_dropout = p_drop > 0.0;
  const bool det = deterministic != 0;

  // kN0=16: conservative deterministic-split granularity (see BwdImpl note). nsplits is
  // sized from max_seqlen_k, the caller's static upper bound -- which MUST bound every
  // per-sequence seqlen_k (cu_seqlens are device/traced; we can't clamp host-side as CK's
  // reference runner does). Passing a seqlen_k > max_seqlen_k under-allocates -> OOB.
  const int32_t kN0 = 16;
  const int32_t nsplits = det ? ceil_div(max_seqlen_k, kN0) : 1;
  const size_t d_bytes = (size_t)B * Hq * max_seqlen_q * sizeof(float);
  const size_t dqacc_bytes = (size_t)nsplits * total_q * Hq * D * sizeof(float);
  auto d_mem = scratch.Allocate(d_bytes, 16);
  auto dqacc_mem = scratch.Allocate(dqacc_bytes, 16);
  if (!d_mem || !dqacc_mem)
    return ffi::Error::Internal("scratch allocation failed");
  if (hipMemsetAsync(*dqacc_mem, 0, dqacc_bytes, stream) != hipSuccess)
    return ffi::Error::Internal("hipMemsetAsync(dq_acc) failed");

  fmha_bwd_traits traits;
  traits.hdim_q = (int)D;
  traits.hdim_v = (int)D;
  traits.data_type = is_bf16 ? "bf16" : "fp16";
  traits.is_group_mode = true;
  traits.mask_type = mask.type;
  traits.bias_type = has_alibi ? bias_enum::alibi : bias_enum::no_bias;
  traits.has_dbias = false;
  traits.has_dropout = has_dropout;
  traits.is_store_randval = false;
  traits.is_deterministic = det;

  const int32_t nhs = (int32_t)D;
  const int32_t stride_q = (int32_t)(Hq * D), stride_kv = (int32_t)(Hk * D);
  const int32_t stride_exp = (int32_t)(Hq * D);
  const int32_t stride_alibi = (has_alibi && alibi.dimensions().size() == 2) ? (int32_t)Hq : 0;

  fmha_bwd_args args{};
  args.q_ptr = q.untyped_data();
  args.k_ptr = k.untyped_data();
  args.v_ptr = v.untyped_data();
  args.bias_ptr = has_alibi ? alibi.untyped_data() : nullptr;
  args.o_ptr = out.untyped_data();
  args.lse_ptr = lse.untyped_data();
  args.do_ptr = dout.untyped_data();
  args.d_ptr = *d_mem;
  args.rand_val_ptr = nullptr;
  args.dq_ptr = dq->untyped_data();
  args.dk_ptr = dk->untyped_data();
  args.dv_ptr = dv->untyped_data();
  args.dbias_ptr = nullptr;
  args.dq_acc_ptr = *dqacc_mem;
  args.seqstart_q_ptr = cu_q.untyped_data();
  args.seqstart_k_ptr = cu_k.untyped_data();
  args.seqlen_q = (int32_t)total_q;
  args.seqlen_k = (int32_t)total_k;
  args.batch = B;
  args.max_seqlen_q = max_seqlen_q;
  args.max_seqlen_k = max_seqlen_k;
  args.hdim_q = (int32_t)D;
  args.hdim_v = (int32_t)D;
  args.nhead_q = (int32_t)Hq;
  args.nhead_k = (int32_t)Hk;
  args.scale = (float)scale;
  args.stride_q = stride_q;
  args.stride_k = args.stride_v = stride_kv;
  args.stride_bias = stride_alibi;
  args.stride_o = stride_q;
  args.stride_randval = 0;
  args.stride_do = stride_q;
  args.stride_dq_acc = stride_exp;
  args.stride_dq = stride_q;
  args.stride_dk = args.stride_dv = stride_exp;
  args.stride_dbias = 0;
  args.nhead_stride_q = args.nhead_stride_k = args.nhead_stride_v = nhs;
  args.nhead_stride_o = args.nhead_stride_do = nhs;
  args.nhead_stride_bias = 0;
  args.nhead_stride_randval = 0;
  args.nhead_stride_lsed = (int32_t)total_q;  // lse/d share stride (group mode)
  args.nhead_stride_dq_acc = nhs;
  args.nhead_stride_dq = args.nhead_stride_dk = args.nhead_stride_dv = nhs;
  args.nhead_stride_dbias = 0;
  args.batch_stride_q = args.batch_stride_k = args.batch_stride_v = 0;
  args.batch_stride_o = args.batch_stride_do = 0;
  args.batch_stride_bias = 0;
  args.batch_stride_lsed = 0;
  args.batch_stride_dq_acc = 0;
  args.batch_stride_dq = args.batch_stride_dk = args.batch_stride_dv = 0;
  args.batch_stride_dbias = 0;
  args.split_stride_dq_acc = (int32_t)(total_q * Hq * D);
  args.window_size_left = mask.left;
  args.window_size_right = mask.right;
  args.mask_type = (int32_t)mask.type;
  args.p_drop = (float)p_drop;
  args.p_undrop = 1.0f - (float)p_drop;
  const uint64_t* rng = static_cast<const uint64_t*>(drop_rng.untyped_data());
  args.drop_seed_offset = std::make_pair((const void*)rng, (const void*)(rng + 1));

  ck_tile::stream_config cfg;
  cfg.stream_id_ = stream;
  float t = fmha_bwd(traits, args, cfg);
  if (t < 0.f)
    return ffi::Error::Internal("fmha_bwd (varlen): no CK kernel instance for this config");
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LumiFaVarlenBwd, VarlenBwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::AnyBuffer>()  // dout (total_q,Hq,D)
        .Arg<ffi::AnyBuffer>()  // q
        .Arg<ffi::AnyBuffer>()  // k
        .Arg<ffi::AnyBuffer>()  // v
        .Arg<ffi::AnyBuffer>()  // out
        .Arg<ffi::AnyBuffer>()  // lse (Hq,total_q)
        .Arg<ffi::AnyBuffer>()  // cu_seqlens_q (B+1,i32)
        .Arg<ffi::AnyBuffer>()  // cu_seqlens_k (B+1,i32)
        .Arg<ffi::AnyBuffer>()  // alibi
        .Arg<ffi::AnyBuffer>()  // drop_rng (2,) u64 device (seed, offset)
        .Attr<double>("scale")
        .Attr<int32_t>("is_bf16")
        .Attr<int32_t>("is_causal")
        .Attr<int32_t>("window_left")
        .Attr<int32_t>("window_right")
        .Attr<int32_t>("has_alibi")
        .Attr<double>("p_drop")
        .Attr<int32_t>("deterministic")
        .Attr<int32_t>("max_seqlen_q")
        .Attr<int32_t>("max_seqlen_k")
        .Ret<ffi::AnyBuffer>()   // dq     (total_q,Hq,D)
        .Ret<ffi::AnyBuffer>()   // dk_exp (total_k,Hq,D)
        .Ret<ffi::AnyBuffer>()); // dv_exp (total_k,Hq,D)
