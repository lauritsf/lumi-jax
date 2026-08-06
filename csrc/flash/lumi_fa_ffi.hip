// Forward: XLA FFI custom call -> container's CK fmha_fwd (gfx90a).
//
// Full port of flash_attn's mha_fwd capability (this CK build): GQA/MQA
// (H_k <= H_q), cross-attention (S_q != S_k), sliding-window / local attention
// (window_left/right via mask_info::decode, exactly as mha_fwd does), ALiBi bias,
// dropout (philox seed/offset passed in), fp16 + bf16, head dim <= 256. (softcap is
// /*commented out*/ in this build, so unsupported here too.)
//
// Built with hipcc against ck_tile headers and linked to a prebuilt CK kernel library --
// standalone libck_fmha.so, or flash_attn_2_cuda.so where that is all the image has. Only
// the dispatchers are called; no kernels are recompiled.
//
// Layout: q (B,Sq,Hq,D); k,v (B,Sk,Hk,D); out (B,Sq,Hq,D); lse (B,Hq,Sq) f32.
// alibi: (Hq) or (B,Hq) f32 when has_alibi, else a dummy buffer (ignored).

#include <cstdint>
#include <string>

#include "fmha_fwd.hpp"  // fmha_fwd_traits/args/fmha_fwd, mask_info, mask_enum, bias_enum, ck_tile::stream_config
#include "rotary.hpp"    // rope_enum (appendkv fused rotary)
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// Build the CK mask exactly as mha_fwd does (causal / no-mask / local window).
static mask_info make_mask(bool is_causal, int win_l, int win_r, int Sq, int Sk) {
  if (win_l >= Sk) win_l = -1;
  if (win_r >= Sk) win_r = -1;
  if (is_causal)
    return mask_info::decode("b:" + std::to_string(win_l) + ",0", Sq, Sk);
  if (win_l == -1 && win_r == -1)
    return mask_info::decode("0", Sq, Sk);
  return mask_info::decode("b:" + std::to_string(win_l) + "," + std::to_string(win_r), Sq, Sk);
}

static ffi::Error FwdImpl(hipStream_t stream,
                          ffi::AnyBuffer q, ffi::AnyBuffer k, ffi::AnyBuffer v,
                          ffi::AnyBuffer alibi, ffi::AnyBuffer drop_rng,
                          double scale, int32_t is_bf16, int32_t is_causal,
                          int32_t window_left, int32_t window_right,
                          int32_t has_alibi, double p_drop,
                          ffi::Result<ffi::AnyBuffer> out,
                          ffi::Result<ffi::AnyBuffer> lse) {
  auto dq = q.dimensions();
  if (dq.size() != 4 || k.dimensions().size() != 4)
    return ffi::Error::InvalidArgument("q,k must be rank 4 (B,S,H,D)");
  const int64_t B = dq[0], Sq = dq[1], Hq = dq[2], D = dq[3];
  const int64_t Sk = k.dimensions()[1], Hk = k.dimensions()[2];
  if (Hq % Hk != 0)
    return ffi::Error::InvalidArgument("H_q must be a multiple of H_k");

  const mask_info mask = make_mask(is_causal, window_left, window_right, (int)Sq, (int)Sk);
  const bool has_dropout = p_drop > 0.0;

  fmha_fwd_traits traits;
  traits.hdim_q = (int)D;
  traits.hdim_v = (int)D;
  traits.data_type = is_bf16 ? "bf16" : "fp16";
  traits.is_group_mode = false;
  traits.is_v_rowmajor = true;
  traits.has_logits_soft_cap = false;
  traits.mask_type = mask.type;
  traits.bias_type = has_alibi ? bias_enum::alibi : bias_enum::no_bias;
  traits.has_lse = true;
  traits.has_dropout = has_dropout;
  traits.qscale_type = quant_scale_enum::no_scale;
  traits.skip_min_seqlen_q = false;

  const int32_t nhs = (int32_t)D;
  const int32_t stride_q = (int32_t)(Hq * D), stride_kv = (int32_t)(Hk * D);
  const int32_t bstride_q = (int32_t)(Sq * Hq * D), bstride_kv = (int32_t)(Sk * Hk * D);
  // ALiBi slopes (Hq) -> stride 0, (B,Hq) -> stride Hq
  const int32_t stride_alibi = (has_alibi && alibi.dimensions().size() == 2) ? (int32_t)Hq : 0;

  fmha_fwd_args args{};
  args.q_ptr = q.untyped_data();
  args.k_ptr = k.untyped_data();
  args.v_ptr = v.untyped_data();
  args.bias_ptr = has_alibi ? alibi.untyped_data() : nullptr;
  args.q_descale_ptr = args.k_descale_ptr = args.v_descale_ptr = nullptr;
  args.rand_val_ptr = nullptr;
  args.lse_ptr = lse->untyped_data();
  args.o_ptr = out->untyped_data();
  args.seqlen_q = (int32_t)Sq;
  args.seqlen_k = (int32_t)Sk;
  args.batch = (int32_t)B;
  args.max_seqlen_q = (int32_t)Sq;
  args.hdim_q = (int32_t)D;
  args.hdim_v = (int32_t)D;
  args.nhead_q = (int32_t)Hq;
  args.nhead_k = (int32_t)Hk;
  args.scale_s = (float)scale;
  args.logits_soft_cap = 0.f;
  args.stride_q = args.stride_o = stride_q;
  args.stride_k = args.stride_v = stride_kv;
  args.stride_bias = stride_alibi;
  args.stride_randval = 0;
  args.nhead_stride_q = args.nhead_stride_k = args.nhead_stride_v = args.nhead_stride_o = nhs;
  args.nhead_stride_bias = 0;
  args.nhead_stride_randval = 0;
  args.nhead_stride_lse = (int32_t)Sq;
  args.batch_stride_q = args.batch_stride_o = bstride_q;
  args.batch_stride_k = args.batch_stride_v = bstride_kv;
  args.batch_stride_bias = 0;
  args.batch_stride_randval = 0;
  args.batch_stride_lse = (int32_t)(Hq * Sq);
  args.window_size_left = mask.left;
  args.window_size_right = mask.right;
  args.mask_type = (int32_t)mask.type;
  args.min_seqlen_q = 0;
  args.p_drop = (float)p_drop;
  args.s_randval = false;
  // (seed, offset) as DEVICE pointers (philox state read on-device each step ->
  // per-step dropout under jit without recompiling).
  const uint64_t* rng = static_cast<const uint64_t*>(drop_rng.untyped_data());
  args.drop_seed_offset = std::make_pair((const void*)rng, (const void*)(rng + 1));

  ck_tile::stream_config cfg;
  cfg.stream_id_ = stream;
  float t = fmha_fwd(traits, args, cfg);
  if (t < 0.f)
    return ffi::Error::Internal(
        "fmha_fwd: no CK kernel instance for this (head dim/dtype/mask) config");
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LumiFaFwd, FwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()  // q
        .Arg<ffi::AnyBuffer>()  // k
        .Arg<ffi::AnyBuffer>()  // v
        .Arg<ffi::AnyBuffer>()  // alibi (dummy if unused)
        .Arg<ffi::AnyBuffer>()  // drop_rng (2,) u64 device (seed, offset)
        .Attr<double>("scale")
        .Attr<int32_t>("is_bf16")
        .Attr<int32_t>("is_causal")
        .Attr<int32_t>("window_left")
        .Attr<int32_t>("window_right")
        .Attr<int32_t>("has_alibi")
        .Attr<double>("p_drop")
        .Ret<ffi::AnyBuffer>()   // out
        .Ret<ffi::AnyBuffer>()); // lse

// ---------------------------------------------------------------------------
// Varlen / packed forward: q (total_q, Hq, D), k/v (total_k, Hk, D), packed; per-
// sequence boundaries in cu_seqlens_q/k (B+1, int32). Group mode (batch_stride=0,
// seqstart = cu_seqlens). out (total_q, Hq, D), lse (Hq, total_q). Mirrors
// mha_varlen_fwd; mask via mask_info::decode(max_seqlen_q, max_seqlen_k).
// ---------------------------------------------------------------------------
static ffi::Error VarlenFwdImpl(hipStream_t stream,
                                ffi::AnyBuffer q, ffi::AnyBuffer k, ffi::AnyBuffer v,
                                ffi::AnyBuffer cu_q, ffi::AnyBuffer cu_k,
                                ffi::AnyBuffer alibi, ffi::AnyBuffer drop_rng,
                                double scale, int32_t is_bf16, int32_t is_causal,
                                int32_t window_left, int32_t window_right,
                                int32_t has_alibi, double p_drop,
                                int32_t max_seqlen_q, int32_t max_seqlen_k,
                                ffi::Result<ffi::AnyBuffer> out,
                                ffi::Result<ffi::AnyBuffer> lse) {
  auto dq = q.dimensions();
  if (dq.size() != 3 || k.dimensions().size() != 3)
    return ffi::Error::InvalidArgument("varlen q,k must be rank 3 (total, H, D)");
  const int64_t total_q = dq[0], Hq = dq[1], D = dq[2];
  const int64_t total_k = k.dimensions()[0], Hk = k.dimensions()[1];
  const int64_t Bp1 = cu_q.dimensions()[0];
  const int32_t B = (int32_t)(Bp1 - 1);
  if (Hq % Hk != 0)
    return ffi::Error::InvalidArgument("H_q must be a multiple of H_k");

  const mask_info mask = make_mask(is_causal, window_left, window_right,
                                   max_seqlen_q, max_seqlen_k);
  const bool has_dropout = p_drop > 0.0;

  fmha_fwd_traits traits;
  traits.hdim_q = (int)D;
  traits.hdim_v = (int)D;
  traits.data_type = is_bf16 ? "bf16" : "fp16";
  traits.is_group_mode = true;
  traits.is_v_rowmajor = true;
  traits.has_logits_soft_cap = false;
  traits.mask_type = mask.type;
  traits.bias_type = has_alibi ? bias_enum::alibi : bias_enum::no_bias;
  traits.has_lse = true;
  traits.has_dropout = has_dropout;
  traits.qscale_type = quant_scale_enum::no_scale;
  traits.skip_min_seqlen_q = false;

  const int32_t nhs = (int32_t)D;
  const int32_t stride_q = (int32_t)(Hq * D), stride_kv = (int32_t)(Hk * D);
  const int32_t stride_alibi = (has_alibi && alibi.dimensions().size() == 2) ? (int32_t)Hq : 0;

  fmha_fwd_args args{};
  args.q_ptr = q.untyped_data();
  args.k_ptr = k.untyped_data();
  args.v_ptr = v.untyped_data();
  args.bias_ptr = has_alibi ? alibi.untyped_data() : nullptr;
  args.q_descale_ptr = args.k_descale_ptr = args.v_descale_ptr = nullptr;
  args.rand_val_ptr = nullptr;
  args.lse_ptr = lse->untyped_data();
  args.o_ptr = out->untyped_data();
  args.seqstart_q_ptr = cu_q.untyped_data();
  args.seqstart_k_ptr = cu_k.untyped_data();
  args.seqlen_q = (int32_t)total_q;
  args.seqlen_k = (int32_t)total_k;
  args.batch = B;
  args.max_seqlen_q = max_seqlen_q;
  args.hdim_q = (int32_t)D;
  args.hdim_v = (int32_t)D;
  args.nhead_q = (int32_t)Hq;
  args.nhead_k = (int32_t)Hk;
  args.scale_s = (float)scale;
  args.logits_soft_cap = 0.f;
  args.stride_q = args.stride_o = stride_q;
  args.stride_k = args.stride_v = stride_kv;
  args.stride_bias = stride_alibi;
  args.stride_randval = 0;
  args.nhead_stride_q = args.nhead_stride_k = args.nhead_stride_v = args.nhead_stride_o = nhs;
  args.nhead_stride_bias = 0;
  args.nhead_stride_randval = 0;
  args.nhead_stride_lse = (int32_t)total_q;  // lse (Hq, total_q)
  args.batch_stride_q = args.batch_stride_o = 0;
  args.batch_stride_k = args.batch_stride_v = 0;
  args.batch_stride_bias = 0;
  args.batch_stride_randval = 0;
  args.batch_stride_lse = 0;
  args.window_size_left = mask.left;
  args.window_size_right = mask.right;
  args.mask_type = (int32_t)mask.type;
  args.min_seqlen_q = 0;
  args.p_drop = (float)p_drop;
  args.s_randval = false;
  const uint64_t* rng = static_cast<const uint64_t*>(drop_rng.untyped_data());
  args.drop_seed_offset = std::make_pair((const void*)rng, (const void*)(rng + 1));

  ck_tile::stream_config cfg;
  cfg.stream_id_ = stream;
  float t = fmha_fwd(traits, args, cfg);
  if (t < 0.f)
    return ffi::Error::Internal("fmha_fwd (varlen): no CK kernel instance for this config");
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LumiFaVarlenFwd, VarlenFwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()  // q (total_q,Hq,D)
        .Arg<ffi::AnyBuffer>()  // k (total_k,Hk,D)
        .Arg<ffi::AnyBuffer>()  // v
        .Arg<ffi::AnyBuffer>()  // cu_seqlens_q (B+1, i32)
        .Arg<ffi::AnyBuffer>()  // cu_seqlens_k (B+1, i32)
        .Arg<ffi::AnyBuffer>()  // alibi (dummy if unused)
        .Arg<ffi::AnyBuffer>()  // drop_rng (2,) u64 device (seed, offset)
        .Attr<double>("scale")
        .Attr<int32_t>("is_bf16")
        .Attr<int32_t>("is_causal")
        .Attr<int32_t>("window_left")
        .Attr<int32_t>("window_right")
        .Attr<int32_t>("has_alibi")
        .Attr<double>("p_drop")
        .Attr<int32_t>("max_seqlen_q")
        .Attr<int32_t>("max_seqlen_k")
        .Ret<ffi::AnyBuffer>()   // out (total_q,Hq,D)
        .Ret<ffi::AnyBuffer>()); // lse (Hq,total_q)

// ---------------------------------------------------------------------------
// Decode / KV-cache forward: CK split-KV (fmha_fwd_splitkv) -- the flash-native
// decode path. RAW torch-free dispatcher (no libtorch), same as fmha_fwd. Attends
// new queries against a pre-written contiguous KV cache; split-KV + an internal
// combine keep the GPU busy for long contexts. Contiguous cache, no KV-append, no
// rotary, no paging; per-batch valid length via cache_seqlens (kvcache batch mode ->
// kernel reads seqlen_k_ptr[b]). num_splits is a Python-side heuristic (decode wins up
// to ~8x on small batch).
//   q (B,Sq,Hq,D); k_cache,v_cache (B,Sk,Hk,D); cache_seqlens (B,) i32; out (B,Sq,Hq,D)
// fp32 split workspaces lse_acc/o_acc come from XLA's ScratchAllocator (transient, freed
// post-call). NB: the ROCm FFI allocator only accepts alignment <= 16 bytes -- request 16.
// ---------------------------------------------------------------------------
static ffi::Error KvCacheImpl(hipStream_t stream, ffi::ScratchAllocator scratch,
                              ffi::AnyBuffer q, ffi::AnyBuffer kc, ffi::AnyBuffer vc,
                              ffi::AnyBuffer cache_seqlens,
                              double scale, int32_t is_causal, int32_t num_splits_attr,
                              ffi::Result<ffi::AnyBuffer> out) {
  auto dq = q.dimensions();
  if (dq.size() != 4 || kc.dimensions().size() != 4)
    return ffi::Error::InvalidArgument("q,k_cache must be rank 4 (B,S,H,D)");
  const int64_t B = dq[0], Sq = dq[1], Hq = dq[2], D = dq[3];
  const int64_t Sk = kc.dimensions()[1], Hk = kc.dimensions()[2];
  if (Hq % Hk != 0)
    return ffi::Error::InvalidArgument("H_q must be a multiple of H_k");

  const int num_splits = num_splits_attr;  // split-KV count (computed in Python heuristic)
  const bool is_bf16 = q.element_type() == ffi::DataType::BF16;
  const mask_info mask = make_mask(is_causal, -1, -1, (int)Sq, (int)Sk);

  // fp32 split workspaces lse_acc (splits,B,Hq,Sq) / o_acc (splits,B,Hq,Sq,D) from XLA's
  // ScratchAllocator (alignment 16 = ROCm FFI max; the returned device ptr is page-aligned).
  const size_t n_lse = (size_t)num_splits * B * Hq * Sq;
  auto lse_acc = scratch.Allocate(n_lse * sizeof(float), 16);
  auto o_acc = scratch.Allocate(n_lse * (size_t)D * sizeof(float), 16);
  if (!lse_acc || !o_acc)
    return ffi::Error::Internal("scratch alloc failed (splitkv workspace)");

  fmha_fwd_splitkv_traits traits;
  traits.hdim_q = (int)D;
  traits.hdim_v = (int)D;
  traits.data_type = is_bf16 ? "bf16" : "fp16";
  traits.is_group_mode = false;
  traits.is_v_rowmajor = true;
  traits.has_logits_soft_cap = false;
  traits.mask_type = mask.type;
  traits.bias_type = bias_enum::no_bias;
  traits.has_lse = false;
  traits.do_fp8_static_quant = false;

  fmha_fwd_splitkv_args a{};
  a.q_ptr = q.untyped_data();
  a.k_ptr = kc.untyped_data();
  a.v_ptr = vc.untyped_data();
  a.bias_ptr = nullptr;
  a.lse_acc_ptr = *lse_acc;
  a.o_acc_ptr = *o_acc;
  a.lse_ptr = nullptr;
  a.o_ptr = out->untyped_data();
  a.block_table_ptr = nullptr;
  a.batch_stride_block_table = 0;
  a.page_block_size = 0;
  a.is_gappy = false;
  a.cache_batch_idx = nullptr;
  a.seqstart_q_ptr = nullptr;
  a.seqstart_k_ptr = nullptr;
  a.seqlen_k_ptr = cache_seqlens.untyped_data();   // per-batch valid cache length (i32)
  a.seqlen_q = (int)Sq;
  a.seqlen_k = (int)Sk;
  a.batch = (int)B;
  a.max_seqlen_q = (int)Sq;
  a.hdim_q = (int)D;
  a.hdim_v = (int)D;
  a.nhead_q = (int)Hq;
  a.nhead_k = (int)Hk;
  a.num_splits = num_splits;
  a.scale_s = (float)scale;
  a.scale_p = 1.f;
  a.scale_o = 1.f;
  a.logits_soft_cap = 0.f;
  a.stride_q = (int)(Hq * D);
  a.stride_k = (int)(Hk * D);
  a.stride_v = (int)(Hk * D);
  a.stride_bias = 0;
  a.stride_o_acc = (int)D;
  a.stride_o = (int)(Hq * D);
  a.nhead_stride_q = (int)D;
  a.nhead_stride_k = (int)D;
  a.nhead_stride_v = (int)D;
  a.nhead_stride_bias = 0;
  a.nhead_stride_lse = 0;
  a.nhead_stride_lse_acc = (int)Sq;
  a.nhead_stride_o_acc = (int)(Sq * D);
  a.nhead_stride_o = (int)D;
  a.batch_stride_q = (int)(Sq * Hq * D);
  a.batch_stride_k = (int)(Sk * Hk * D);
  a.batch_stride_v = (int)(Sk * Hk * D);
  a.batch_stride_bias = 0;
  a.batch_stride_lse = 0;
  a.batch_stride_lse_acc = (int)(Hq * Sq);
  a.batch_stride_o_acc = (int)(Hq * Sq * D);
  a.batch_stride_o = (int)(Sq * Hq * D);
  a.split_stride_lse_acc = (int)(B * Hq * Sq);
  a.split_stride_o_acc = (int)(B * Hq * Sq * D);
  a.window_size_left = mask.left;
  a.window_size_right = mask.right;
  a.mask_type = (int)mask.type;

  ck_tile::stream_config cfg;
  cfg.stream_id_ = stream;
  float t = fmha_fwd_splitkv(traits, a, cfg);
  if (t < 0.f)
    return ffi::Error::Internal(
        "fmha_fwd_splitkv: no CK kernel instance for this (head dim/dtype/mask) config");
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LumiFaKvCache, KvCacheImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::AnyBuffer>()  // q (B,Sq,Hq,D)
        .Arg<ffi::AnyBuffer>()  // k_cache (B,Sk,Hk,D)
        .Arg<ffi::AnyBuffer>()  // v_cache (B,Sk,Hk,D)
        .Arg<ffi::AnyBuffer>()  // cache_seqlens (B,) i32
        .Attr<double>("scale")
        .Attr<int32_t>("is_causal")
        .Attr<int32_t>("num_splits")
        .Ret<ffi::AnyBuffer>()); // out (B,Sq,Hq,D)

// ---------------------------------------------------------------------------
// Full decode step: fmha_fwd_appendkv (append knew/vnew into the cache + fused
// rotary on q/knew, optionally paged via block_table) then fmha_fwd_splitkv on
// the updated cache -- the exact composition of flash_attn's mha_fwd_kvcache,
// in one custom call on one stream.
//
// The append (and q-rotation) is IN-PLACE on q/k_cache/v_cache, so the wrapper
// binds q,kc,vc as aliased outputs (XLA input_output_aliases); the handler
// writes through the input pointers and the rotated q comes back as q_out.
//   contiguous: kc/vc (B,Sk,Hk,D);   paged: kc/vc (num_blocks,pbs,Hk,D),
//   block_table (B,max_blocks) i32, Sk = max_blocks*pbs, pbs % 128 == 0.
// cache_seqlens (B,) i32 = pre-append valid length (where knew lands);
// append_seqlens = cache_seqlens + Sknew computed JAX-side for the attend.
// rotary_cos/sin (seqlen_ro, rotary_dim/2), q dtype, rotary_dim%16==0; unused
// operands are 1-element dummies. No alibi/window (decode path), no LSE.
// ---------------------------------------------------------------------------
static ffi::Error KvCacheStepImpl(hipStream_t stream, ffi::ScratchAllocator scratch,
                                  ffi::AnyBuffer q, ffi::AnyBuffer kc, ffi::AnyBuffer vc,
                                  ffi::AnyBuffer knew, ffi::AnyBuffer vnew,
                                  ffi::AnyBuffer rcos, ffi::AnyBuffer rsin,
                                  ffi::AnyBuffer cache_seqlens, ffi::AnyBuffer append_seqlens,
                                  ffi::AnyBuffer block_table,
                                  double scale, int32_t is_causal, int32_t num_splits,
                                  int32_t rotary_dim, int32_t rotary_interleaved,
                                  int32_t has_knew, int32_t is_paged,
                                  ffi::Result<ffi::AnyBuffer> out,
                                  ffi::Result<ffi::AnyBuffer> q_out,
                                  ffi::Result<ffi::AnyBuffer> kc_out,
                                  ffi::Result<ffi::AnyBuffer> vc_out) {
  auto dq = q.dimensions();
  if (dq.size() != 4 || kc.dimensions().size() != 4)
    return ffi::Error::InvalidArgument("q,k_cache must be rank 4");
  const int64_t B = dq[0], Sq = dq[1], Hq = dq[2], D = dq[3];
  const int64_t Hk = kc.dimensions()[2];
  if (Hq % Hk != 0)
    return ffi::Error::InvalidArgument("H_q must be a multiple of H_k");
  if (q_out->untyped_data() != q.untyped_data() ||
      kc_out->untyped_data() != kc.untyped_data() ||
      vc_out->untyped_data() != vc.untyped_data())
    return ffi::Error::InvalidArgument(
        "kvcache_step: q/k_cache/v_cache outputs must alias inputs (input_output_aliases)");

  const int64_t pbs = is_paged ? kc.dimensions()[1] : 0;                 // page block size
  const int64_t Sk = is_paged ? block_table.dimensions()[1] * pbs : kc.dimensions()[1];
  const int64_t bstride_kv = (is_paged ? pbs : Sk) * Hk * D;             // block stride if paged
  const int64_t Sknew = has_knew ? knew.dimensions()[1] : 0;
  const bool is_bf16 = q.element_type() == ffi::DataType::BF16;

  // mha_fwd_kvcache: single-query no-alibi causal degenerates to no-mask.
  if (Sq == 1) is_causal = 0;
  const mask_info mask = make_mask(is_causal, -1, -1, (int)Sq, (int)Sk);

  ck_tile::stream_config cfg;
  cfg.stream_id_ = stream;

  if (Sknew > 0 || rotary_dim > 0) {
    fmha_fwd_appendkv_traits at;
    at.hdim_q = (int)D;
    at.hdim_v = (int)D;
    at.data_type = is_bf16 ? "bf16" : "fp16";
    at.is_v_rowmajor = true;
    at.rope_type = rotary_dim > 0 ? (rotary_interleaved ? rope_enum::interleaved
                                                        : rope_enum::half_rotated)
                                  : rope_enum::none;

    fmha_fwd_appendkv_args aa{};
    aa.q_ptr = q.untyped_data();
    aa.k_ptr = kc.untyped_data();
    aa.knew_ptr = knew.untyped_data();
    aa.v_ptr = vc.untyped_data();
    aa.vnew_ptr = vnew.untyped_data();
    aa.seqlen_k_ptr = cache_seqlens.untyped_data();  // append lands at the pre-append length
    aa.seqlen_q = (int)Sq;
    aa.seqlen_knew = (int)Sknew;
    aa.batch = (int)B;
    aa.hdim_q = (int)D;
    aa.hdim_v = (int)D;
    aa.nhead_q = (int)Hq;
    aa.nhead_k = (int)Hk;
    aa.rotary_cos_ptr = rotary_dim > 0 ? rcos.untyped_data() : nullptr;
    aa.rotary_sin_ptr = rotary_dim > 0 ? rsin.untyped_data() : nullptr;
    aa.rotary_dim = rotary_dim;
    aa.has_mask = mask.type != mask_enum::no_mask;
    aa.block_table_ptr = is_paged ? block_table.untyped_data() : nullptr;
    aa.batch_stride_block_table = is_paged ? (int)block_table.dimensions()[1] : 0;
    aa.page_block_size = (int)pbs;
    aa.cache_batch_idx = nullptr;
    aa.stride_q = (int)(Hq * D);
    aa.stride_k = (int)(Hk * D);
    aa.stride_knew = (int)(Hk * D);
    aa.stride_v = (int)(Hk * D);
    aa.stride_vnew = (int)(Hk * D);
    aa.nhead_stride_q = (int)D;
    aa.nhead_stride_k = (int)D;
    aa.nhead_stride_knew = (int)D;
    aa.nhead_stride_v = (int)D;
    aa.nhead_stride_vnew = (int)D;
    aa.batch_stride_q = (int)(Sq * Hq * D);
    aa.batch_stride_k = (int)bstride_kv;
    aa.batch_stride_knew = (int)(Sknew * Hk * D);
    aa.batch_stride_v = (int)bstride_kv;
    aa.batch_stride_vnew = (int)(Sknew * Hk * D);

    float t = fmha_fwd_appendkv(at, aa, cfg);
    if (t < 0.f)
      return ffi::Error::Internal(
          "fmha_fwd_appendkv: no CK kernel instance for this (head dim/dtype/rope) config");
  }

  const size_t n_lse = (size_t)num_splits * B * Hq * Sq;
  auto lse_acc = scratch.Allocate(n_lse * sizeof(float), 16);
  auto o_acc = scratch.Allocate(n_lse * (size_t)D * sizeof(float), 16);
  if (!lse_acc || !o_acc)
    return ffi::Error::Internal("scratch alloc failed (splitkv workspace)");

  fmha_fwd_splitkv_traits traits;
  traits.hdim_q = (int)D;
  traits.hdim_v = (int)D;
  traits.data_type = is_bf16 ? "bf16" : "fp16";
  traits.is_group_mode = false;
  traits.is_v_rowmajor = true;
  traits.has_logits_soft_cap = false;
  traits.mask_type = mask.type;
  traits.bias_type = bias_enum::no_bias;
  traits.has_lse = false;
  traits.do_fp8_static_quant = false;

  fmha_fwd_splitkv_args a{};
  a.q_ptr = q.untyped_data();
  a.k_ptr = kc.untyped_data();
  a.v_ptr = vc.untyped_data();
  a.bias_ptr = nullptr;
  a.lse_acc_ptr = *lse_acc;
  a.o_acc_ptr = *o_acc;
  a.lse_ptr = nullptr;
  a.o_ptr = out->untyped_data();
  a.block_table_ptr = is_paged ? block_table.untyped_data() : nullptr;
  a.batch_stride_block_table = is_paged ? (int)block_table.dimensions()[1] : 0;
  a.page_block_size = (int)pbs;
  a.is_gappy = false;
  a.cache_batch_idx = nullptr;
  a.seqstart_q_ptr = nullptr;
  a.seqstart_k_ptr = nullptr;
  a.seqlen_k_ptr = append_seqlens.untyped_data();  // post-append valid length (i32)
  a.seqlen_q = (int)Sq;
  a.seqlen_k = (int)Sk;
  a.batch = (int)B;
  a.max_seqlen_q = (int)Sq;
  a.hdim_q = (int)D;
  a.hdim_v = (int)D;
  a.nhead_q = (int)Hq;
  a.nhead_k = (int)Hk;
  a.num_splits = num_splits;
  a.scale_s = (float)scale;
  a.scale_p = 1.f;
  a.scale_o = 1.f;
  a.logits_soft_cap = 0.f;
  a.stride_q = (int)(Hq * D);
  a.stride_k = (int)(Hk * D);
  a.stride_v = (int)(Hk * D);
  a.stride_bias = 0;
  a.stride_o_acc = (int)D;
  a.stride_o = (int)(Hq * D);
  a.nhead_stride_q = (int)D;
  a.nhead_stride_k = (int)D;
  a.nhead_stride_v = (int)D;
  a.nhead_stride_bias = 0;
  a.nhead_stride_lse = 0;
  a.nhead_stride_lse_acc = (int)Sq;
  a.nhead_stride_o_acc = (int)(Sq * D);
  a.nhead_stride_o = (int)D;
  a.batch_stride_q = (int)(Sq * Hq * D);
  a.batch_stride_k = (int)bstride_kv;
  a.batch_stride_v = (int)bstride_kv;
  a.batch_stride_bias = 0;
  a.batch_stride_lse = 0;
  a.batch_stride_lse_acc = (int)(Hq * Sq);
  a.batch_stride_o_acc = (int)(Hq * Sq * D);
  a.batch_stride_o = (int)(Sq * Hq * D);
  a.split_stride_lse_acc = (int)(B * Hq * Sq);
  a.split_stride_o_acc = (int)(B * Hq * Sq * D);
  a.window_size_left = mask.left;
  a.window_size_right = mask.right;
  a.mask_type = (int)mask.type;

  float t = fmha_fwd_splitkv(traits, a, cfg);
  if (t < 0.f)
    return ffi::Error::Internal(
        "fmha_fwd_splitkv: no CK kernel instance for this (head dim/dtype/mask) config");
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LumiFaKvCacheStep, KvCacheStepImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::AnyBuffer>()  // q (B,Sq,Hq,D)
        .Arg<ffi::AnyBuffer>()  // k_cache (B,Sk,Hk,D) | (nblocks,pbs,Hk,D)
        .Arg<ffi::AnyBuffer>()  // v_cache
        .Arg<ffi::AnyBuffer>()  // knew (B,Sknew,Hk,D) | dummy
        .Arg<ffi::AnyBuffer>()  // vnew
        .Arg<ffi::AnyBuffer>()  // rotary_cos (seqlen_ro, rdim/2) | dummy
        .Arg<ffi::AnyBuffer>()  // rotary_sin
        .Arg<ffi::AnyBuffer>()  // cache_seqlens (B,) i32 (pre-append)
        .Arg<ffi::AnyBuffer>()  // append_seqlens (B,) i32 (post-append)
        .Arg<ffi::AnyBuffer>()  // block_table (B,max_blocks) i32 | dummy
        .Attr<double>("scale")
        .Attr<int32_t>("is_causal")
        .Attr<int32_t>("num_splits")
        .Attr<int32_t>("rotary_dim")
        .Attr<int32_t>("rotary_interleaved")
        .Attr<int32_t>("has_knew")
        .Attr<int32_t>("is_paged")
        .Ret<ffi::AnyBuffer>()   // out (B,Sq,Hq,D)
        .Ret<ffi::AnyBuffer>()   // q_out  (aliased q)
        .Ret<ffi::AnyBuffer>()   // kc_out (aliased k_cache)
        .Ret<ffi::AnyBuffer>()); // vc_out (aliased v_cache)
