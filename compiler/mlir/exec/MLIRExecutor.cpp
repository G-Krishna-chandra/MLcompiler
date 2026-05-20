#include "compiler/mlir/exec/MLIRExecutor.h"
#include "compiler/mlir/exec/MLXBuilder.h"
#include "compiler/mlir/exec/MLXQuantize.h"
#include "compiler/mlir/exec/Quantize.h"
#include "compiler/frontends/ggml_types.hpp"
#include "compiler/mlir/dialect/MLCDialect.h"
#include "compiler/mlir/dialect/MLCOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"

#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include <cmath>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mx = mlx::core;

namespace mlir::mlc::exec {

namespace {

// ----- Per-op MLX builders -----
// Kept as free functions inside this TU so the walker switch below stays
// short and readable. Each builder takes already-materialized mx::array
// operands and returns the result of the op as a (lazy) mx::array.

// mx::quantized_matmul always returns float32 regardless of x's dtype.
// We leave it in fp32; all activations flow in fp32 downstream. The custom
// kernel had fp32 accumulation + fp16 output; keeping fp32 throughout here
// gives better precision and matches the expected greedy-decode trajectory.
inline mx::array qmatmul(const mx::array &x, const MLXQuantWeights &pkg) {
  return mx::quantized_matmul(x, pkg.w_q, pkg.scales, pkg.biases,
                              /*transpose=*/true, /*group_size=*/32,
                              /*bits=*/4, "affine");
}

mx::array buildNorm(const mx::array &x, const mx::array &gamma, float eps) {
  // RMSNorm with fused gamma. MLX's fast::rms_norm matches what TinyLlama
  // and Llama-style models use.
  return mx::fast::rms_norm(x, gamma, eps);
}

mx::array buildAdd(const mx::array &a, const mx::array &b) {
  return mx::add(a, b);
}

mx::array buildMatMul(const mx::array &x, const mx::array &w,
                      bool transpose_b) {
  return mx::matmul(x, transpose_b ? mx::transpose(w) : w);
}

// SwiGLU FFN: down(silu(gate(x)) * up(x)).
//
// Gate and up share x, so we concat them along the out-dim once (cached by
// the caller, identified by the FeedForward Operation*) and do a single
// batched matmul, splitting the result. Down is its own matmul. All three
// weights are stored [out, in] in numpy row-major, hence the transpose.
mx::array buildFeedForward(const mx::array &x, const mx::array &gate_up_concat,
                           const mx::array &w_down, int ffn_dim) {
  auto gu = mx::matmul(x, mx::transpose(gate_up_concat));
  // gu shape: [seq, 2*ffn_dim]. Split along last axis.
  auto g = mx::slice(gu, {0, 0}, {gu.shape(0), ffn_dim});
  auto u = mx::slice(gu, {0, ffn_dim}, {gu.shape(0), 2 * ffn_dim});
  // silu(x) = x * sigmoid(x).
  auto h = mx::multiply(mx::multiply(g, mx::sigmoid(g)), u);
  return mx::matmul(h, mx::transpose(w_down));
}

// Embedding lookup. TinyLlama stores the table as [hidden, vocab] (column =
// one token's embedding); Llama 2/3 store it [vocab, hidden] (row = one
// token's embedding). Detect by which dim matches the activation hidden
// dim and gather accordingly.
mx::array buildEmbedding(const mx::array &table, const mx::array &ids,
                         int hidden) {
  if (table.shape(0) == hidden) {
    // [hidden, vocab] -> transpose, then take along vocab axis = 0.
    auto t = mx::transpose(table);
    return mx::take(t, ids, /*axis=*/0);
  }
  // [vocab, hidden]
  return mx::take(table, ids, /*axis=*/0);
}

// LM head: x @ w (with auto transpose like embedding).
mx::array buildLMHead(const mx::array &x, const mx::array &w) {
  int hidden = x.shape(static_cast<int>(x.ndim()) - 1);
  if (w.shape(0) == hidden)
    return mx::matmul(x, w);  // [hidden, vocab]
  return mx::matmul(x, mx::transpose(w));  // [vocab, hidden]
}

// Forward declaration (implemented below, after buildAttention).
mx::array buildAttention(const mx::array &q_flat, const mx::array &k_flat,
                         const mx::array &v_flat, const mx::array &positions,
                         int n_heads, int n_kv_heads, int head_dim,
                         KVCacheSlot *cache);

// Batched attention for N concurrent requests, all at the same decode position.
// Implemented as N separate buildAttention calls to reuse the proven
// single-stream path. Outputs are concatenated into [N, n_heads * head_dim].
//
// V2 can replace this with a true batched SDPA once the GQA broadcasting
// semantics in MLX are verified for N>1.
mx::array buildBatchedAttention(const mx::array &q_flat,
                                const mx::array &k_flat,
                                const mx::array &v_flat,
                                int common_offset, int seq_new,
                                int n_heads, int n_kv_heads, int head_dim,
                                std::vector<KVCacheSlot *> &caches) {
  int N = q_flat.shape(0);

  // Build a scalar positions array with value = common_offset.
  // buildAttention reads positions[0] to get the rope offset.
  int32_t *pbuf = static_cast<int32_t *>(std::malloc(sizeof(int32_t)));
  if (!pbuf) throw std::bad_alloc();
  *pbuf = common_offset;
  mx::array pos_scalar(static_cast<void *>(pbuf), mx::Shape{1}, mx::int32,
                       [](void *p) { std::free(p); });

  // Per-request q/k/v: slice row i from the [N, ...] input.
  std::vector<mx::array> outs;
  outs.reserve(N);
  int q_row = n_heads    * head_dim;
  int k_row = n_kv_heads * head_dim;

  for (int i = 0; i < N; ++i) {
    // Slice out a [1, dim] row for request i.
    auto qi = mx::slice(q_flat, {i, 0}, {i+1, q_row});  // [1, q_dim]
    auto ki = mx::slice(k_flat, {i, 0}, {i+1, k_row});  // [1, k_dim]
    auto vi = mx::slice(v_flat, {i, 0}, {i+1, k_row});  // [1, v_dim]
    outs.push_back(buildAttention(qi, ki, vi, pos_scalar,
                                  n_heads, n_kv_heads, head_dim,
                                  caches[i]));
  }

  // Concatenate N outputs: [1, H*D] × N → [N, H*D]
  return mx::concatenate(outs, 0);
}

// Dense attention with causal mask, inline RoPE, and an optional KV cache.
//
// q: [seq_new, n_heads * head_dim]
// k, v: [seq_new, n_kv_heads * head_dim]
// positions: [seq_new] int32. positions[0] is the absolute index of the
// first new token; the rest are assumed contiguous (this matches both
// prefill-from-zero and one-token-decode workloads).
// cache: when non-null, the slot's K and V arrays hold the prefix
// (post-RoPE for K, raw for V); we append new K/V and write the full
// arrays back. nullptr → recompute from scratch each call.
mx::array buildAttention(const mx::array &q_flat, const mx::array &k_flat,
                         const mx::array &v_flat, const mx::array &positions,
                         int n_heads, int n_kv_heads, int head_dim,
                         KVCacheSlot *cache) {
  int seq_new = q_flat.shape(0);

  // Reshape to [seq, head, dim] then transpose to [head, seq, dim].
  auto q = mx::transpose(mx::reshape(q_flat, {seq_new, n_heads, head_dim}),
                         {1, 0, 2});
  auto k = mx::transpose(mx::reshape(k_flat, {seq_new, n_kv_heads, head_dim}),
                         {1, 0, 2});
  auto v = mx::transpose(mx::reshape(v_flat, {seq_new, n_kv_heads, head_dim}),
                         {1, 0, 2});

  // RoPE — start offset from positions[0]. We pull it eagerly (sync) since
  // MLX's rope op takes an int, not an array. v1 only supports
  // contiguous-from-offset sequences, which is what prefill + greedy decode
  // produce anyway.
  mx::array p0 = mx::take(positions, 0);
  mx::eval(p0);
  int offset = static_cast<int>(p0.item<int32_t>());
  // GGUF-format Llama uses consecutive-pair RoPE (rotates dims [2i, 2i+1])
  // — MLX calls this `traditional=true`. Confirmed against the runtime's
  // applyRotaryToBuffer in attention_cpu.cpp.
  //
  // MLX's `fast::rope` returns NaN at seq_len=1 in fp16 (verified empirically
  // on M-series during R2). Working around it by running the rope, the cache
  // store, and the QK^T matmul in fp32. The downstream attention math (V,
  // softmax, etc.) returns to whatever dtype `v` already has.
  auto q_dtype = q.dtype();
  q = mx::astype(q, mx::float32);
  k = mx::astype(k, mx::float32);
  q = mx::fast::rope(q, head_dim, /*traditional=*/true, /*base=*/10000.0f,
                     /*scale=*/1.0f, offset);
  k = mx::fast::rope(k, head_dim, true, 10000.0f, 1.0f, offset);

  // KV cache: pre-allocate K and V buffers at max-seq capacity on first
  // use, then patch new K/V slots in via `slice_update`. Attention reads
  // back the contiguous prefix. The old code rebuilt a new concat'd array
  // every step (O(seq_so_far) copy per token); this is O(seq_new).
  //
  // 2048 covers TinyLlama's default context; bump if we run longer
  // prompts/decodes.
  constexpr int kMaxSeq = 2048;
  int seq_full;
  if (cache) {
    if (!cache->K.has_value()) {
      auto K0 =
          mx::zeros({n_kv_heads, kMaxSeq, head_dim}, k.dtype());
      auto V0 =
          mx::zeros({n_kv_heads, kMaxSeq, head_dim}, v.dtype());
      mx::eval({K0, V0});
      cache->K = K0;
      cache->V = V0;
      cache->seq_filled = 0;
    }
    int start = cache->seq_filled;
    int stop = start + seq_new;
    if (stop > kMaxSeq)
      throw std::runtime_error("KV cache overflow (raise kMaxSeq)");
    auto K_new = mx::slice_update(*cache->K, k, {0, start, 0},
                                   {n_kv_heads, stop, head_dim});
    auto V_new = mx::slice_update(*cache->V, v, {0, start, 0},
                                   {n_kv_heads, stop, head_dim});
    mx::eval({K_new, V_new});
    cache->K = K_new;
    cache->V = V_new;
    cache->seq_filled = stop;
    // Attention reads the contiguous prefix only.
    k = mx::slice(*cache->K, {0, 0, 0}, {n_kv_heads, stop, head_dim});
    v = mx::slice(*cache->V, {0, 0, 0}, {n_kv_heads, stop, head_dim});
    seq_full = stop;
  } else {
    seq_full = k.shape(1);
  }
  // Keep Q in fp32 too so the QK^T matmul is consistent with the fp32 K.
  // V stays in fp16 — the matmul to V happens after softmax, where fp32
  // probabilities multiplying fp16 values is well-conditioned.

  // GQA: replicate K, V across each group of n_heads/n_kv_heads.
  if (n_heads != n_kv_heads) {
    int n_rep = n_heads / n_kv_heads;
    auto k_exp = mx::expand_dims(k, 1);  // [n_kv, 1, seq_full, dim]
    auto v_exp = mx::expand_dims(v, 1);
    k_exp = mx::broadcast_to(k_exp, mx::Shape{n_kv_heads, n_rep, seq_full, head_dim});
    v_exp = mx::broadcast_to(v_exp, mx::Shape{n_kv_heads, n_rep, seq_full, head_dim});
    k = mx::reshape(k_exp, {n_heads, seq_full, head_dim});
    v = mx::reshape(v_exp, {n_heads, seq_full, head_dim});
  }

  // MLX's `scaled_dot_product_attention` does the softmax(QK^T/√d)V dance for
  // us, with a built-in causal mask path. Inputs are [B, H, L, D]; we have a
  // single batch so add a leading dim and drop it on the way out.
  auto q4 = mx::expand_dims(q, 0);                  // [1, H, seq_new, D]
  auto k4 = mx::expand_dims(k, 0);                  // [1, H, seq_full, D]
  auto v4 = mx::expand_dims(v, 0);                  // [1, H, seq_full, D]
  // Cast V up to match Q/K (both fp32 by now) — the kernel asks for matching dtypes.
  v4 = mx::astype(v4, q4.dtype());
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  // Causal mask only when there's history we need to hide (prefill).
  // For one-token decode every cached key is in the past, so no mask.
  std::string mask_mode = (seq_new > 1) ? "causal" : "";
  auto out4 = mx::fast::scaled_dot_product_attention(q4, k4, v4, scale,
                                                     mask_mode);
  auto out = mx::squeeze(out4, 0);                  // [H, seq_new, D]
  out = mx::transpose(out, {1, 0, 2});              // [seq_new, H, D]
  out = mx::reshape(out, {seq_new, n_heads * head_dim});
  return mx::astype(out, q_dtype);
}

// ----- Arg binding -----

const std::string &argName(::mlir::func::FuncOp fn, unsigned idx) {
  static thread_local std::string holder;
  auto attr = fn.getArgAttrOfType<::mlir::StringAttr>(idx, "mlc.name");
  if (!attr)
    throw std::runtime_error("func arg missing mlc.name attribute");
  holder = attr.str();
  return holder;
}

float f32Of(::mlir::FloatAttr a) { return a.getValueAsDouble(); }
int64_t i64Of(::mlir::IntegerAttr a) { return a.getInt(); }

} // namespace

MLIRExecutor::MLIRExecutor(::mlir::ModuleOp module,
                           const ::mlc::frontend::GGUFLoader &loader,
                           bool use_ane)
    : module_(module), loader_(loader), use_ane_(use_ane) {
  {
    const char *env = std::getenv("MLC_Q4_CUSTOM_KERNEL");
    use_custom_q4_ = env && std::string(env) == "1";
  }
  {
    const char *env = std::getenv("MLC_COMPILER_PROFILE");
    profile_enabled_ = env && std::string(env) == "1";
  }
  auto func = *module_.getOps<::mlir::func::FuncOp>().begin();

  // First pass: identify which func args are used as the weight operand of
  // a matmul-style op (mlc.matmul, fused_norm_matmul, fused_norm_qkv_matmul,
  // feedforward). Those are the candidates for the Q4_0 byte-resident path.
  ::std::unordered_set<unsigned> matmul_arg_idxs;
  auto markArg = [&](::mlir::Value v) {
    if (auto barg = ::llvm::dyn_cast<::mlir::BlockArgument>(v))
      matmul_arg_idxs.insert(barg.getArgNumber());
  };
  func.walk([&](::mlir::Operation *op) {
    if (auto mm = ::llvm::dyn_cast<::mlir::mlc::MatMulOp>(op)) markArg(mm.getW());
    else if (auto fnm = ::llvm::dyn_cast<::mlir::mlc::FusedNormMatMulOp>(op)) markArg(fnm.getW());
    else if (auto qkv = ::llvm::dyn_cast<::mlir::mlc::FusedNormQKVMatMulOp>(op)) {
      markArg(qkv.getWQ()); markArg(qkv.getWK()); markArg(qkv.getWV());
    } else if (auto ff = ::llvm::dyn_cast<::mlir::mlc::FeedForwardOp>(op)) {
      markArg(ff.getWGate()); markArg(ff.getWUp()); markArg(ff.getWDown());
    }
  });

  // Second pass: per-arg, decide between MLX native-quantized (Q4_0 matmul
  // weights → mx::quantized_matmul) and fp16 dequant (embedding table, norm
  // gamma, Q6_K lm_head, and anything non-Q4_0).
  auto &entry = func.getBody().front();
  for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
    const std::string &nm = argName(func, i);
    if (nm == "ids" || nm == "positions") continue;
    const auto &info = loader_.tensors().at(nm);
    bool is_q4_target = matmul_arg_idxs.count(i) &&
                        info.dtype == ::mlc::frontend::GGML_TYPE_Q4_0 &&
                        info.shape.size() == 2;
    if (is_q4_target && use_custom_q4_) {
      // Legacy path: keep raw GGUF Q4_0 bytes for the custom Metal kernel.
      auto bytes = gufWeightToBytesMLX(loader_, nm);
      q4_bytes_cache_.emplace(nm, bytes);
      int in_dim = static_cast<int>(info.shape[0]);
      int out_dim = static_cast<int>(info.shape[1]);
      q4_dims_cache_.emplace(nm, std::pair<int, int>{in_dim, out_dim});
      mx::eval(bytes);
    } else if (is_q4_target) {
      // ggufQ4_0ToMLXQuantized builds and evaluates (w_q, scales, biases)
      // from GGUF bytes in one shot. The conversion is bit-exact: no
      // re-quantization, no fp16 dequant of the full weight matrix.
      mlxq_cache_.emplace(nm, ggufQ4_0ToMLXQuantized(loader_, nm));
    } else {
      auto a32 = gufWeightToMLX(loader_, nm);
      auto a16 = mx::astype(a32, mx::float16);
      auto [it, inserted] = weight_cache_.emplace(nm, a16);
      if (inserted) mx::eval(a16);
    }
  }

  // Third pass: if ANE is enabled, bake a CoreML model for every op the
  // scheduler marked Device::ANE. The models hold their weights baked in
  // as fp16 constants, so we can drop the corresponding Q4_0 byte cache
  // entries after baking — but only if no GPU op references the same
  // weight, which on TinyLlama doesn't happen for ANE-eligible ops.
  if (use_ane_) {
    std::string ane_dir = "/tmp/mlc-ane-cache";
    auto deviceIsANE = [](Operation *op) {
      auto attr = op->getAttrOfType<DeviceAttr>("target_device");
      return attr && attr.getValue() == Device::ANE;
    };
    auto bytesToFP16MLX = [&](Value w_val) -> mx::array {
      // Dequant to fp32 + cast to fp16 for the CoreML model bake.
      auto barg = ::llvm::cast<BlockArgument>(w_val);
      const std::string &nm = argName(func, barg.getArgNumber());
      auto f32 = dequantizeToF32(loader_, nm);
      // GGUF row-major: [out, in]. CoreML's matmul wants weight [K=in, N=out],
      // so transpose during the upload.
      const auto &info = loader_.tensors().at(nm);
      int OUT = static_cast<int>(info.shape[1]);
      int IN = static_cast<int>(info.shape[0]);
      std::vector<float> w_kn(static_cast<size_t>(IN) * OUT);
      for (int o = 0; o < OUT; ++o)
        for (int k = 0; k < IN; ++k)
          w_kn[static_cast<size_t>(k) * OUT + o] =
              f32[static_cast<size_t>(o) * IN + k];
      auto w_mlx = fp32ToMLX(w_kn, {IN, OUT});
      auto w_fp16 = mx::astype(w_mlx, mx::float16);
      mx::eval(w_fp16);
      return w_fp16;
    };
    auto getWeightName = [&](Value w_val) {
      auto barg = ::llvm::cast<BlockArgument>(w_val);
      return argName(func, barg.getArgNumber());
    };

    // Optional: skip the QKV-fused ANE path by setting MLC_ANE_QKV=0 (env).
    // Lets us isolate the contribution of just attn_output during benching.
    bool ane_qkv = []() {
      const char *s = std::getenv("MLC_ANE_QKV");
      return !s || std::string(s) != "0";
    }();

    func.walk([&](Operation *op) {
      if (!deviceIsANE(op)) return;
      if (auto mm = ::llvm::dyn_cast<MatMulOp>(op)) {
        std::string key = getWeightName(mm.getW());
        auto w_fp16 = bytesToFP16MLX(mm.getW());
        const auto &info = loader_.tensors().at(key);
        int IN = static_cast<int>(info.shape[0]);
        int OUT = static_cast<int>(info.shape[1]);
        auto pkg = buildANEMatMulPackage(ane_dir, key, /*M=*/1, IN, OUT, w_fp16);
        ane_cache_.emplace(op, std::make_shared<ANEMatMul>(pkg, 1, IN, OUT));
      } else if (auto qkv = ::llvm::dyn_cast<FusedNormQKVMatMulOp>(op);
                 qkv && ane_qkv) {
        // Bake Q ‖ K ‖ V as a single [IN, OUT_total] fp16 weight.
        auto wq_name = getWeightName(qkv.getWQ());
        auto wk_name = getWeightName(qkv.getWK());
        auto wv_name = getWeightName(qkv.getWV());
        const auto &iq = loader_.tensors().at(wq_name);
        const auto &ik = loader_.tensors().at(wk_name);
        const auto &iv = loader_.tensors().at(wv_name);
        int IN = static_cast<int>(iq.shape[0]);
        int Oq = static_cast<int>(iq.shape[1]);
        int Ok = static_cast<int>(ik.shape[1]);
        int Ov = static_cast<int>(iv.shape[1]);
        int OUT = Oq + Ok + Ov;
        auto wq = bytesToFP16MLX(qkv.getWQ());
        auto wk = bytesToFP16MLX(qkv.getWK());
        auto wv = bytesToFP16MLX(qkv.getWV());
        auto wcat = mx::concatenate({wq, wk, wv}, /*axis=*/1);
        mx::eval(wcat);
        std::string key = wq_name + "+kv";
        auto pkg = buildANEMatMulPackage(ane_dir, key, 1, IN, OUT, wcat);
        ane_cache_.emplace(op, std::make_shared<ANEMatMul>(pkg, 1, IN, OUT));
      }
    });
  }
}

void MLIRExecutor::reset() {
  kv_cache_.clear();
  batch_kv_slots_.clear();
}

void MLIRExecutor::resetSlot(int slot_idx) {
  if (slot_idx == 0 && batch_kv_slots_.empty()) {
    kv_cache_.clear();
  } else if (slot_idx >= 0 &&
             static_cast<size_t>(slot_idx) < batch_kv_slots_.size()) {
    batch_kv_slots_[slot_idx].clear();
  }
}

// Prefill N requests in one batched call.
// For V1: all prompts must be the same length (equal-length constraint).
// Strategy: run ONE single-stream prefill (reusing existing run()), then
// replicate the resulting KV cache N times into batch_kv_slots_.
// This avoids N full forward passes for an identical prompt.
std::vector<MLIRExecutor::RunResult>
MLIRExecutor::prefillBatch(
    const std::vector<std::vector<int32_t>> &prompt_ids) {
  if (prompt_ids.empty())
    throw std::runtime_error("prefillBatch: empty request list");
  int N = static_cast<int>(prompt_ids.size());
  // V1: require all prompts the same length.
  size_t L = prompt_ids[0].size();
  for (int i = 1; i < N; ++i)
    if (prompt_ids[i].size() != L)
      throw std::runtime_error("prefillBatch: prompts must have equal length in V1");

  // Run one prefill using the single-stream path (populates kv_cache_).
  reset();
  std::vector<int32_t> pos(L);
  std::iota(pos.begin(), pos.end(), 0);
  auto single_out = run(prompt_ids[0], pos);

  // Replicate the single-stream KV cache N times.
  batch_kv_slots_.clear();
  batch_kv_slots_.resize(N);
  for (int i = 0; i < N; ++i) {
    batch_kv_slots_[i].resize(kv_cache_.size());
    for (size_t layer = 0; layer < kv_cache_.size(); ++layer) {
      auto &src = kv_cache_[layer];
      auto &dst = batch_kv_slots_[i][layer];
      // Deep-copy the KV arrays so each request owns independent storage.
      if (src.K.has_value()) {
        dst.K = mx::copy(*src.K);
        dst.V = mx::copy(*src.V);
        mx::eval({*dst.K, *dst.V});
      }
      dst.seq_filled = src.seq_filled;
    }
  }

  // Return N identical logit results (all from the same prefill).
  std::vector<RunResult> out(N, single_out);
  return out;
}

// Batched decode: one token per request, all at the same absolute position.
// batch_kv_slots_ must have been populated by prefillBatch first.
//
// token_ids[i] = next token for request i
// positions[i] = absolute position (must all be equal in V1; first is used)
//
// Returns N results, each shape [1, vocab_size].
std::vector<MLIRExecutor::RunResult>
MLIRExecutor::runBatch(const std::vector<int32_t> &token_ids,
                       const std::vector<int32_t> &positions) {
  int N = static_cast<int>(token_ids.size());
  if (N == 0) throw std::runtime_error("runBatch: empty token list");
  if (positions.size() != static_cast<size_t>(N))
    throw std::runtime_error("runBatch: token_ids and positions size mismatch");
  if (batch_kv_slots_.size() != static_cast<size_t>(N))
    throw std::runtime_error("runBatch: must call prefillBatch first with same N");

  auto func = *module_.getOps<::mlir::func::FuncOp>().begin();
  auto &entryBlock = func.getBody().front();

  // Build integer arrays for IDs and positions.
  auto makeI32 = [](int32_t v) {
    int32_t *buf = static_cast<int32_t *>(std::malloc(sizeof(int32_t)));
    if (!buf) throw std::bad_alloc();
    *buf = v;
    mx::Shape s{1};
    return mx::array(static_cast<void *>(buf), std::move(s), mx::int32,
                     [](void *p) { std::free(p); });
  };

  // Stack N token embeddings into one [N, 1] ids array.
  // The embedding table is looked up per-request: embed each separately,
  // then concatenate → [N, hidden_dim].
  // positions: use positions[0] as the common offset (V1 equal-pos assumption).
  int common_offset = positions[0];

  // We need to build [N, hidden] activations. The cleanest approach:
  // treat the N tokens as a mini-batch. Build a single [N] IDs tensor,
  // look up the embedding table, get [N, hidden].
  // Then run the same IR walker with [N, hidden] tensors.
  // At attention ops: dispatch to buildBatchedAttention with batch_kv_slots_.

  // Build [N] integer token array.
  int32_t *id_buf = static_cast<int32_t *>(std::malloc(N * sizeof(int32_t)));
  if (!id_buf) throw std::bad_alloc();
  for (int i = 0; i < N; ++i) id_buf[i] = token_ids[i];
  mx::array ids_arr(static_cast<void *>(id_buf), mx::Shape{N}, mx::int32,
                    [](void *p) { std::free(p); });
  // positions array (all same value in V1)
  auto pos_arr = makeI32(common_offset);

  // Bind func args.
  ::llvm::DenseMap<::mlir::Value, mx::array> values;

  mlxq_value_pkg_.clear();
  q4_value_dims_.clear();
  for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
    const std::string &nm = argName(func, i);
    auto value = entryBlock.getArgument(i);
    if (nm == "ids") {
      values.insert({value, ids_arr});
    } else if (nm == "positions") {
      values.insert({value, pos_arr});
    } else if (use_custom_q4_) {
      if (auto it = q4_bytes_cache_.find(nm); it != q4_bytes_cache_.end()) {
        values.insert({value, it->second});
        q4_value_dims_.insert({value, q4_dims_cache_.at(nm)});
      } else {
        auto it2 = weight_cache_.find(nm);
        if (it2 == weight_cache_.end())
          throw std::runtime_error("runBatch: missing weight: " + nm);
        values.insert({value, it2->second});
      }
    } else if (auto q4it = mlxq_cache_.find(nm); q4it != mlxq_cache_.end()) {
      values.insert({value, q4it->second.w_q});
      mlxq_value_pkg_.insert({value, &q4it->second});
    } else {
      auto it = weight_cache_.find(nm);
      if (it == weight_cache_.end())
        throw std::runtime_error("runBatch: missing weight: " + nm);
      values.insert({value, it->second});
    }
  }

  auto get = [&](::mlir::Value v) -> mx::array & {
    auto it = values.find(v);
    if (it == values.end()) throw std::runtime_error("SSA value not bound (batch)");
    return it->second;
  };
  auto put = [&](::mlir::Value v, mx::array a) {
    values.insert_or_assign(v, std::move(a));
  };

  // Build per-request KV cache pointer lists (one pointer per attention layer,
  // one list per attention op encounter). We'll grow this as we walk the IR.
  std::vector<std::vector<KVCacheSlot *>> batch_attn_caches;
  size_t n_attn_layers = batch_kv_slots_[0].size();
  batch_attn_caches.reserve(n_attn_layers);
  for (size_t layer = 0; layer < n_attn_layers; ++layer) {
    std::vector<KVCacheSlot *> ptrs(N);
    for (int i = 0; i < N; ++i)
      ptrs[i] = &batch_kv_slots_[i][layer];
    batch_attn_caches.push_back(std::move(ptrs));
  }

  ::mlir::Value result;
  size_t attn_idx = 0;
  for (auto &op : entryBlock) {
    if (auto matmul = ::llvm::dyn_cast<::mlir::mlc::MatMulOp>(op)) {
      auto w_v = matmul.getW();
      auto x_in = get(matmul.getX());
      if (use_custom_q4_) {
        if (auto dims = q4_value_dims_.find(w_v); dims != q4_value_dims_.end())
          put(matmul.getY(), q4_0_matmul(x_in, get(w_v),
                                         dims->second.first, dims->second.second));
        else
          put(matmul.getY(), buildMatMul(x_in, get(w_v), matmul.getTransposeB()));
      } else if (auto pkg_it = mlxq_value_pkg_.find(w_v);
                 pkg_it != mlxq_value_pkg_.end()) {
        put(matmul.getY(), qmatmul(x_in, *pkg_it->second));
      } else {
        put(matmul.getY(), buildMatMul(x_in, get(w_v), matmul.getTransposeB()));
      }
    } else if (auto norm = ::llvm::dyn_cast<::mlir::mlc::NormOp>(op)) {
      put(norm.getY(), buildNorm(get(norm.getX()), get(norm.getGamma()),
                                 f32Of(norm.getEpsilonAttr())));
    } else if (auto add = ::llvm::dyn_cast<::mlir::mlc::AddOp>(op)) {
      put(add.getY(), buildAdd(get(add.getA()), get(add.getB())));
    } else if (auto ff = ::llvm::dyn_cast<::mlir::mlc::FeedForwardOp>(op)) {
      auto w_gate_ty =
          ::llvm::cast<::mlir::RankedTensorType>(ff.getWGate().getType());
      int ffn_dim = w_gate_ty.getShape()[0];
      auto gate_pkg = mlxq_value_pkg_.find(ff.getWGate());
      auto up_pkg   = mlxq_value_pkg_.find(ff.getWUp());
      auto down_pkg = mlxq_value_pkg_.find(ff.getWDown());
      bool all_q4 = gate_pkg != mlxq_value_pkg_.end() &&
                    up_pkg   != mlxq_value_pkg_.end() &&
                    down_pkg != mlxq_value_pkg_.end();
      if (all_q4) {
        auto gu_it = mlxq_ffngu_cache_.find(ff.getOperation());
        if (gu_it == mlxq_ffngu_cache_.end()) {
          const auto &gp = *gate_pkg->second;
          const auto &up = *up_pkg->second;
          auto w_q_cat    = mx::concatenate({gp.w_q, up.w_q}, 0);
          auto scales_cat = mx::concatenate({gp.scales, up.scales}, 0);
          auto biases_cat = mx::concatenate({gp.biases, up.biases}, 0);
          mx::eval({w_q_cat, scales_cat, biases_cat});
          MLXQuantWeights pkg{std::move(w_q_cat), std::move(scales_cat),
                              std::move(biases_cat), gp.in_dim,
                              gp.out_dim + up.out_dim};
          gu_it = mlxq_ffngu_cache_.emplace(ff.getOperation(), std::move(pkg)).first;
        }
        const auto &gu = gu_it->second;
        auto x_ff   = get(ff.getX());
        auto gu_out = qmatmul(x_ff, gu);     // [N, 2*ffn_dim]
        auto g = mx::slice(gu_out, {0, 0},       {gu_out.shape(0), ffn_dim});
        auto u = mx::slice(gu_out, {0, ffn_dim},  {gu_out.shape(0), 2 * ffn_dim});
        auto h = mx::multiply(mx::multiply(g, mx::sigmoid(g)), u);
        put(ff.getY(), qmatmul(h, *down_pkg->second));
      } else {
        auto cache_it = ffn_gateup_cache_.find(ff.getOperation());
        if (cache_it == ffn_gateup_cache_.end()) {
          auto concat = mx::concatenate({get(ff.getWGate()), get(ff.getWUp())}, 0);
          auto owned  = mx::copy(concat);
          mx::eval(owned);
          cache_it = ffn_gateup_cache_.emplace(ff.getOperation(), owned).first;
        }
        put(ff.getY(), buildFeedForward(get(ff.getX()), cache_it->second,
                                        get(ff.getWDown()), ffn_dim));
      }
    } else if (auto emb = ::llvm::dyn_cast<::mlir::mlc::EmbeddingOp>(op)) {
      int hidden = emb.getY().getType().getShape().back();
      put(emb.getY(),
          buildEmbedding(get(emb.getTable()), get(emb.getIds()), hidden));
    } else if (auto lm = ::llvm::dyn_cast<::mlir::mlc::LMHeadOp>(op)) {
      put(lm.getLogits(), buildLMHead(get(lm.getX()), get(lm.getW())));
    } else if (auto fused = ::llvm::dyn_cast<::mlir::mlc::FusedNormMatMulOp>(op)) {
      auto n = buildNorm(get(fused.getX()), get(fused.getGamma()),
                         f32Of(fused.getEpsilonAttr()));
      auto w_v = fused.getW();
      if (use_custom_q4_) {
        if (auto dims = q4_value_dims_.find(w_v); dims != q4_value_dims_.end())
          put(fused.getY(), q4_0_matmul(n, get(w_v),
                                        dims->second.first, dims->second.second));
        else
          put(fused.getY(), buildMatMul(n, get(w_v), fused.getTransposeB()));
      } else if (auto pkg_it = mlxq_value_pkg_.find(w_v);
                 pkg_it != mlxq_value_pkg_.end()) {
        put(fused.getY(), qmatmul(n, *pkg_it->second));
      } else {
        put(fused.getY(), buildMatMul(n, get(w_v), fused.getTransposeB()));
      }
    } else if (auto qkv = ::llvm::dyn_cast<::mlir::mlc::FusedNormQKVMatMulOp>(op)) {
      auto n = buildNorm(get(qkv.getX()), get(qkv.getGamma()),
                         f32Of(qkv.getEpsilonAttr()));
      auto q_ty = ::llvm::cast<::mlir::RankedTensorType>(qkv.getQ().getType());
      auto k_ty = ::llvm::cast<::mlir::RankedTensorType>(qkv.getK().getType());
      auto v_ty = ::llvm::cast<::mlir::RankedTensorType>(qkv.getV().getType());
      int q_dim = q_ty.getShape().back();
      int k_dim = k_ty.getShape().back();
      int v_dim = v_ty.getShape().back();
      auto wq_pkg = mlxq_value_pkg_.find(qkv.getWQ());
      bool all_q4 = wq_pkg != mlxq_value_pkg_.end() &&
                    mlxq_value_pkg_.count(qkv.getWK()) &&
                    mlxq_value_pkg_.count(qkv.getWV());
      if (all_q4) {
        auto qkv_it = mlxq_qkv_cache_.find(qkv.getOperation());
        if (qkv_it == mlxq_qkv_cache_.end()) {
          const auto &qp = *mlxq_value_pkg_.at(qkv.getWQ());
          const auto &kp = *mlxq_value_pkg_.at(qkv.getWK());
          const auto &vp = *mlxq_value_pkg_.at(qkv.getWV());
          auto w_q_cat    = mx::concatenate({qp.w_q, kp.w_q, vp.w_q}, 0);
          auto scales_cat = mx::concatenate({qp.scales, kp.scales, vp.scales}, 0);
          auto biases_cat = mx::concatenate({qp.biases, kp.biases, vp.biases}, 0);
          mx::eval({w_q_cat, scales_cat, biases_cat});
          MLXQuantWeights pkg{std::move(w_q_cat), std::move(scales_cat),
                              std::move(biases_cat), qp.in_dim,
                              qp.out_dim + kp.out_dim + vp.out_dim};
          qkv_it = mlxq_qkv_cache_.emplace(qkv.getOperation(), std::move(pkg)).first;
        }
        auto out_arr = qmatmul(n, qkv_it->second);  // [N, q+k+v]
        put(qkv.getQ(), mx::slice(out_arr, {0,0}, {N, q_dim}));
        put(qkv.getK(), mx::slice(out_arr, {0,q_dim}, {N, q_dim+k_dim}));
        put(qkv.getV(), mx::slice(out_arr, {0,q_dim+k_dim}, {N, q_dim+k_dim+v_dim}));
      } else {
        auto cache_it = qkv_concat_cache_.find(qkv.getOperation());
        if (cache_it == qkv_concat_cache_.end()) {
          auto w_concat = mx::concatenate({get(qkv.getWQ()), get(qkv.getWK()),
                                            get(qkv.getWV())}, 0);
          auto owned = mx::copy(w_concat); mx::eval(owned);
          cache_it = qkv_concat_cache_.emplace(qkv.getOperation(), owned).first;
        }
        auto out_arr = buildMatMul(n, cache_it->second, qkv.getTransposeB());
        put(qkv.getQ(), mx::slice(out_arr, {0,0}, {N, q_dim}));
        put(qkv.getK(), mx::slice(out_arr, {0,q_dim}, {N, q_dim+k_dim}));
        put(qkv.getV(), mx::slice(out_arr, {0,q_dim+k_dim}, {N, q_dim+k_dim+v_dim}));
      }
    } else if (auto attn = ::llvm::dyn_cast<::mlir::mlc::AttentionOp>(op)) {
      if (attn_idx >= batch_attn_caches.size())
        throw std::runtime_error("runBatch: more attention ops than KV cache layers");
      auto &cache_ptrs = batch_attn_caches[attn_idx];
      put(attn.getOut(),
          buildBatchedAttention(get(attn.getQ()), get(attn.getK()),
                                get(attn.getV()),
                                common_offset, /*seq_new=*/1,
                                i64Of(attn.getNumHeadsAttr()),
                                i64Of(attn.getNumKvHeadsAttr()),
                                i64Of(attn.getHeadDimAttr()),
                                cache_ptrs));
      ++attn_idx;
    } else if (auto ret = ::llvm::dyn_cast<::mlir::func::ReturnOp>(op)) {
      result = ret.getOperand(0);
      break;
    }
  }

  // result is logits [N, vocab_size]. Convert to fp32, split into N results.
  auto &logits_raw = get(result);
  // mlxToF32 handles any dtype → host fp32 buffer.
  auto logits_f32 = mlxToF32(logits_raw);   // evaluates and converts
  int ndim = static_cast<int>(logits_raw.ndim());
  int vocab = static_cast<int>(logits_raw.shape(ndim - 1));
  std::vector<RunResult> results;
  results.reserve(N);
  for (int i = 0; i < N; ++i) {
    RunResult r;
    r.shape = {1, vocab};
    r.data.assign(logits_f32.begin() + static_cast<ptrdiff_t>(i) * vocab,
                  logits_f32.begin() + static_cast<ptrdiff_t>(i + 1) * vocab);
    results.push_back(std::move(r));
  }
  return results;
}

void MLIRExecutor::printProfile() const {
  if (!profile_enabled_) return;
  double total = 0;
  for (auto &[k, p] : prof_buckets_) total += p.first;
  std::printf("\n[profile] %d decode steps  total_measured=%.1f ms"
              "  per_step=%.2f ms\n",
              profile_steps_, total,
              profile_steps_ > 0 ? total / profile_steps_ : 0.0);
  std::printf("%-28s  %8s  %6s  %8s  %8s\n",
              "op category", "total ms", "   %", "calls", "µs/call");
  std::printf("%s\n", std::string(70, '-').c_str());
  for (auto &k : prof_order_) {
    auto it = prof_buckets_.find(k);
    if (it == prof_buckets_.end()) continue;
    double ms = it->second.first;
    int    calls = it->second.second;
    double pct = total > 0 ? 100.0 * ms / total : 0.0;
    double upc = calls > 0 ? 1000.0 * ms / calls : 0.0;
    std::printf("%-28s  %8.2f  %5.1f%%  %8d  %8.1f\n",
                k.c_str(), ms, pct, calls, upc);
  }
  std::printf("%s\n", std::string(70, '-').c_str());
  std::printf("%-28s  %8.2f  %5.1f%%\n", "TOTAL", total, 100.0);
}

MLIRExecutor::RunResult
MLIRExecutor::run(const std::vector<int32_t> &token_ids,
                  const std::vector<int32_t> &positions) {
  if (token_ids.size() != positions.size())
    throw std::runtime_error("token_ids and positions must have equal length");
  auto func = *module_.getOps<::mlir::func::FuncOp>().begin();
  if (!func)
    throw std::runtime_error("no function in module");
  auto &entryBlock = func.getBody().front();

  // Bind func args. Inputs (ids, positions) come from the caller; everything
  // else is a GGUF weight bound by its `mlc.name` arg attribute. Bound lazily
  // so unreferenced weights aren't paid for (currently every weight is
  // referenced, but the lazy structure keeps the door open for dead-arg
  // elimination later).
  ::llvm::DenseMap<::mlir::Value, mx::array> values;
  int seq = static_cast<int>(token_ids.size());

  // Build the ids and positions arrays once and reuse.
  std::vector<float> ids_f(token_ids.begin(), token_ids.end());
  std::vector<int> ids_shape = {seq};
  // MLX wants integer dtype for take indices and rope offset source.
  // Build a fresh int32 buffer.
  auto makeI32 = [&](const std::vector<int32_t> &v) {
    std::vector<int> shape = {static_cast<int>(v.size())};
    int32_t *buf = static_cast<int32_t *>(std::malloc(v.size() * sizeof(int32_t)));
    if (!buf) throw std::bad_alloc();
    std::memcpy(buf, v.data(), v.size() * sizeof(int32_t));
    mx::Shape s(shape.begin(), shape.end());
    return mx::array(static_cast<void *>(buf), std::move(s), mx::int32,
                     [](void *p) { std::free(p); });
  };
  mx::array ids_arr = makeI32(token_ids);
  mx::array pos_arr = makeI32(positions);

  // Profiling helper: forces eval of `arr` and records elapsed time.
  // Inlined as a lambda so it captures profile_enabled_ by reference.
  // In non-profile mode this is dead code; the optimizer removes it.
  auto profEval = [&](const std::string &key, mx::array &arr) {
    if (!profile_enabled_) return;
    auto t0 = std::chrono::steady_clock::now();
    mx::eval(arr);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto it = prof_buckets_.find(key);
    if (it == prof_buckets_.end()) {
      prof_order_.push_back(key);
      prof_buckets_[key] = {ms, 1};
    } else {
      it->second.first  += ms;
      it->second.second += 1;
    }
  };
  auto profEval2 = [&](const std::string &key, mx::array &a, mx::array &b) {
    if (!profile_enabled_) return;
    auto t0 = std::chrono::steady_clock::now();
    mx::eval({a, b});
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto it = prof_buckets_.find(key);
    if (it == prof_buckets_.end()) {
      prof_order_.push_back(key);
      prof_buckets_[key] = {ms, 1};
    } else {
      it->second.first  += ms;
      it->second.second += 1;
    }
  };
  auto profEval3 = [&](const std::string &key,
                        mx::array &a, mx::array &b, mx::array &c) {
    if (!profile_enabled_) return;
    auto t0 = std::chrono::steady_clock::now();
    mx::eval({a, b, c});
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto it = prof_buckets_.find(key);
    if (it == prof_buckets_.end()) {
      prof_order_.push_back(key);
      prof_buckets_[key] = {ms, 1};
    } else {
      it->second.first  += ms;
      it->second.second += 1;
    }
  };

  mlxq_value_pkg_.clear();
  q4_value_dims_.clear();
  for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
    const std::string &nm = argName(func, i);
    auto value = entryBlock.getArgument(i);
    if (nm == "ids") {
      values.insert({value, ids_arr});
    } else if (nm == "positions") {
      values.insert({value, pos_arr});
    } else if (use_custom_q4_) {
      if (auto q4it = q4_bytes_cache_.find(nm); q4it != q4_bytes_cache_.end()) {
        values.insert({value, q4it->second});
        q4_value_dims_.insert({value, q4_dims_cache_.at(nm)});
      } else {
        auto it = weight_cache_.find(nm);
        if (it == weight_cache_.end())
          throw std::runtime_error("missing weight in cache: " + nm);
        values.insert({value, it->second});
      }
    } else if (auto q4it = mlxq_cache_.find(nm); q4it != mlxq_cache_.end()) {
      values.insert({value, q4it->second.w_q});
      mlxq_value_pkg_.insert({value, &q4it->second});
    } else {
      auto it = weight_cache_.find(nm);
      if (it == weight_cache_.end())
        throw std::runtime_error("missing weight in cache: " + nm);
      values.insert({value, it->second});
    }
  }

  auto get = [&](::mlir::Value v) -> mx::array & {
    auto it = values.find(v);
    if (it == values.end())
      throw std::runtime_error("SSA value not bound");
    return it->second;
  };
  auto put = [&](::mlir::Value v, mx::array a) {
    values.insert_or_assign(v, std::move(a));
  };

  // Walk ops top-down. Each mlc-dialect op dispatches to one of the builders
  // above; func.return marks the end and surfaces its operand as the result.
  ::mlir::Value result;
  size_t attn_idx = 0;
  for (auto &op : entryBlock) {
    if (auto matmul = ::llvm::dyn_cast<::mlir::mlc::MatMulOp>(op)) {
      auto w_v = matmul.getW();
      auto ane_it = ane_cache_.find(matmul.getOperation());
      auto x_in = get(matmul.getX());
      // ANE models are baked at a fixed M (1 today). At other seq_lens
      // (prefill) fall through to the quantized kernel.
      bool ane_match = ane_it != ane_cache_.end() &&
                       x_in.shape(0) == ane_it->second->M();
      if (ane_match) {
        auto y_out = ane_it->second->predict(x_in);
        put(matmul.getY(), mx::astype(y_out, x_in.dtype()));
        profEval("matmul_ane", get(matmul.getY()));
      } else if (use_custom_q4_) {
        if (auto dims = q4_value_dims_.find(w_v); dims != q4_value_dims_.end())
          put(matmul.getY(), q4_0_matmul(x_in, get(w_v),
                                         dims->second.first, dims->second.second));
        else
          put(matmul.getY(), buildMatMul(x_in, get(w_v), matmul.getTransposeB()));
        profEval("matmul_q4", get(matmul.getY()));
      } else if (auto pkg_it = mlxq_value_pkg_.find(w_v);
                 pkg_it != mlxq_value_pkg_.end()) {
        put(matmul.getY(), qmatmul(x_in, *pkg_it->second));
        profEval("matmul_q4", get(matmul.getY()));
      } else {
        put(matmul.getY(), buildMatMul(x_in, get(w_v), matmul.getTransposeB()));
        profEval("matmul_fp", get(matmul.getY()));
      }
    } else if (auto norm = ::llvm::dyn_cast<::mlir::mlc::NormOp>(op)) {
      put(norm.getY(),
          buildNorm(get(norm.getX()), get(norm.getGamma()),
                    f32Of(norm.getEpsilonAttr())));
      profEval("norm", get(norm.getY()));
    } else if (auto add = ::llvm::dyn_cast<::mlir::mlc::AddOp>(op)) {
      put(add.getY(), buildAdd(get(add.getA()), get(add.getB())));
      profEval("add_residual", get(add.getY()));
    } else if (auto ff = ::llvm::dyn_cast<::mlir::mlc::FeedForwardOp>(op)) {
      auto w_gate_ty =
          ::llvm::cast<::mlir::RankedTensorType>(ff.getWGate().getType());
      int ffn_dim = w_gate_ty.getShape()[0];
      auto gate_pkg = mlxq_value_pkg_.find(ff.getWGate());
      auto up_pkg   = mlxq_value_pkg_.find(ff.getWUp());
      auto down_pkg = mlxq_value_pkg_.find(ff.getWDown());
      bool all_q4 = gate_pkg != mlxq_value_pkg_.end() &&
                    up_pkg   != mlxq_value_pkg_.end() &&
                    down_pkg != mlxq_value_pkg_.end();
      if (use_custom_q4_) {
        bool cq4 = q4_value_dims_.count(ff.getWGate()) &&
                   q4_value_dims_.count(ff.getWUp()) &&
                   q4_value_dims_.count(ff.getWDown());
        auto gu_it = q4_ffngu_concat_cache_.find(ff.getOperation());
        if (gu_it == q4_ffngu_concat_cache_.end()) {
          auto cat = mx::concatenate({get(ff.getWGate()), get(ff.getWUp())}, 0);
          auto owned = mx::copy(cat); mx::eval(owned);
          gu_it = q4_ffngu_concat_cache_.emplace(ff.getOperation(), owned).first;
        }
        if (cq4) {
          int in_dim = q4_value_dims_.at(ff.getWGate()).first;
          auto gu = q4_0_matmul_gate_up(get(ff.getX()), gu_it->second,
                                        in_dim, ffn_dim, ffn_dim);
          auto h = mx::multiply(mx::multiply(gu.gate, mx::sigmoid(gu.gate)), gu.up);
          int din = q4_value_dims_.at(ff.getWDown()).first;
          int dout = q4_value_dims_.at(ff.getWDown()).second;
          put(ff.getY(), q4_0_matmul(h, get(ff.getWDown()), din, dout));
        } else {
          put(ff.getY(), buildFeedForward(get(ff.getX()), gu_it->second,
                                         get(ff.getWDown()), ffn_dim));
        }
      } else if (all_q4) {
        // Concat gate ‖ up quantized packs along the out-row axis once,
        // then do a single quantized_matmul + slice. Cache in mlxq_ffngu_cache_.
        auto gu_cache_it = mlxq_ffngu_cache_.find(ff.getOperation());
        if (gu_cache_it == mlxq_ffngu_cache_.end()) {
          const auto &gp = *gate_pkg->second;
          const auto &up = *up_pkg->second;
          auto w_q_cat    = mx::concatenate({gp.w_q, up.w_q}, /*axis=*/0);
          auto scales_cat = mx::concatenate({gp.scales, up.scales}, /*axis=*/0);
          auto biases_cat = mx::concatenate({gp.biases, up.biases}, /*axis=*/0);
          mx::eval({w_q_cat, scales_cat, biases_cat});
          MLXQuantWeights pkg{std::move(w_q_cat), std::move(scales_cat),
                              std::move(biases_cat), gp.in_dim,
                              gp.out_dim + up.out_dim};
          gu_cache_it = mlxq_ffngu_cache_.emplace(ff.getOperation(),
                                                   std::move(pkg)).first;
        }
        const auto &gu = gu_cache_it->second;
        auto x_ff = get(ff.getX());
        auto gu_out = qmatmul(x_ff, gu);
        profEval("ffn_gate_up_q4", gu_out);
        auto g = mx::slice(gu_out, {0, 0}, {gu_out.shape(0), ffn_dim});
        auto u = mx::slice(gu_out, {0, ffn_dim}, {gu_out.shape(0), 2 * ffn_dim});
        auto h = mx::multiply(mx::multiply(g, mx::sigmoid(g)), u);
        profEval2("silu_mul", g, u);  // forces g, u eval to time silu*mul
        put(ff.getY(), qmatmul(h, *down_pkg->second));
        profEval("ffn_down_q4", get(ff.getY()));
      } else {
        auto cache_it = ffn_gateup_cache_.find(ff.getOperation());
        if (cache_it == ffn_gateup_cache_.end()) {
          auto concat = mx::concatenate({get(ff.getWGate()), get(ff.getWUp())},
                                        /*axis=*/0);
          auto owned = mx::copy(concat);
          mx::eval(owned);
          cache_it = ffn_gateup_cache_.emplace(ff.getOperation(), owned).first;
        }
        put(ff.getY(), buildFeedForward(get(ff.getX()), cache_it->second,
                                       get(ff.getWDown()), ffn_dim));
        profEval("ffn_fp", get(ff.getY()));
      }
    } else if (auto emb = ::llvm::dyn_cast<::mlir::mlc::EmbeddingOp>(op)) {
      int hidden = emb.getY().getType().getShape().back();
      put(emb.getY(),
          buildEmbedding(get(emb.getTable()), get(emb.getIds()), hidden));
      profEval("embedding", get(emb.getY()));
    } else if (auto lm = ::llvm::dyn_cast<::mlir::mlc::LMHeadOp>(op)) {
      put(lm.getLogits(), buildLMHead(get(lm.getX()), get(lm.getW())));
      profEval("lm_head", get(lm.getLogits()));
    } else if (auto fused = ::llvm::dyn_cast<::mlir::mlc::FusedNormMatMulOp>(op)) {
      auto n = buildNorm(get(fused.getX()), get(fused.getGamma()),
                         f32Of(fused.getEpsilonAttr()));
      auto w_v = fused.getW();
      if (use_custom_q4_) {
        if (auto dims = q4_value_dims_.find(w_v); dims != q4_value_dims_.end())
          put(fused.getY(), q4_0_matmul(n, get(w_v),
                                        dims->second.first, dims->second.second));
        else
          put(fused.getY(), buildMatMul(n, get(w_v), fused.getTransposeB()));
        profEval("fused_norm_matmul_q4", get(fused.getY()));
      } else if (auto pkg_it = mlxq_value_pkg_.find(w_v);
                 pkg_it != mlxq_value_pkg_.end()) {
        put(fused.getY(), qmatmul(n, *pkg_it->second));
        profEval("fused_norm_matmul_q4", get(fused.getY()));
      } else {
        put(fused.getY(), buildMatMul(n, get(w_v), fused.getTransposeB()));
        profEval("fused_norm_matmul_fp", get(fused.getY()));
      }
    } else if (auto qkv = ::llvm::dyn_cast<::mlir::mlc::FusedNormQKVMatMulOp>(op)) {
      auto n = buildNorm(get(qkv.getX()), get(qkv.getGamma()),
                         f32Of(qkv.getEpsilonAttr()));
      auto q_ty = ::llvm::cast<::mlir::RankedTensorType>(qkv.getQ().getType());
      auto k_ty = ::llvm::cast<::mlir::RankedTensorType>(qkv.getK().getType());
      auto v_ty = ::llvm::cast<::mlir::RankedTensorType>(qkv.getV().getType());
      int q_dim = q_ty.getShape().back();
      int k_dim = k_ty.getShape().back();
      int v_dim = v_ty.getShape().back();
      // CoreML/ANE path: one predict() over the baked concat'd Q‖K‖V weight,
      // then split. Falls through at any seq_len other than the baked M.
      auto ane_qkv_it = ane_cache_.find(qkv.getOperation());
      if (ane_qkv_it != ane_cache_.end() &&
          n.shape(0) == ane_qkv_it->second->M()) {
        auto out_arr = ane_qkv_it->second->predict(n);
        out_arr = mx::astype(out_arr, n.dtype());
        put(qkv.getQ(),
            mx::slice(out_arr, {0, 0}, {out_arr.shape(0), q_dim}));
        put(qkv.getK(),
            mx::slice(out_arr, {0, q_dim},
                      {out_arr.shape(0), q_dim + k_dim}));
        put(qkv.getV(),
            mx::slice(out_arr, {0, q_dim + k_dim},
                      {out_arr.shape(0), q_dim + k_dim + v_dim}));
        continue;
      }
      auto wq_pkg = mlxq_value_pkg_.find(qkv.getWQ());
      bool all_q4 = wq_pkg != mlxq_value_pkg_.end() &&
                    mlxq_value_pkg_.count(qkv.getWK()) &&
                    mlxq_value_pkg_.count(qkv.getWV());
      if (use_custom_q4_) {
        bool cq4 = q4_value_dims_.count(qkv.getWQ()) &&
                   q4_value_dims_.count(qkv.getWK()) &&
                   q4_value_dims_.count(qkv.getWV());
        auto qkv_it = q4_qkv_concat_cache_.find(qkv.getOperation());
        if (qkv_it == q4_qkv_concat_cache_.end()) {
          auto cat = mx::concatenate({get(qkv.getWQ()), get(qkv.getWK()),
                                       get(qkv.getWV())}, 0);
          auto owned = mx::copy(cat); mx::eval(owned);
          qkv_it = q4_qkv_concat_cache_.emplace(qkv.getOperation(), owned).first;
        }
        if (cq4) {
          int in_dim = q4_value_dims_.at(qkv.getWQ()).first;
          auto r = q4_0_matmul_qkv(n, qkv_it->second, in_dim,
                                    q_dim, k_dim, v_dim);
          put(qkv.getQ(), r.q); put(qkv.getK(), r.k); put(qkv.getV(), r.v);
          profEval3("qkv_proj_q4", get(qkv.getQ()), get(qkv.getK()), get(qkv.getV()));
        } else {
          auto out_arr = buildMatMul(n, qkv_it->second, qkv.getTransposeB());
          put(qkv.getQ(), mx::slice(out_arr, {0,0}, {out_arr.shape(0), q_dim}));
          put(qkv.getK(), mx::slice(out_arr, {0,q_dim}, {out_arr.shape(0), q_dim+k_dim}));
          put(qkv.getV(), mx::slice(out_arr, {0,q_dim+k_dim}, {out_arr.shape(0), q_dim+k_dim+v_dim}));
          profEval3("qkv_proj_fp", get(qkv.getQ()), get(qkv.getK()), get(qkv.getV()));
        }
      } else if (all_q4) {
        auto qkv_cache_it = mlxq_qkv_cache_.find(qkv.getOperation());
        if (qkv_cache_it == mlxq_qkv_cache_.end()) {
          const auto &qp = *mlxq_value_pkg_.at(qkv.getWQ());
          const auto &kp = *mlxq_value_pkg_.at(qkv.getWK());
          const auto &vp = *mlxq_value_pkg_.at(qkv.getWV());
          auto w_q_cat    = mx::concatenate({qp.w_q, kp.w_q, vp.w_q}, 0);
          auto scales_cat = mx::concatenate({qp.scales, kp.scales, vp.scales}, 0);
          auto biases_cat = mx::concatenate({qp.biases, kp.biases, vp.biases}, 0);
          mx::eval({w_q_cat, scales_cat, biases_cat});
          MLXQuantWeights pkg{std::move(w_q_cat), std::move(scales_cat),
                              std::move(biases_cat), qp.in_dim,
                              qp.out_dim + kp.out_dim + vp.out_dim};
          qkv_cache_it = mlxq_qkv_cache_.emplace(qkv.getOperation(),
                                                   std::move(pkg)).first;
        }
        const auto &qkv_pkg = qkv_cache_it->second;
        auto out_arr = qmatmul(n, qkv_pkg);
        put(qkv.getQ(),
            mx::slice(out_arr, {0, 0}, {out_arr.shape(0), q_dim}));
        put(qkv.getK(),
            mx::slice(out_arr, {0, q_dim}, {out_arr.shape(0), q_dim + k_dim}));
        put(qkv.getV(),
            mx::slice(out_arr, {0, q_dim + k_dim},
                      {out_arr.shape(0), q_dim + k_dim + v_dim}));
        profEval3("qkv_proj_q4", get(qkv.getQ()), get(qkv.getK()), get(qkv.getV()));
      } else {
        // fp16 path: gate||K||V concat cached in qkv_concat_cache_.
        auto cache_it = qkv_concat_cache_.find(qkv.getOperation());
        if (cache_it == qkv_concat_cache_.end()) {
          auto w_concat = mx::concatenate({get(qkv.getWQ()), get(qkv.getWK()),
                                            get(qkv.getWV())},
                                           0);
          auto owned = mx::copy(w_concat);
          mx::eval(owned);
          cache_it = qkv_concat_cache_.emplace(qkv.getOperation(), owned).first;
        }
        auto out_arr = buildMatMul(n, cache_it->second, qkv.getTransposeB());
        put(qkv.getQ(),
            mx::slice(out_arr, {0, 0}, {out_arr.shape(0), q_dim}));
        put(qkv.getK(),
            mx::slice(out_arr, {0, q_dim}, {out_arr.shape(0), q_dim + k_dim}));
        put(qkv.getV(),
            mx::slice(out_arr, {0, q_dim + k_dim},
                      {out_arr.shape(0), q_dim + k_dim + v_dim}));
        profEval3("qkv_proj_fp", get(qkv.getQ()), get(qkv.getK()), get(qkv.getV()));
      }
    } else if (auto attn = ::llvm::dyn_cast<::mlir::mlc::AttentionOp>(op)) {
      if (attn_idx >= kv_cache_.size())
        kv_cache_.emplace_back();
      put(attn.getOut(),
          buildAttention(get(attn.getQ()), get(attn.getK()), get(attn.getV()),
                         get(attn.getPositions()),
                         i64Of(attn.getNumHeadsAttr()),
                         i64Of(attn.getNumKvHeadsAttr()),
                         i64Of(attn.getHeadDimAttr()),
                         &kv_cache_[attn_idx]));
      profEval("attention_sdpa_kvcache", get(attn.getOut()));
      ++attn_idx;
    } else if (auto ret = ::llvm::dyn_cast<::mlir::func::ReturnOp>(op)) {
      result = ret.getOperand(0);
      break;
    }
  }

  auto &resArr = get(result);
  {
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> host = mlxToF32(resArr);
    auto t1 = std::chrono::steady_clock::now();
    if (profile_enabled_) {
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      auto it = prof_buckets_.find("final_eval_host");
      if (it == prof_buckets_.end()) {
        prof_order_.push_back("final_eval_host");
        prof_buckets_["final_eval_host"] = {ms, 1};
      } else {
        it->second.first += ms; it->second.second += 1;
      }
    }
    std::vector<int> shape;
    for (auto d : resArr.shape())
      shape.push_back(static_cast<int>(d));
    ++profile_steps_;
    return RunResult{std::move(host), std::move(shape)};
  }
}

} // namespace mlir::mlc::exec
