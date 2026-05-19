#include "compiler/mlir/exec/MLIRExecutor.h"
#include "compiler/mlir/exec/MLXBuilder.h"
#include "compiler/mlir/exec/Q4MatMul.h"
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
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

namespace mx = mlx::core;

namespace mlir::mlc::exec {

namespace {

// ----- Per-op MLX builders -----
// Kept as free functions inside this TU so the walker switch below stays
// short and readable. Each builder takes already-materialized mx::array
// operands and returns the result of the op as a (lazy) mx::array.

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

  // Second pass: per-arg, decide between Q4_0 bytes (matmul + Q4_0 dtype)
  // and fp16 dequant (everything else, including embedding table, norm
  // gamma, and the Q6_K lm_head weight).
  auto &entry = func.getBody().front();
  ::std::vector<mx::array> to_eval;
  to_eval.reserve(entry.getNumArguments());
  for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
    const std::string &nm = argName(func, i);
    if (nm == "ids" || nm == "positions") continue;
    const auto &info = loader_.tensors().at(nm);
    bool is_q4_target = matmul_arg_idxs.count(i) &&
                        info.dtype == ::mlc::frontend::GGML_TYPE_Q4_0 &&
                        info.shape.size() == 2;
    if (is_q4_target) {
      auto bytes = gufWeightToBytesMLX(loader_, nm);
      q4_bytes_cache_.emplace(nm, bytes);
      // GGUF shape is innermost-first → shape[0]=in, shape[1]=out.
      int in_dim = static_cast<int>(info.shape[0]);
      int out_dim = static_cast<int>(info.shape[1]);
      q4_dims_cache_.emplace(nm, std::pair<int, int>{in_dim, out_dim});
      to_eval.push_back(bytes);
    } else {
      auto a32 = gufWeightToMLX(loader_, nm);
      auto a16 = mx::astype(a32, mx::float16);
      auto [it, inserted] = weight_cache_.emplace(nm, a16);
      if (inserted) to_eval.push_back(a16);
    }
  }
  mx::eval(to_eval);

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
      // The bound value in q4_bytes_cache_ is the raw GGUF byte array.
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

void MLIRExecutor::reset() { kv_cache_.clear(); }

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

  q4_value_dims_.clear();
  for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
    const std::string &nm = argName(func, i);
    auto value = entryBlock.getArgument(i);
    if (nm == "ids") {
      values.insert({value, ids_arr});
    } else if (nm == "positions") {
      values.insert({value, pos_arr});
    } else if (auto q4 = q4_bytes_cache_.find(nm); q4 != q4_bytes_cache_.end()) {
      values.insert({value, q4->second});
      q4_value_dims_.insert({value, q4_dims_cache_.at(nm)});
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
      // (prefill) fall through to the Q4_0 kernel — already cached.
      bool ane_match = ane_it != ane_cache_.end() &&
                       x_in.shape(0) == ane_it->second->M();
      if (ane_match) {
        // CoreML/ANE path. Input x is whatever dtype; ANE wraps it as
        // fp16 internally; output comes back as fp32 (CoreML default).
        auto y_out = ane_it->second->predict(x_in);
        put(matmul.getY(), mx::astype(y_out, x_in.dtype()));
      } else if (auto dims = q4_value_dims_.find(w_v);
                 dims != q4_value_dims_.end()) {
        put(matmul.getY(), q4_0_matmul(get(matmul.getX()), get(w_v),
                                       dims->second.first,
                                       dims->second.second));
      } else {
        put(matmul.getY(),
            buildMatMul(get(matmul.getX()), get(w_v),
                        matmul.getTransposeB()));
      }
    } else if (auto norm = ::llvm::dyn_cast<::mlir::mlc::NormOp>(op)) {
      put(norm.getY(),
          buildNorm(get(norm.getX()), get(norm.getGamma()),
                    f32Of(norm.getEpsilonAttr())));
    } else if (auto add = ::llvm::dyn_cast<::mlir::mlc::AddOp>(op)) {
      put(add.getY(), buildAdd(get(add.getA()), get(add.getB())));
    } else if (auto ff = ::llvm::dyn_cast<::mlir::mlc::FeedForwardOp>(op)) {
      auto w_gate_ty =
          ::llvm::cast<::mlir::RankedTensorType>(ff.getWGate().getType());
      int ffn_dim = w_gate_ty.getShape()[0];
      bool all_q4 = q4_value_dims_.count(ff.getWGate()) &&
                    q4_value_dims_.count(ff.getWUp()) &&
                    q4_value_dims_.count(ff.getWDown());
      auto cache_it = ffn_gateup_cache_.find(ff.getOperation());
      if (cache_it == ffn_gateup_cache_.end()) {
        auto concat =
            mx::concatenate({get(ff.getWGate()), get(ff.getWUp())}, /*axis=*/0);
        auto owned = mx::copy(concat);
        mx::eval(owned);
        cache_it = ffn_gateup_cache_.emplace(ff.getOperation(), owned).first;
      }
      mx::array y_ff = [&]() {
        if (all_q4) {
          int in_dim = q4_value_dims_.at(ff.getWGate()).first;
          // Gate+Up batched matmul through the Q4 kernel:
          auto gu = q4_0_matmul(get(ff.getX()), cache_it->second, in_dim,
                                 2 * ffn_dim);
          auto g = mx::slice(gu, {0, 0}, {gu.shape(0), ffn_dim});
          auto u = mx::slice(gu, {0, ffn_dim}, {gu.shape(0), 2 * ffn_dim});
          auto h = mx::multiply(mx::multiply(g, mx::sigmoid(g)), u);
          int down_in = q4_value_dims_.at(ff.getWDown()).first;
          int down_out = q4_value_dims_.at(ff.getWDown()).second;
          return q4_0_matmul(h, get(ff.getWDown()), down_in, down_out);
        }
        return buildFeedForward(get(ff.getX()), cache_it->second,
                                 get(ff.getWDown()), ffn_dim);
      }();
      put(ff.getY(), y_ff);
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
      auto dims = q4_value_dims_.find(w_v);
      if (dims != q4_value_dims_.end()) {
        put(fused.getY(), q4_0_matmul(n, get(w_v), dims->second.first,
                                       dims->second.second));
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
      // CoreML/ANE path: one predict() over the baked concat'd Q‖K‖V weight,
      // then split. Saves three Q4_0 kernel launches per layer. Falls
      // through to the Q4_0 kernel at any seq_len other than the baked
      // M (1 today) — that's the prefill case.
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
      bool all_q4 = q4_value_dims_.count(qkv.getWQ()) &&
                    q4_value_dims_.count(qkv.getWK()) &&
                    q4_value_dims_.count(qkv.getWV());
      auto cache_it = qkv_concat_cache_.find(qkv.getOperation());
      if (cache_it == qkv_concat_cache_.end()) {
        if (all_q4) {
          // Q4_0 byte concat: each row's bytes are contiguous (row_bytes =
          // (in/32)*18); rows for Q come first, then K, then V. The Q4 kernel
          // walks `out_total` rows in one launch.
          auto w_concat = mx::concatenate({get(qkv.getWQ()), get(qkv.getWK()),
                                            get(qkv.getWV())},
                                           /*axis=*/0);
          auto owned = mx::copy(w_concat);
          mx::eval(owned);
          cache_it = qkv_concat_cache_.emplace(qkv.getOperation(), owned).first;
        } else {
          auto w_concat = mx::concatenate({get(qkv.getWQ()), get(qkv.getWK()),
                                            get(qkv.getWV())},
                                           0);
          auto owned = mx::copy(w_concat);
          mx::eval(owned);
          cache_it = qkv_concat_cache_.emplace(qkv.getOperation(), owned).first;
        }
      }
      mx::array out_arr = [&]() {
        if (all_q4) {
          int in_dim = q4_value_dims_.at(qkv.getWQ()).first;
          return q4_0_matmul(n, cache_it->second, in_dim,
                              q_dim + k_dim + v_dim);
        }
        return buildMatMul(n, cache_it->second, qkv.getTransposeB());
      }();
      put(qkv.getQ(),
          mx::slice(out_arr, {0, 0}, {out_arr.shape(0), q_dim}));
      put(qkv.getK(),
          mx::slice(out_arr, {0, q_dim}, {out_arr.shape(0), q_dim + k_dim}));
      put(qkv.getV(),
          mx::slice(out_arr, {0, q_dim + k_dim},
                    {out_arr.shape(0), q_dim + k_dim + v_dim}));
    } else if (auto attn = ::llvm::dyn_cast<::mlir::mlc::AttentionOp>(op)) {
      // Grow the cache lazily so an executor used without any decode
      // doesn't pay for the storage.
      if (attn_idx >= kv_cache_.size())
        kv_cache_.emplace_back();
      put(attn.getOut(),
          buildAttention(get(attn.getQ()), get(attn.getK()), get(attn.getV()),
                         get(attn.getPositions()),
                         i64Of(attn.getNumHeadsAttr()),
                         i64Of(attn.getNumKvHeadsAttr()),
                         i64Of(attn.getHeadDimAttr()),
                         &kv_cache_[attn_idx]));
      ++attn_idx;
    } else if (auto ret = ::llvm::dyn_cast<::mlir::func::ReturnOp>(op)) {
      result = ret.getOperand(0);
      break;
    }
    // (Skip ops we don't recognize; the emitter today produces only the
    // 7 mlc ops + func.return.)
  }

  auto &resArr = get(result);
  std::vector<float> host = mlxToF32(resArr);
  std::vector<int> shape;
  for (auto d : resArr.shape())
    shape.push_back(static_cast<int>(d));
  return RunResult{std::move(host), std::move(shape)};
}

} // namespace mlir::mlc::exec
