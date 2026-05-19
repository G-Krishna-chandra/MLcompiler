#include "compiler/mlir/exec/MLIRExecutor.h"
#include "compiler/mlir/exec/MLXBuilder.h"
#include "compiler/mlir/exec/Quantize.h"
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

// SwiGLU FFN: down(silu(gate(x)) * up(x)). v1 calls matmul thrice.
// All three weights are stored [out, in] (GGUF / numpy row-major), so each
// matmul transposes the weight to get the matmul-convention [in, out].
mx::array buildFeedForward(const mx::array &x, const mx::array &w_gate,
                           const mx::array &w_up, const mx::array &w_down) {
  auto g = mx::matmul(x, mx::transpose(w_gate));
  auto u = mx::matmul(x, mx::transpose(w_up));
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

  // KV cache: concat the new (post-RoPE) K and (raw) V onto whatever the
  // cache already holds, then write a fresh, materialized copy back. K is
  // stored fp32 to dodge the fp16 RoPE precision issue noted above; V is
  // stored at its natural dtype.
  if (cache) {
    if (cache->K.has_value())
      k = mx::concatenate({*cache->K, k}, /*axis=*/1);
    if (cache->V.has_value())
      v = mx::concatenate({*cache->V, v}, 1);
    auto k_owned = mx::copy(k);
    auto v_owned = mx::copy(v);
    mx::eval({k_owned, v_owned});
    cache->K = k_owned;
    cache->V = v_owned;
    k = k_owned;
    v = v_owned;
  }
  int seq_full = k.shape(1);
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
                           const ::mlc::frontend::GGUFLoader &loader)
    : module_(module), loader_(loader) {
  // Pre-load every weight arg: dequant Q4_0 / Q6_K / etc. to fp32 via the
  // runtime helpers, upload as an mx::array, cast to fp16, eval to force
  // materialization, then drop the fp32 view. Each weight is materialized
  // exactly once for the executor's lifetime — `run()` does zero dequant.
  auto func = *module_.getOps<::mlir::func::FuncOp>().begin();
  auto &entry = func.getBody().front();
  ::std::vector<mx::array> to_eval;
  to_eval.reserve(entry.getNumArguments());
  for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
    const std::string &nm = argName(func, i);
    if (nm == "ids" || nm == "positions")
      continue;
    auto a32 = gufWeightToMLX(loader_, nm);
    auto a16 = mx::astype(a32, mx::float16);
    auto [it, inserted] = weight_cache_.emplace(nm, a16);
    if (inserted)
      to_eval.push_back(a16);
  }
  // Batch-materialize so MLX can pipeline the host→device uploads.
  mx::eval(to_eval);
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

  for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
    const std::string &nm = argName(func, i);
    auto value = entryBlock.getArgument(i);
    if (nm == "ids") {
      values.insert({value, ids_arr});
    } else if (nm == "positions") {
      values.insert({value, pos_arr});
    } else {
      // GGUF weight — resident fp16, populated once in the constructor.
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
      put(matmul.getY(),
          buildMatMul(get(matmul.getX()), get(matmul.getW()),
                      matmul.getTransposeB()));
    } else if (auto norm = ::llvm::dyn_cast<::mlir::mlc::NormOp>(op)) {
      put(norm.getY(),
          buildNorm(get(norm.getX()), get(norm.getGamma()),
                    f32Of(norm.getEpsilonAttr())));
    } else if (auto add = ::llvm::dyn_cast<::mlir::mlc::AddOp>(op)) {
      put(add.getY(), buildAdd(get(add.getA()), get(add.getB())));
    } else if (auto ff = ::llvm::dyn_cast<::mlir::mlc::FeedForwardOp>(op)) {
      put(ff.getY(),
          buildFeedForward(get(ff.getX()), get(ff.getWGate()),
                           get(ff.getWUp()), get(ff.getWDown())));
    } else if (auto emb = ::llvm::dyn_cast<::mlir::mlc::EmbeddingOp>(op)) {
      int hidden = emb.getY().getType().getShape().back();
      put(emb.getY(),
          buildEmbedding(get(emb.getTable()), get(emb.getIds()), hidden));
    } else if (auto lm = ::llvm::dyn_cast<::mlir::mlc::LMHeadOp>(op)) {
      put(lm.getLogits(), buildLMHead(get(lm.getX()), get(lm.getW())));
    } else if (auto fused = ::llvm::dyn_cast<::mlir::mlc::FusedNormMatMulOp>(op)) {
      auto n = buildNorm(get(fused.getX()), get(fused.getGamma()),
                         f32Of(fused.getEpsilonAttr()));
      put(fused.getY(), buildMatMul(n, get(fused.getW()),
                                    fused.getTransposeB()));
    } else if (auto qkv = ::llvm::dyn_cast<::mlir::mlc::FusedNormQKVMatMulOp>(op)) {
      auto n = buildNorm(get(qkv.getX()), get(qkv.getGamma()),
                         f32Of(qkv.getEpsilonAttr()));
      // Concatenate Q/K/V weights once at first use along the out-dim
      // axis; cache keyed by the op so subsequent forward passes skip
      // the re-concat.
      auto cache_it = qkv_concat_cache_.find(qkv.getOperation());
      if (cache_it == qkv_concat_cache_.end()) {
        auto w_concat = mx::concatenate(
            {get(qkv.getWQ()), get(qkv.getWK()), get(qkv.getWV())}, /*axis=*/0);
        auto owned = mx::copy(w_concat);
        mx::eval(owned);
        cache_it =
            qkv_concat_cache_.emplace(qkv.getOperation(), owned).first;
      }
      auto out = buildMatMul(n, cache_it->second, qkv.getTransposeB());
      // out has shape [seq, q_out + k_out + v_out]. Split by the result
      // types' last dims.
      auto q_ty = ::llvm::cast<::mlir::RankedTensorType>(qkv.getQ().getType());
      auto k_ty = ::llvm::cast<::mlir::RankedTensorType>(qkv.getK().getType());
      auto v_ty = ::llvm::cast<::mlir::RankedTensorType>(qkv.getV().getType());
      int q_dim = q_ty.getShape().back();
      int k_dim = k_ty.getShape().back();
      int v_dim = v_ty.getShape().back();
      put(qkv.getQ(),
          mx::slice(out, {0, 0}, {out.shape(0), q_dim}));
      put(qkv.getK(),
          mx::slice(out, {0, q_dim}, {out.shape(0), q_dim + k_dim}));
      put(qkv.getV(),
          mx::slice(out, {0, q_dim + k_dim},
                    {out.shape(0), q_dim + k_dim + v_dim}));
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
