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
mx::array buildFeedForward(const mx::array &x, const mx::array &w_gate,
                           const mx::array &w_up, const mx::array &w_down) {
  auto g = mx::matmul(x, w_gate);
  auto u = mx::matmul(x, w_up);
  // silu(x) = x * sigmoid(x).
  auto h = mx::multiply(mx::multiply(g, mx::sigmoid(g)), u);
  return mx::matmul(h, w_down);
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

// Dense attention with causal mask and inline RoPE for Q/K.
// q: [seq, n_heads * head_dim]
// k, v: [seq, n_kv_heads * head_dim]
// positions: [seq] int32. Used only for the RoPE start offset (positions[0]).
mx::array buildAttention(const mx::array &q_flat, const mx::array &k_flat,
                         const mx::array &v_flat, const mx::array &positions,
                         int n_heads, int n_kv_heads, int head_dim) {
  int seq = q_flat.shape(0);

  // Reshape to [seq, head, dim] then transpose to [head, seq, dim].
  auto q = mx::transpose(mx::reshape(q_flat, {seq, n_heads, head_dim}),
                         {1, 0, 2});
  auto k = mx::transpose(mx::reshape(k_flat, {seq, n_kv_heads, head_dim}),
                         {1, 0, 2});
  auto v = mx::transpose(mx::reshape(v_flat, {seq, n_kv_heads, head_dim}),
                         {1, 0, 2});

  // RoPE — start offset from positions[0]. We pull it eagerly (sync) since
  // MLX's rope op takes an int, not an array. v1 only supports
  // contiguous-from-offset sequences.
  mx::array p0 = mx::take(positions, 0);
  mx::eval(p0);
  int offset = static_cast<int>(p0.item<int32_t>());
  // Llama-style RoPE: non-traditional layout, base = 10000.
  q = mx::fast::rope(q, head_dim, /*traditional=*/false, /*base=*/10000.0f,
                     /*scale=*/1.0f, offset);
  k = mx::fast::rope(k, head_dim, false, 10000.0f, 1.0f, offset);

  // GQA: replicate K, V across each group of n_heads/n_kv_heads.
  if (n_heads != n_kv_heads) {
    int n_rep = n_heads / n_kv_heads;
    auto k_exp = mx::expand_dims(k, 1);  // [n_kv, 1, seq, dim]
    auto v_exp = mx::expand_dims(v, 1);
    k_exp = mx::broadcast_to(k_exp, mx::Shape{n_kv_heads, n_rep, seq, head_dim});
    v_exp = mx::broadcast_to(v_exp, mx::Shape{n_kv_heads, n_rep, seq, head_dim});
    k = mx::reshape(k_exp, {n_heads, seq, head_dim});
    v = mx::reshape(v_exp, {n_heads, seq, head_dim});
  }

  // scores = Q @ K^T / sqrt(head_dim) — [n_heads, seq_q, seq_k]
  auto k_T = mx::transpose(k, {0, 2, 1});
  auto scale_scalar = mx::array(1.0f / std::sqrt(static_cast<float>(head_dim)));
  auto scores = mx::multiply(mx::matmul(q, k_T), scale_scalar);

  // Causal mask: -inf above the diagonal.
  auto mask = mx::tril(mx::ones({seq, seq}, mx::float32));
  auto neg_inf = mx::array(-1e9f);
  scores = mx::where(mx::equal(mask, mx::array(0.0f)), neg_inf, scores);

  scores = mx::softmax(scores, /*axes=*/std::vector<int>{-1});
  auto out = mx::matmul(scores, v);                    // [n_heads, seq, dim]
  out = mx::transpose(out, {1, 0, 2});                  // [seq, n_heads, dim]
  return mx::reshape(out, {seq, n_heads * head_dim});
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
    : module_(module), loader_(loader) {}

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
      // GGUF weight — dequant + upload.
      values.insert({value, gufWeightToMLX(loader_, nm)});
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
    } else if (auto attn = ::llvm::dyn_cast<::mlir::mlc::AttentionOp>(op)) {
      put(attn.getOut(),
          buildAttention(get(attn.getQ()), get(attn.getK()), get(attn.getV()),
                         get(attn.getPositions()),
                         i64Of(attn.getNumHeadsAttr()),
                         i64Of(attn.getNumKvHeadsAttr()),
                         i64Of(attn.getHeadDimAttr())));
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
