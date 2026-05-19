#pragma once

#include "compiler/frontends/gguf_loader.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/DenseMap.h"

#include <mlx/array.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir::mlc::exec {

// One per-layer KV cache slot. K/V are stored in `[n_kv_heads, seq, head_dim]`
// layout (post-transpose, post-RoPE for K), so attention can read them
// directly. Optional because cache is empty before the first prefill call.
struct KVCacheSlot {
  std::optional<mlx::core::array> K;
  std::optional<mlx::core::array> V;
};

// Walks a mlc-dialect ModuleOp and executes it on MLX. Weight tensors come
// from a GGUFLoader (matched by the `mlc.name` arg attribute the emitter
// attaches). Result of the inference function is realized to a host fp32
// buffer.
//
// Weights are dequantized to fp16 once at construction and held resident
// for the lifetime of the executor; per-run() calls do no dequant work.
//
// Each `run()` appends to a per-attention-op KV cache, so generation is
// O(1) per token (decode mode = pass a single token + its absolute position;
// the cache holds prefix K/V across calls). `reset()` clears the cache for
// a fresh prompt.
class MLIRExecutor {
public:
  MLIRExecutor(::mlir::ModuleOp module,
               const ::mlc::frontend::GGUFLoader &loader);

  struct RunResult {
    std::vector<float> data;
    std::vector<int> shape;
  };
  RunResult run(const std::vector<int32_t> &token_ids,
                const std::vector<int32_t> &positions);

  // Drop all KV cache state. Call this before a new prompt.
  void reset();

private:
  ::mlir::ModuleOp module_;
  const ::mlc::frontend::GGUFLoader &loader_;
  // Resident fp16 weights, keyed by GGUF tensor name. Used for non-matmul
  // operands (norm gamma, embedding table, lm_head's Q6_K weight) and as
  // a fallback when a matmul weight isn't in Q4_0.
  std::unordered_map<std::string, mlx::core::array> weight_cache_;
  // Q4_0 weight bytes (raw GGUF blocks, no dequant) keyed by tensor name.
  // Used by the custom Q4_0 Metal kernel — drops weight memory by ~2x vs
  // the fp16-resident path.
  std::unordered_map<std::string, mlx::core::array> q4_bytes_cache_;
  // (in_dim, out_dim) for each Q4_0 weight, parallel to q4_bytes_cache_.
  std::unordered_map<std::string, std::pair<int, int>> q4_dims_cache_;
  // Set of mlir::Values bound to Q4_0 bytes (vs fp16 weights). Builders
  // check this to dispatch to the Q4_0 kernel.
  ::llvm::DenseMap<::mlir::Value, std::pair<int, int>> q4_value_dims_;
  // Pre-concatenated W_QKV weights, keyed by the fused_norm_qkv_matmul
  // Operation* that owns them. Populated lazily on first encounter.
  std::unordered_map<::mlir::Operation *, mlx::core::array> qkv_concat_cache_;
  // Pre-concatenated W_gate ++ W_up weights, keyed by the mlc.feedforward
  // Operation*. Same idea as `qkv_concat_cache_` — one batched matmul
  // instead of two for the SwiGLU gate/up projections.
  std::unordered_map<::mlir::Operation *, mlx::core::array> ffn_gateup_cache_;
  // KV cache, one slot per mlc.attention op encountered (in walk order).
  // Grows on the first prefill call; updated in place every step.
  std::vector<KVCacheSlot> kv_cache_;
};

} // namespace mlir::mlc::exec
