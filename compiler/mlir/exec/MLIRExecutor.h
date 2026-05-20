#pragma once

#include "compiler/frontends/gguf_loader.hpp"

#include "compiler/mlir/exec/ANEMatMul.h"
#include "compiler/mlir/exec/FusedKernels.h"
#include "compiler/mlir/exec/MLXQuantize.h"
#include "compiler/mlir/exec/Q4MatMul.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/DenseMap.h"

#include <memory>

#include <mlx/array.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir::mlc::exec {

// One per-layer KV cache slot. K and V are pre-allocated at full
// `[n_kv_heads, max_seq, head_dim]` capacity on first use, and updated
// in place via `mx::slice_update` as tokens stream in — no per-token
// concat-and-copy. `seq_filled` is the contiguous prefix that has been
// written; attention reads `[:, :seq_filled, :]`.
struct KVCacheSlot {
  std::optional<mlx::core::array> K;
  std::optional<mlx::core::array> V;
  int seq_filled = 0;
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
               const ::mlc::frontend::GGUFLoader &loader,
               bool use_ane = false);

  struct RunResult {
    std::vector<float> data;
    std::vector<int> shape;
  };
  RunResult run(const std::vector<int32_t> &token_ids,
                const std::vector<int32_t> &positions);

  // Batched decode: N requests, each providing one token per step.
  // token_ids[i] and positions[i] are the current token/position for request i.
  // Returns one RunResult per request (logits over vocab, shape [1, vocab]).
  // Each request has its own KV cache slot maintained across calls.
  // All requests must have the same KV-cache depth (equal-length constraint
  // for V1 — variable-length sequences come in V4 with the scheduler).
  std::vector<RunResult> runBatch(const std::vector<int32_t> &token_ids,
                                  const std::vector<int32_t> &positions);

  // Prefill N requests in one batched forward pass.
  // prompt_ids[i]: token ids for request i (same length required for V1).
  // Returns one RunResult per request, shape [seq_len, vocab].
  std::vector<RunResult> prefillBatch(
      const std::vector<std::vector<int32_t>> &prompt_ids);

  // Drop KV cache for all requests.
  void reset();
  // Drop KV cache for one request slot.
  void resetSlot(int slot_idx);

private:
  ::mlir::ModuleOp module_;
  const ::mlc::frontend::GGUFLoader &loader_;
  // Resident fp16 weights, keyed by GGUF tensor name. Used for non-matmul
  // operands (norm gamma, embedding table, lm_head's Q6_K weight) and as
  // a fallback when a matmul weight isn't in Q4_0.
  std::unordered_map<std::string, mlx::core::array> weight_cache_;
  // MLX native-quantized weights (group_size=32, bits=4, "affine") keyed by
  // GGUF tensor name. Built once at construction from GGUF Q4_0 bytes.
  std::unordered_map<std::string, MLXQuantWeights> mlxq_cache_;
  // Quick lookup: is a given func-arg Value backed by a Q4_0 quantized pack?
  // Populated per-run() from mlxq_cache_; pointer stays valid because the
  // map itself lives for the executor lifetime.
  ::llvm::DenseMap<::mlir::Value, const MLXQuantWeights *> mlxq_value_pkg_;
  // Pre-concatenated (w_q, scales, biases) for the QKV fused op, keyed by
  // Operation*. Holds Q ‖ K ‖ V along the out-row axis.
  std::unordered_map<::mlir::Operation *, MLXQuantWeights> mlxq_qkv_cache_;
  // Pre-concatenated (w_q, scales, biases) for the FFN gate+up op, keyed by
  // Operation*. Holds gate ‖ up along the out-row axis.
  std::unordered_map<::mlir::Operation *, MLXQuantWeights> mlxq_ffngu_cache_;
  // Pre-concatenated W_QKV weights, keyed by the fused_norm_qkv_matmul
  // Operation* that owns them. Used only for non-Q4_0 (fp16) weights.
  std::unordered_map<::mlir::Operation *, mlx::core::array> qkv_concat_cache_;
  // CoreML/ANE matmuls keyed by op. Populated at construction for ops
  // the scheduler marked Device::ANE. Empty when ANE is disabled.
  std::unordered_map<::mlir::Operation *, std::shared_ptr<ANEMatMul>>
      ane_cache_;
  // Whether to actually route ANE-annotated ops to CoreML.
  bool use_ane_ = false;
  // Pre-concatenated W_gate ++ W_up weights, keyed by the mlc.feedforward
  // Operation*. Same idea as `qkv_concat_cache_` — one batched matmul
  // instead of two for the SwiGLU gate/up projections.
  std::unordered_map<::mlir::Operation *, mlx::core::array> ffn_gateup_cache_;
  // KV cache, one slot per mlc.attention op encountered (in walk order).
  // Grows on the first prefill call; updated in place every step.
  std::vector<KVCacheSlot> kv_cache_;
  // Batched KV cache: one inner vector per concurrent request, each with
  // one KVCacheSlot per attention layer. Populated by prefillBatch /
  // runBatch; kv_cache_ is unused when batch_kv_slots_ is active.
  std::vector<std::vector<KVCacheSlot>> batch_kv_slots_;
  // MLC_FUSED_KERNELS=1: use fused multi-op Metal kernels (X1-X3).
  // Reduces dispatch count from ~300 to ~200 per decode step.
  // Raw Q4_0 bytes loaded alongside MLX native format for the fused kernels.
  bool use_fused_kernels_ = false;
  // Per-layer concat'd raw Q4_0 bytes for fused kernels (QKV and gate+up).
  // Populated at construction when use_fused_kernels_=true.
  std::unordered_map<::mlir::Operation *, mlx::core::array> fused_qkv_bytes_cache_;
  std::unordered_map<::mlir::Operation *, std::pair<int,int>> fused_qkv_dims_; // (K, N_total)
  std::unordered_map<::mlir::Operation *, mlx::core::array> fused_gu_bytes_cache_;
  std::unordered_map<::mlir::Operation *, std::pair<int,int>> fused_gu_dims_;  // (K, ffn_dim)
  std::unordered_map<::mlir::Operation *, mlx::core::array> fused_down_bytes_cache_;
  std::unordered_map<::mlir::Operation *, std::pair<int,int>> fused_down_dims_; // (K_ffn, N)

  // MLC_Q4_CUSTOM_KERNEL=1: fall back to the hand-written Q4_0 Metal kernel
  // instead of mx::quantized_matmul. The custom kernel's fp16 output truncation
  // breaks TinyLlama's repetition trap at temperature=0; the native MLX path
  // matches the Python/mlx-lm reference but degenerates on this prompt.
  bool use_custom_q4_ = false;
  // Q4_0 weight bytes (raw GGUF blocks) used by the custom kernel fallback.
  std::unordered_map<std::string, mlx::core::array> q4_bytes_cache_;
  std::unordered_map<std::string, std::pair<int, int>> q4_dims_cache_;
  ::llvm::DenseMap<::mlir::Value, std::pair<int, int>> q4_value_dims_;
  std::unordered_map<::mlir::Operation *, mlx::core::array>
      q4_qkv_concat_cache_;
  std::unordered_map<::mlir::Operation *, mlx::core::array>
      q4_ffngu_concat_cache_;
  // Profiling state — populated when MLC_COMPILER_PROFILE=1.
  // Exposed so run() can accumulate into them and printProfile() can dump.
  bool profile_enabled_ = false;
  int  profile_steps_   = 0;
  // Insertion-ordered keys + per-key (total_ms, call_count) buckets.
  std::vector<std::string> prof_order_;
  std::unordered_map<std::string, std::pair<double, int>> prof_buckets_;

public:
  // Print accumulated profile table to stdout. Call after N decode steps.
  void printProfile() const;
};

} // namespace mlir::mlc::exec
