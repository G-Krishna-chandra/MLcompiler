#pragma once

// BatchedWalker — op-by-op batched per-pass execution for N requests.
//
// Phase H1 of continuous batching v2. Sits alongside the per-request
// ExecutionExecutor and the chunked-walk path (G1). BatchedWalker
// dispatches every per-op kernel ONCE across all N requests' inputs:
//
//   For each op in the per-layer sequence:
//     1. gather: stack N requests' input tensors into a flat [N, dim] buffer
//     2. dispatch: call the batched kernel variant
//     3. scatter: split the [N, dim] output into per-request context tensors
//
// The walker's op sequence is hardcoded for the TinyLlama IR shape
// (RMSNorm → QKV-fused matmul → slices → attention → attn_output → Add →
//  RMSNorm → gate_up-fused matmul → slices → SiLU*mul → down → Add).
// Generalizing to any IR is a follow-up arc.
//
// Single-stream path (DecodeRunner / chat-repl) is untouched; this is a
// new code path enabled when paged storage is attached.

#include "runtime/batched_executor.hpp"
#include "runtime/execution_context.hpp"
#include "runtime/execution_graph.hpp"
#include "runtime/paged_kv.hpp"
#include "runtime/session.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mlc {
namespace runtime {

class BatchedWalker {
public:
    explicit BatchedWalker(BatchedExecutor& exec);

    // Run one decode step for N requests. Returns logits per request.
    BatchedDecodeOutput step(const std::vector<RequestSlot>& slots);

private:
    BatchedExecutor& exec_;
    const Session& session_;
    const ExecutionGraph& graph_;

    // Cached model shape (set on first step).
    bool shape_cached_ = false;
    size_t hidden_size_ = 0;
    size_t num_heads_ = 0;
    size_t kv_heads_ = 0;
    size_t head_dim_ = 0;
    size_t q_rows_ = 0;
    size_t k_rows_ = 0;
    size_t v_rows_ = 0;
    size_t qkv_rows_ = 0;
    size_t ffn_inner_ = 0;
    size_t ffn_gate_up_rows_ = 0;
    size_t vocab_size_ = 0;
    size_t n_layers_ = 0;
    size_t rotary_dim_ = 0;
    float rope_base_ = 10000.0f;
    float rope_scale_ = 1.0f;

    // Per-pass scratch (resized on demand).
    std::vector<float> scratch_a_;
    std::vector<float> scratch_b_;
    std::vector<float> scratch_c_;

    void cache_shape();
    void prepare_pages_for_pass(const std::vector<BatchedExecutor::PerRequest*>& reqs,
                                std::vector<uint32_t>& page_id_per_req,
                                std::vector<uint32_t>& slot_per_req);

    void gather(const std::vector<ExecutionContext*>& ctxs,
                const std::string& tensor_name,
                size_t per_req_dim,
                std::vector<float>& out) const;
    void scatter(const std::vector<ExecutionContext*>& ctxs,
                 const std::string& tensor_name,
                 size_t per_req_dim,
                 const std::vector<float>& in) const;

    // Per-op batched dispatches.
    void op_embedding(const std::vector<ExecutionContext*>& ctxs,
                      const std::vector<uint64_t>& tokens);
    void op_norm(const std::vector<ExecutionContext*>& ctxs,
                 const std::string& in_name,
                 const std::string& out_name,
                 const std::string& weight_name,
                 size_t dim);
    void op_q4_matmul(const std::vector<ExecutionContext*>& ctxs,
                      const std::string& in_name,
                      const std::string& out_name,
                      const std::string& weight_name,
                      size_t in_dim,
                      size_t out_dim);
    void op_attention(const std::vector<ExecutionContext*>& ctxs,
                      const std::vector<RequestSlot>& slots,
                      const std::vector<BatchedExecutor::PerRequest*>& reqs,
                      const std::vector<uint32_t>& page_ids,
                      const std::vector<uint32_t>& slot_ids,
                      size_t layer_idx);
    void op_add(const std::vector<ExecutionContext*>& ctxs,
                const std::string& a_name,
                const std::string& b_name,
                const std::string& out_name,
                size_t dim);
    void op_silu_mul(const std::vector<ExecutionContext*>& ctxs,
                     const std::string& gate_name,
                     const std::string& up_name,
                     const std::string& out_name,
                     size_t dim);
    void op_slice(const std::vector<ExecutionContext*>& ctxs,
                  const std::string& src_name,
                  const std::string& dst_name,
                  size_t src_offset,
                  size_t length);
    void op_lm_head(const std::vector<ExecutionContext*>& ctxs,
                    const std::string& in_name,
                    const std::string& out_name);
};

} // namespace runtime
} // namespace mlc
