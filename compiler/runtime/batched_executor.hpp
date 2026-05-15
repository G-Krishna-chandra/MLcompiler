#pragma once

// BatchedExecutor — top-level runtime for continuous batching.
//
// Phase B1 set up the API surface and an N=1 implementation that delegated
// to the existing single-stream ExecutionExecutor + ExecutionContext.
//
// Phase B2c upgrades to N>=1 by holding per-request ExecutionContext +
// ExecutionExecutor state. At N=1 the behavior is unchanged (and remains
// bit-identical to DecodeRunner). At N>1 each request runs sequentially
// with its own isolated KV state, so per-request output matches what the
// single-stream path would produce — Option (a) "sequential per-request
// attention" from the design doc.
//
// Phase B3 will wire paged-KV storage into the per-request contexts.
// Phase E may add cross-request batching for matmul / attention to capture
// compute-density wins beyond CB-sharing amortization.

#include "runtime/session.hpp"
#include "runtime/execution_graph.hpp"
#include "runtime/execution_context.hpp"
#include "runtime/execution_executor.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace mlc {
namespace runtime {

struct RequestSlot {
    uint32_t request_id = 0;
    uint64_t input_token = 0;
    size_t   sequence_position = 0;
};

struct BatchedDecodeOutput {
    struct PerRequest {
        uint32_t request_id = 0;
        bool success = false;
        std::vector<float> logits;  // empty on failure
    };
    std::vector<PerRequest> per_request;
};

class BatchedExecutor {
public:
    explicit BatchedExecutor(const Session& session);

    // Run one decoder pass per slot. Each slot's KV state is isolated in a
    // per-request ExecutionContext. Slots run sequentially within this
    // call (Option a from the design doc).
    BatchedDecodeOutput run_decode(const std::vector<RequestSlot>& slots);

    // Run prefill for a single request: process `tokens` through the
    // request's context one token at a time, returning logits at the
    // last position.
    std::vector<float> run_prefill(uint32_t request_id,
                                   const std::vector<uint64_t>& tokens);

    // Drop all per-request state (KV cache + position trackers +
    // executor/context objects). Useful for tests; in production the
    // scheduler removes individual requests via release_request.
    void reset();

    // Drop a single request's state and free its context.
    void release_request(uint32_t request_id);

    const ExecutionGraph& graph() const { return graph_; }
    size_t known_position(uint32_t request_id) const;
    size_t live_request_count() const { return requests_.size(); }

private:
    struct PerRequest {
        std::unique_ptr<ExecutionContext> context;
        std::unique_ptr<ExecutionExecutor> executor;
        size_t next_position = 0;
    };

    PerRequest& ensure_request(uint32_t request_id);
    void prime_token_tensors(ExecutionContext& ctx, uint64_t token) const;

    const Session& session_;
    ExecutionGraph graph_;
    std::unordered_map<uint32_t, PerRequest> requests_;
};

} // namespace runtime
} // namespace mlc
