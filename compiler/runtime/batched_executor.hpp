#pragma once

// BatchedExecutor — top-level runtime for continuous batching.
//
// Phase B1 sets up the API surface and an N=1 implementation that
// delegates to the existing single-stream ExecutionExecutor + ExecutionContext.
// At N=1 the output is bit-identical to DecodeRunner; future Phase B2/B3
// commits replace internals with batched dispatches and paged KV.
//
// The class manages its own ExecutionGraph + ExecutionContext + ExecutionExecutor
// for the duration of its life — request state is held in this BatchedExecutor,
// not in the surrounding scheduler. The scheduler (Phase C) drives admission
// and decoding; BatchedExecutor handles the per-pass mechanics.

#include "runtime/session.hpp"
#include "runtime/execution_graph.hpp"
#include "runtime/execution_context.hpp"
#include "runtime/execution_executor.hpp"

#include <cstdint>
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
    // Construct over a previously loaded Session. The graph is built once;
    // context is owned by this executor and reset on demand.
    explicit BatchedExecutor(const Session& session);

    // Run one decoder pass for the given slots. Currently supports batch
    // size 1 only; N>1 returns an empty output (will be wired in B2).
    BatchedDecodeOutput run_decode(const std::vector<RequestSlot>& slots);

    // Run prefill for a single request: process `tokens` through the model
    // sequentially (token-by-token; multi-token batched prefill is a v2
    // optimization), populating the request's KV state. Returns logits at
    // the last token. The request_id is used for future per-request KV
    // partitioning; in B1 (single-stream backing) it's recorded but not
    // routed.
    std::vector<float> run_prefill(uint32_t request_id,
                                   const std::vector<uint64_t>& tokens);

    // Reset the internal context (clears KV cache, position trackers).
    // Used by the test harness and between independent runs.
    void reset();

    // Accessors for testing / debugging.
    const ExecutionGraph& graph() const { return graph_; }
    size_t known_position(uint32_t request_id) const;

private:
    const Session& session_;
    ExecutionGraph graph_;
    ExecutionContext context_;
    ExecutionExecutor executor_;
    std::unordered_map<uint32_t, size_t> request_positions_;
};

} // namespace runtime
} // namespace mlc
