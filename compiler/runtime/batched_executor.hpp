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
#include "runtime/paged_kv.hpp"

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

// IBatchedExecutor — abstract interface the Scheduler depends on. Lets unit
// tests inject a mock without standing up the full Metal-backed runtime.
class IBatchedExecutor {
public:
    virtual ~IBatchedExecutor() = default;
    virtual BatchedDecodeOutput run_decode(const std::vector<RequestSlot>& slots) = 0;
    virtual std::vector<float>  run_prefill(uint32_t request_id,
                                            const std::vector<uint64_t>& tokens) = 0;
    virtual void release_request(uint32_t request_id) = 0;
};

class BatchedExecutor : public IBatchedExecutor {
public:
    explicit BatchedExecutor(const Session& session);

    // Phase B3 — attach a PagePool. When attached, BatchedExecutor maintains
    // a RequestKVState per request and extends each request's page table by
    // one slot per decode step (one page allocated per page_size_tokens
    // tokens). The actual K/V data continues to flow through each request's
    // contiguous ExecutionContext for now; the page table tracks ownership
    // and provides the lifecycle hook the scheduler (Phase C) will consume.
    // Attaching with no pool (default) skips paged tracking entirely.
    void attach_page_pool(PagePool* pool, uint32_t page_size_tokens = 64);

    // Read-only accessor for the per-request page table (paged mode only).
    // Returns nullptr when paged tracking is off or the request isn't tracked.
    const RequestKVState* page_state(uint32_t request_id) const;

    // Run one decoder pass per slot. Each slot's KV state is isolated in a
    // per-request ExecutionContext. Slots run sequentially within this
    // call (Option a from the design doc).
    BatchedDecodeOutput run_decode(const std::vector<RequestSlot>& slots) override;

    // Run prefill for a single request: process `tokens` through the
    // request's context one token at a time, returning logits at the
    // last position.
    std::vector<float> run_prefill(uint32_t request_id,
                                   const std::vector<uint64_t>& tokens) override;

    // Drop all per-request state (KV cache + position trackers +
    // executor/context objects). Useful for tests; in production the
    // scheduler removes individual requests via release_request.
    void reset();

    // Drop a single request's state and free its context.
    void release_request(uint32_t request_id) override;

    const ExecutionGraph& graph() const { return graph_; }
    size_t known_position(uint32_t request_id) const;
    size_t live_request_count() const { return requests_.size(); }

private:
    struct PerRequest {
        std::unique_ptr<ExecutionContext> context;
        std::unique_ptr<ExecutionExecutor> executor;
        size_t next_position = 0;
        // B3: optional paged-KV page table for this request. Populated only
        // when a PagePool is attached and grown on each successful decode
        // step. Released back to the pool on release_request / reset.
        std::unique_ptr<RequestKVState> page_state;
    };

    PerRequest& ensure_request(uint32_t request_id);
    void prime_token_tensors(ExecutionContext& ctx, uint64_t token) const;
    bool advance_page_state(PerRequest& req);

    const Session& session_;
    ExecutionGraph graph_;
    std::unordered_map<uint32_t, PerRequest> requests_;

    // B3 paged-KV state. Optional; nullptr means paged tracking is off.
    PagePool* page_pool_ = nullptr;
    uint32_t page_size_tokens_ = 64;
};

} // namespace runtime
} // namespace mlc
