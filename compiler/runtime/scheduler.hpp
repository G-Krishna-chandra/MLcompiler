#pragma once

// Scheduler — request lifecycle + iteration-level batching.
//
// Phase C of continuous batching. Holds a queue of pending requests, an
// active set being processed, and a completed set ready to be reaped.
// Each iteration of run_until_idle:
//   1. Admits queued requests into active until max_batch_size.
//   2. Runs prefill on any newly admitted request (one at a time for v1).
//   3. Runs one batched decode step across all DECODING requests.
//   4. Samples each request's next token (greedy v1), emits via callback,
//      checks stop condition (max_new_tokens / EOS).
//   5. Reaps completed requests (releases their KV state).
//
// FIFO admission, no preemption, no eviction, decode-only batches in v1.

#include "runtime/batched_executor.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <vector>

namespace mlc {
namespace runtime {

struct GenerationParams {
    size_t max_new_tokens = 100;
    int64_t eos_token_id  = -1;  // -1 disables EOS check
};

struct SchedulerStats {
    size_t total_requests_completed = 0;
    size_t total_decode_steps       = 0;
    size_t total_tokens_generated   = 0;
};

class Scheduler {
public:
    enum class State {
        Queued,
        Prefilling,
        Decoding,
        Complete,
        Failed,
    };

    struct Request {
        uint32_t id;
        std::vector<uint64_t> prompt_tokens;
        GenerationParams params;
        State state = State::Queued;
        std::vector<uint64_t> generated_tokens;
        size_t position = 0;             // next sequence position to feed
        uint64_t last_token = 0;          // most-recently-sampled token
        std::string failure_reason;
    };

    using TokenCallback = std::function<void(uint32_t request_id, uint64_t token)>;
    using CompleteCallback = std::function<void(const Request& req)>;

    Scheduler(IBatchedExecutor* executor, size_t max_batch_size = 8);

    // Add a request to the queue. Returns the assigned request id.
    uint32_t add_request(std::vector<uint64_t> prompt_tokens, GenerationParams params);

    void set_token_callback(TokenCallback cb);
    void set_complete_callback(CompleteCallback cb);

    // Drive the scheduler loop until both the queue and active set are
    // empty. Returns the number of decode steps executed.
    size_t run_until_idle();

    // Single-tick: admit + prefill + one decode + reap. Returns true if
    // any work was done. Useful for tests that want to inspect state
    // between ticks.
    bool tick();

    bool empty() const;
    size_t queued_count() const { return queued_.size(); }
    size_t active_count() const { return active_.size(); }
    size_t completed_count() const { return completed_.size(); }
    const SchedulerStats& stats() const { return stats_; }

    // Drain completed requests. Caller takes ownership.
    std::vector<Request> drain_completed();

private:
    void admit_new();
    void prefill_one_active();
    void decode_active();
    void reap_completed();
    static uint64_t argmax(const std::vector<float>& logits);

    IBatchedExecutor* executor_;
    size_t max_batch_size_;
    std::deque<Request> queued_;
    std::vector<Request> active_;
    std::vector<Request> completed_;
    TokenCallback token_cb_;
    CompleteCallback complete_cb_;
    uint32_t next_id_ = 1;
    SchedulerStats stats_;
};

} // namespace runtime
} // namespace mlc
