#include <gtest/gtest.h>

#include "runtime/scheduler.hpp"
#include "runtime/batched_executor.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using mlc::runtime::IBatchedExecutor;
using mlc::runtime::BatchedDecodeOutput;
using mlc::runtime::RequestSlot;
using mlc::runtime::Scheduler;
using mlc::runtime::GenerationParams;

// Synthetic vocab. For each request id we deterministically generate a
// sequence of tokens (just (request_id * 100) + step). The mock executor
// returns logits as a one-hot at the next token's id so argmax works.
class MockExecutor : public IBatchedExecutor {
public:
    static constexpr size_t kVocab = 1024;
    std::vector<RequestSlot> last_decode_slots;
    std::unordered_map<uint32_t, size_t> decode_calls;
    std::unordered_map<uint32_t, size_t> prefill_calls;
    std::unordered_set<uint32_t> released_requests;

    BatchedDecodeOutput run_decode(const std::vector<RequestSlot>& slots) override {
        last_decode_slots = slots;
        BatchedDecodeOutput out;
        out.per_request.reserve(slots.size());
        for (const auto& s : slots) {
            ++decode_calls[s.request_id];
            BatchedDecodeOutput::PerRequest per;
            per.request_id = s.request_id;
            per.success = true;
            per.logits.assign(kVocab, -1e30f);
            // Next token = request_id * 100 + step (current decode call count)
            uint64_t next_tok = (s.request_id * 100ULL) + decode_calls[s.request_id];
            per.logits[next_tok % kVocab] = 1.0f;
            out.per_request.push_back(std::move(per));
        }
        return out;
    }
    std::vector<float> run_prefill(uint32_t request_id,
                                   const std::vector<uint64_t>& tokens) override {
        ++prefill_calls[request_id];
        std::vector<float> logits(kVocab, -1e30f);
        // Sample first decode token = request_id * 100 + 0 (no decode steps yet).
        uint64_t first_tok = request_id * 100ULL;
        logits[first_tok % kVocab] = 1.0f;
        (void)tokens;
        return logits;
    }
    void release_request(uint32_t request_id) override {
        released_requests.insert(request_id);
    }
};

} // namespace

TEST(SchedulerC1, EmptyQueueIsIdle) {
    MockExecutor mock;
    Scheduler sched(&mock);
    EXPECT_TRUE(sched.empty());
    EXPECT_EQ(sched.run_until_idle(), 0u);
}

TEST(SchedulerC1, SingleRequestRunsToCompletion) {
    MockExecutor mock;
    Scheduler sched(&mock, /*max_batch_size=*/8);

    GenerationParams params;
    params.max_new_tokens = 5;
    uint32_t id = sched.add_request({1, 2, 3}, params);
    EXPECT_EQ(id, 1u);
    EXPECT_EQ(sched.queued_count(), 1u);

    std::vector<uint64_t> emitted;
    sched.set_token_callback([&](uint32_t /*req*/, uint64_t tok) { emitted.push_back(tok); });

    sched.run_until_idle();

    EXPECT_TRUE(sched.empty());
    EXPECT_EQ(sched.completed_count(), 1u);
    EXPECT_EQ(emitted.size(), 5u);  // first from prefill + 4 from decode
    EXPECT_EQ(mock.prefill_calls[1], 1u);
    EXPECT_EQ(mock.decode_calls[1], 4u);  // 5 total tokens, 1 from prefill = 4 decodes
    EXPECT_EQ(mock.released_requests.count(1), 1u);

    auto drained = sched.drain_completed();
    ASSERT_EQ(drained.size(), 1u);
    EXPECT_EQ(drained[0].state, Scheduler::State::Complete);
    EXPECT_EQ(drained[0].generated_tokens.size(), 5u);
}

TEST(SchedulerC1, MultipleRequestsBatchedTogether) {
    MockExecutor mock;
    Scheduler sched(&mock, /*max_batch_size=*/4);

    GenerationParams params;
    params.max_new_tokens = 3;
    uint32_t a = sched.add_request({1, 2}, params);
    uint32_t b = sched.add_request({3, 4}, params);
    uint32_t c = sched.add_request({5, 6}, params);

    sched.run_until_idle();

    EXPECT_EQ(sched.completed_count(), 3u);
    EXPECT_EQ(mock.prefill_calls[a], 1u);
    EXPECT_EQ(mock.prefill_calls[b], 1u);
    EXPECT_EQ(mock.prefill_calls[c], 1u);
    EXPECT_EQ(mock.decode_calls[a], 2u);
    EXPECT_EQ(mock.decode_calls[b], 2u);
    EXPECT_EQ(mock.decode_calls[c], 2u);
    EXPECT_EQ(mock.released_requests.size(), 3u);
    // Stats sanity.
    EXPECT_EQ(sched.stats().total_requests_completed, 3u);
    // 3 reqs × (1 prefill token + 2 decode tokens) = 9.
    EXPECT_EQ(sched.stats().total_tokens_generated, 9u);
}

TEST(SchedulerC1, MaxBatchSizeBoundsActiveSet) {
    MockExecutor mock;
    Scheduler sched(&mock, /*max_batch_size=*/2);

    GenerationParams params;
    params.max_new_tokens = 1;
    sched.add_request({1}, params);
    sched.add_request({2}, params);
    sched.add_request({3}, params);
    sched.add_request({4}, params);

    // First tick: admits at most 2.
    sched.tick();
    EXPECT_LE(sched.active_count(), 2u);

    // Run remaining; total completed must be 4.
    sched.run_until_idle();
    EXPECT_EQ(sched.completed_count(), 4u);
    EXPECT_TRUE(sched.empty());
}

TEST(SchedulerC1, EOSCompletesEarly) {
    MockExecutor mock;
    Scheduler sched(&mock);

    GenerationParams params;
    params.max_new_tokens = 100;
    // Request id will be 1; first generated token = 1*100 + 0 = 100. Set EOS = 100.
    params.eos_token_id = 100;

    sched.add_request({1}, params);
    sched.run_until_idle();

    EXPECT_EQ(sched.completed_count(), 1u);
    auto drained = sched.drain_completed();
    ASSERT_EQ(drained.size(), 1u);
    // Prefill emitted token 100 (= eos), so EOS check fires immediately and
    // no decode step runs. generated_tokens contains the single EOS token.
    EXPECT_EQ(drained[0].generated_tokens.size(), 1u);
    EXPECT_EQ(drained[0].generated_tokens[0], 100u);
    EXPECT_EQ(mock.decode_calls[1], 0u);
}

TEST(SchedulerC1, IterationLevelAdmissionFillsVacancies) {
    MockExecutor mock;
    Scheduler sched(&mock, /*max_batch_size=*/2);

    GenerationParams short_params;
    short_params.max_new_tokens = 1;  // completes after first decode
    GenerationParams long_params;
    long_params.max_new_tokens = 5;

    // Two short + one long. After short ones complete, the long one should
    // remain active and the queue should empty.
    sched.add_request({1}, short_params);
    sched.add_request({2}, short_params);
    sched.add_request({3}, long_params);

    size_t total_steps = sched.run_until_idle();
    EXPECT_GT(total_steps, 0u);
    EXPECT_EQ(sched.completed_count(), 3u);
    EXPECT_TRUE(sched.empty());
}
