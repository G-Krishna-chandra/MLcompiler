#include <gtest/gtest.h>

#include "runtime/batched_executor.hpp"
#include "runtime/scheduler.hpp"
#include "runtime/session.hpp"

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace {

std::string tinyLlamaPath() {
    return "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
}
bool tinyLlamaAvailable() {
    return std::filesystem::exists(tinyLlamaPath());
}

uint64_t argmax(const std::vector<float>& logits) {
    auto it = std::max_element(logits.begin(), logits.end());
    return static_cast<uint64_t>(std::distance(logits.begin(), it));
}

// Produce the greedy continuation of `prompt` for `max_new_tokens` steps
// using BatchedExecutor at N=1. Returns the generated token list.
std::vector<uint64_t> greedyReference(mlc::runtime::Session& session,
                                      const std::vector<uint64_t>& prompt,
                                      size_t max_new_tokens) {
    mlc::runtime::BatchedExecutor exec(session);
    exec.reset();
    constexpr uint32_t kRefId = 0xCAFE;
    auto first = exec.run_prefill(kRefId, prompt);
    if (first.empty()) return {};

    std::vector<uint64_t> generated;
    uint64_t tok = argmax(first);
    generated.push_back(tok);

    for (size_t step = 1; step < max_new_tokens; ++step) {
        std::vector<mlc::runtime::RequestSlot> slot{
            {kRefId, tok, prompt.size() + step - 1}};
        auto out = exec.run_decode(slot);
        if (out.per_request.empty() || !out.per_request[0].success) break;
        tok = argmax(out.per_request[0].logits);
        generated.push_back(tok);
    }
    return generated;
}

} // namespace

TEST(SchedulerC2, FourPromptsEndToEndGreedyMatch) {
    if (!tinyLlamaAvailable()) {
        GTEST_SKIP() << "TinyLlama model not present at " << tinyLlamaPath();
    }

    constexpr size_t kMaxNew = 8;

    // 4 distinct token sequences chosen for diversity. Real prompts are
    // tokenized in D1; here we feed raw ids to keep the test minimal.
    std::vector<std::vector<uint64_t>> prompts = {
        {1, 1462, 1554, 304, 263},   // "Hello, ..." style
        {1, 17320, 366, 460, 263},   // alt tokens
        {1, 415, 7483, 310, 3444},   // "The capital of France"
        {1, 17320, 590, 1024, 5181}, // unrelated
    };

    // Reference outputs: greedy continuation per prompt via BatchedExecutor
    // at N=1 (which is bit-identical to DecodeRunner per B1).
    std::vector<std::vector<uint64_t>> refs;
    {
        mlc::runtime::Session ref_session(tinyLlamaPath());
        for (const auto& p : prompts) {
            refs.push_back(greedyReference(ref_session, p, kMaxNew));
            ASSERT_EQ(refs.back().size(), kMaxNew)
                << "greedy reference produced fewer tokens than expected";
        }
    }

    // Scheduler-driven batched run.
    mlc::runtime::Session session(tinyLlamaPath());
    mlc::runtime::BatchedExecutor exec(session);
    mlc::runtime::Scheduler sched(&exec, /*max_batch_size=*/4);

    mlc::runtime::GenerationParams params;
    params.max_new_tokens = kMaxNew;

    std::vector<uint32_t> ids;
    for (const auto& p : prompts) {
        ids.push_back(sched.add_request(p, params));
    }

    sched.run_until_idle();
    auto completed = sched.drain_completed();
    ASSERT_EQ(completed.size(), prompts.size());

    // Lookup completed by id for stable per-prompt comparison.
    std::vector<mlc::runtime::Scheduler::Request*> by_id(prompts.size());
    for (auto& req : completed) {
        for (size_t i = 0; i < ids.size(); ++i) {
            if (req.id == ids[i]) by_id[i] = &req;
        }
    }
    for (size_t i = 0; i < prompts.size(); ++i) {
        ASSERT_NE(by_id[i], nullptr) << "request id " << ids[i] << " not in completed";
        EXPECT_EQ(by_id[i]->state, mlc::runtime::Scheduler::State::Complete);
        ASSERT_EQ(by_id[i]->generated_tokens.size(), kMaxNew)
            << "prompt " << i << " generated " << by_id[i]->generated_tokens.size()
            << " tokens (expected " << kMaxNew << ")";
        for (size_t t = 0; t < kMaxNew; ++t) {
            EXPECT_EQ(by_id[i]->generated_tokens[t], refs[i][t])
                << "prompt=" << i << " step=" << t
                << " batched=" << by_id[i]->generated_tokens[t]
                << " ref=" << refs[i][t];
        }
    }

    EXPECT_EQ(sched.stats().total_requests_completed, prompts.size());
    // 4 requests × 8 tokens each (1 prefill + 7 decode) = 32 tokens.
    EXPECT_EQ(sched.stats().total_tokens_generated, prompts.size() * kMaxNew);
}
