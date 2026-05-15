#include <gtest/gtest.h>

#include "runtime/batched_executor.hpp"
#include "runtime/decode_runner.hpp"
#include "runtime/session.hpp"

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace {

// Path to the TinyLlama Q4_0 model used for the bit-identity check.
// If absent (CI / sandboxed), the test skips. We don't ship the model in
// this repo; production benchmarks fetch it locally.
std::string tinyLlamaPath() {
    return "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
}

bool tinyLlamaAvailable() {
    return std::filesystem::exists(tinyLlamaPath());
}

// Greedy: argmax over logits.
uint64_t argmax(const std::vector<float>& logits) {
    auto it = std::max_element(logits.begin(), logits.end());
    return static_cast<uint64_t>(std::distance(logits.begin(), it));
}

} // namespace

TEST(BatchedExecutorB1, EmptySlotsReturnsEmpty) {
    if (!tinyLlamaAvailable()) {
        GTEST_SKIP() << "TinyLlama model not present at " << tinyLlamaPath();
    }
    mlc::runtime::Session session(tinyLlamaPath());
    mlc::runtime::BatchedExecutor exec(session);
    auto out = exec.run_decode({});
    EXPECT_TRUE(out.per_request.empty());
}

TEST(BatchedExecutorB1, NGreaterThanOneReturnsFailureSentinels) {
    if (!tinyLlamaAvailable()) {
        GTEST_SKIP() << "TinyLlama model not present at " << tinyLlamaPath();
    }
    mlc::runtime::Session session(tinyLlamaPath());
    mlc::runtime::BatchedExecutor exec(session);
    std::vector<mlc::runtime::RequestSlot> slots = {
        {1, 100, 0},
        {2, 200, 0},
    };
    auto out = exec.run_decode(slots);
    ASSERT_EQ(out.per_request.size(), 2u);
    EXPECT_FALSE(out.per_request[0].success);
    EXPECT_FALSE(out.per_request[1].success);
}

// Bit-identity check: for the same sequence of input tokens at the same
// positions, BatchedExecutor at N=1 must produce identical logits to a
// fresh DecodeRunner. Both share the same underlying ExecutionExecutor
// + ExecutionContext machinery, so this is true by construction; the
// test guards the construction.
TEST(BatchedExecutorB1, BitIdenticalToDecodeRunnerAtN1) {
    if (!tinyLlamaAvailable()) {
        GTEST_SKIP() << "TinyLlama model not present at " << tinyLlamaPath();
    }

    // A handful of arbitrary-but-deterministic input token IDs covering
    // the first few decode steps. Position 0 is treated as the first
    // pre-prompt token for both runners.
    std::vector<uint64_t> tokens = {1, 1462, 1554, 304, 263};  // BOS + a few words
    constexpr size_t kSteps = 5;

    // Reference: DecodeRunner.
    mlc::runtime::DecodeOptions opts;
    opts.tokens = tokens;
    opts.max_steps = kSteps;
    opts.start_position = 0;
    opts.top_k = 0;

    mlc::runtime::DecodeRunner ref_runner(tinyLlamaPath());
    auto ref_result = ref_runner.run(opts);
    ASSERT_TRUE(ref_result.success) << "reference DecodeRunner failed";
    ASSERT_EQ(ref_result.steps.size(), kSteps);

    // Test path: BatchedExecutor at N=1, fed token-by-token like DecodeRunner.
    mlc::runtime::Session session(tinyLlamaPath());
    mlc::runtime::BatchedExecutor exec(session);
    exec.reset();

    for (size_t i = 0; i < kSteps; ++i) {
        std::vector<mlc::runtime::RequestSlot> slot{{
            /*request_id=*/0,
            /*input_token=*/tokens[i],
            /*sequence_position=*/i,
        }};
        auto out = exec.run_decode(slot);
        ASSERT_EQ(out.per_request.size(), 1u);
        ASSERT_TRUE(out.per_request[0].success) << "step=" << i;

        const auto& batched_logits = out.per_request[0].logits;
        const auto& ref_logits     = ref_result.steps[i].logits;
        ASSERT_EQ(batched_logits.size(), ref_logits.size())
            << "logits size mismatch at step=" << i;

        // Bit-identical: BatchedExecutor at N=1 delegates to the same
        // ExecutionExecutor::run() the DecodeRunner uses, with identical
        // context state. Any drift here is a bug.
        for (size_t j = 0; j < ref_logits.size(); ++j) {
            ASSERT_EQ(batched_logits[j], ref_logits[j])
                << "step=" << i << " logit_idx=" << j
                << " batched=" << batched_logits[j]
                << " ref=" << ref_logits[j];
        }

        // Greedy match (sanity).
        EXPECT_EQ(argmax(batched_logits), argmax(ref_logits)) << "step=" << i;
    }
}
