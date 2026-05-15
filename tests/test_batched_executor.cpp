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

TEST(BatchedExecutorB2c, N2BothRequestsSucceed) {
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
    EXPECT_TRUE(out.per_request[0].success);
    EXPECT_TRUE(out.per_request[1].success);
    EXPECT_FALSE(out.per_request[0].logits.empty());
    EXPECT_FALSE(out.per_request[1].logits.empty());
}

namespace {
double cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}
} // namespace

// B2c: per-request cosine match against single-stream baseline at N=2.
// Two prompts run through DecodeRunner (one each) define the reference;
// the same two prompts run through BatchedExecutor at N=2 must produce
// per-request logits that match cosine >= 0.999 across multiple steps.
//
// Bit-identity is the design intent (sequential per-request execution
// preserves the single-stream code path), but per the validation spec
// we accept >= 0.999 cosine to allow any tiny drift from per-pass state
// ordering (e.g., unordered_map iteration affecting weight cache hit
// patterns). In practice this should be 1.000000.
TEST(BatchedExecutorB2c, N2CosineMatchesSingleStream) {
    if (!tinyLlamaAvailable()) {
        GTEST_SKIP() << "TinyLlama model not present at " << tinyLlamaPath();
    }

    std::vector<uint64_t> prompt_a = {1, 1462, 1554, 304, 263};
    std::vector<uint64_t> prompt_b = {1, 17320, 366, 460, 263};
    constexpr size_t kSteps = 5;
    ASSERT_EQ(prompt_a.size(), kSteps);
    ASSERT_EQ(prompt_b.size(), kSteps);

    auto runReference = [&](const std::vector<uint64_t>& tokens) {
        mlc::runtime::DecodeOptions opts;
        opts.tokens = tokens;
        opts.max_steps = kSteps;
        opts.start_position = 0;
        opts.top_k = 0;
        mlc::runtime::DecodeRunner runner(tinyLlamaPath());
        auto res = runner.run(opts);
        EXPECT_TRUE(res.success);
        return res;
    };

    auto ref_a = runReference(prompt_a);
    auto ref_b = runReference(prompt_b);
    ASSERT_EQ(ref_a.steps.size(), kSteps);
    ASSERT_EQ(ref_b.steps.size(), kSteps);

    // Batched run: N=2 stepping in lockstep.
    mlc::runtime::Session session(tinyLlamaPath());
    mlc::runtime::BatchedExecutor exec(session);
    exec.reset();
    constexpr uint32_t REQ_A = 11;
    constexpr uint32_t REQ_B = 22;

    for (size_t i = 0; i < kSteps; ++i) {
        std::vector<mlc::runtime::RequestSlot> slots = {
            {REQ_A, prompt_a[i], i},
            {REQ_B, prompt_b[i], i},
        };
        auto out = exec.run_decode(slots);
        ASSERT_EQ(out.per_request.size(), 2u);

        const auto& got_a = (out.per_request[0].request_id == REQ_A
                             ? out.per_request[0].logits
                             : out.per_request[1].logits);
        const auto& got_b = (out.per_request[0].request_id == REQ_B
                             ? out.per_request[0].logits
                             : out.per_request[1].logits);
        ASSERT_FALSE(got_a.empty()) << "step=" << i;
        ASSERT_FALSE(got_b.empty()) << "step=" << i;

        double cos_a = cosineSimilarity(got_a, ref_a.steps[i].logits);
        double cos_b = cosineSimilarity(got_b, ref_b.steps[i].logits);
        EXPECT_GE(cos_a, 0.999) << "request A step=" << i << " cos=" << cos_a;
        EXPECT_GE(cos_b, 0.999) << "request B step=" << i << " cos=" << cos_b;

        // Greedy match — top-1 token must agree.
        EXPECT_EQ(argmax(got_a), argmax(ref_a.steps[i].logits))
            << "request A step=" << i;
        EXPECT_EQ(argmax(got_b), argmax(ref_b.steps[i].logits))
            << "request B step=" << i;
    }
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
