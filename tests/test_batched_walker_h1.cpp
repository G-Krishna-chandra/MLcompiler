#include <gtest/gtest.h>

#include "runtime/batched_walker.hpp"
#include "runtime/batched_executor.hpp"
#include "runtime/decode_runner.hpp"
#include "runtime/paged_kv.hpp"
#include "runtime/session.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>

namespace {

std::string tinyLlamaPath() { return "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"; }
bool tinyLlamaAvailable() { return std::filesystem::exists(tinyLlamaPath()); }

double cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na == 0 || nb == 0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

struct Fixture {
    mlc::runtime::Session session;
    mlc::runtime::BatchedExecutor exec;
    mlc::runtime::PagePool pool;
    std::unique_ptr<mlc::runtime::PagedKVStorage> storage;

    Fixture(uint32_t capacity_pages, uint32_t page_size_tokens)
        : session(tinyLlamaPath()),
          exec(session),
          pool(capacity_pages) {
        const auto& g = exec.graph();
        size_t n_kv_heads = 0, head_dim = 0, n_layers = 0;
        for (const auto& n : g.nodes()) {
            if (n.op == mlc::runtime::ExecOpType::Attention) {
                if (n_kv_heads == 0) {
                    n_kv_heads = static_cast<size_t>(n.attributes.at("kv_heads"));
                    head_dim   = static_cast<size_t>(n.attributes.at("head_dim"));
                }
                ++n_layers;
            }
        }
        storage = std::make_unique<mlc::runtime::PagedKVStorage>(
            capacity_pages, n_layers, page_size_tokens, n_kv_heads, head_dim, 2);
        if (!storage->initialize()) throw std::runtime_error("storage init failed");
        exec.attach_page_pool(&pool, page_size_tokens);
        exec.attach_paged_storage(storage.get());
    }
};

} // namespace

TEST(BatchedWalkerH1, N1FirstStepCosineMatchesDecodeRunner) {
    if (!tinyLlamaAvailable()) GTEST_SKIP() << "TinyLlama model missing";

    std::vector<uint64_t> tokens = {1, 1462, 1554, 304, 263};

    // Reference: DecodeRunner all 5 steps.
    mlc::runtime::DecodeOptions opts;
    opts.tokens = tokens;
    opts.max_steps = tokens.size();
    opts.start_position = 0;
    opts.top_k = 0;
    mlc::runtime::DecodeRunner runner(tinyLlamaPath());
    auto ref = runner.run(opts);
    ASSERT_TRUE(ref.success);

    // Test: BatchedWalker, single request, run tokens one at a time.
    Fixture fx(/*pages=*/32, /*pgsize=*/64);
    mlc::runtime::BatchedWalker walker(fx.exec);

    constexpr uint32_t REQ = 100;
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::vector<mlc::runtime::RequestSlot> slots = {{REQ, tokens[i], i}};
        auto out = walker.step(slots);
        ASSERT_EQ(out.per_request.size(), 1u);
        ASSERT_TRUE(out.per_request[0].success) << "step=" << i;
        ASSERT_FALSE(out.per_request[0].logits.empty()) << "step=" << i;

        double cos = cosineSimilarity(out.per_request[0].logits, ref.steps[i].logits);
        EXPECT_GE(cos, 0.999) << "step=" << i << " cos=" << cos;
    }
}

TEST(BatchedWalkerH1, N2EachRequestCosineMatch) {
    if (!tinyLlamaAvailable()) GTEST_SKIP() << "TinyLlama model missing";

    std::vector<uint64_t> tokens_a = {1, 1462, 1554, 304, 263};
    std::vector<uint64_t> tokens_b = {1, 17320, 366, 460, 263};
    constexpr size_t kSteps = 5;

    auto runRef = [&](const std::vector<uint64_t>& toks) {
        mlc::runtime::DecodeOptions opts;
        opts.tokens = toks;
        opts.max_steps = kSteps;
        opts.start_position = 0;
        opts.top_k = 0;
        mlc::runtime::DecodeRunner r(tinyLlamaPath());
        return r.run(opts);
    };
    auto ref_a = runRef(tokens_a);
    auto ref_b = runRef(tokens_b);
    ASSERT_TRUE(ref_a.success && ref_b.success);

    Fixture fx(/*pages=*/32, /*pgsize=*/64);
    mlc::runtime::BatchedWalker walker(fx.exec);
    constexpr uint32_t REQ_A = 11, REQ_B = 22;

    for (size_t i = 0; i < kSteps; ++i) {
        std::vector<mlc::runtime::RequestSlot> slots = {
            {REQ_A, tokens_a[i], i},
            {REQ_B, tokens_b[i], i},
        };
        auto out = walker.step(slots);
        ASSERT_EQ(out.per_request.size(), 2u);

        const std::vector<float>* got_a = nullptr;
        const std::vector<float>* got_b = nullptr;
        for (const auto& per : out.per_request) {
            if (per.request_id == REQ_A) got_a = &per.logits;
            else if (per.request_id == REQ_B) got_b = &per.logits;
        }
        ASSERT_NE(got_a, nullptr);
        ASSERT_NE(got_b, nullptr);
        EXPECT_GE(cosineSimilarity(*got_a, ref_a.steps[i].logits), 0.999) << "A step=" << i;
        EXPECT_GE(cosineSimilarity(*got_b, ref_b.steps[i].logits), 0.999) << "B step=" << i;
    }
}

TEST(BatchedWalkerH1, N4EachRequestCosineMatch) {
    if (!tinyLlamaAvailable()) GTEST_SKIP() << "TinyLlama model missing";

    std::vector<std::vector<uint64_t>> prompts = {
        {1, 1462, 1554, 304, 263},
        {1, 17320, 366, 460, 263},
        {1, 415, 7483, 310, 3444},
        {1, 17320, 590, 1024, 5181},
    };
    constexpr size_t kSteps = 5;
    std::vector<mlc::runtime::DecodeResult> refs;
    for (const auto& p : prompts) {
        mlc::runtime::DecodeOptions opts;
        opts.tokens = p;
        opts.max_steps = kSteps;
        opts.start_position = 0;
        opts.top_k = 0;
        mlc::runtime::DecodeRunner r(tinyLlamaPath());
        refs.push_back(r.run(opts));
        ASSERT_TRUE(refs.back().success);
    }

    Fixture fx(/*pages=*/64, /*pgsize=*/64);
    mlc::runtime::BatchedWalker walker(fx.exec);

    std::vector<uint32_t> ids = {31, 32, 33, 34};
    for (size_t i = 0; i < kSteps; ++i) {
        std::vector<mlc::runtime::RequestSlot> slots;
        for (size_t r = 0; r < prompts.size(); ++r) {
            slots.push_back({ids[r], prompts[r][i], i});
        }
        auto out = walker.step(slots);
        ASSERT_EQ(out.per_request.size(), prompts.size());
        for (size_t r = 0; r < prompts.size(); ++r) {
            const std::vector<float>* got = nullptr;
            for (const auto& per : out.per_request) {
                if (per.request_id == ids[r]) { got = &per.logits; break; }
            }
            ASSERT_NE(got, nullptr) << "req=" << r << " step=" << i;
            double cos = cosineSimilarity(*got, refs[r].steps[i].logits);
            EXPECT_GE(cos, 0.999) << "req=" << r << " step=" << i << " cos=" << cos;
        }
    }
}
