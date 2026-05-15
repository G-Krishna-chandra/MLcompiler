#include <gtest/gtest.h>

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

uint64_t argmax(const std::vector<float>& v) {
    return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

} // namespace

// G1: BatchedExecutor with paged-flash attention should produce per-request
// logits with cosine >= 0.999 against single-stream DecodeRunner.
//
// We use the prefill+decode pattern: prefill goes through the existing
// (non-paged) multi-token path so the per-request KV state is built up
// in the contiguous context cache. Then for the FIRST decode token,
// run_decode_paged dispatches paged-flash. The K/V history seen by
// paged-flash comes from paged storage that was populated incrementally.
//
// For this validation: compare the BATCHED-PAGED first-decode-step
// logits (the very first decode token's distribution) against the
// single-stream first-decode-step logits.
//
// Caveat: the single-stream KV storage is contiguous fp32; the paged
// storage is paged fp16. Some drift is expected from fp16 quantization
// and the different attention kernel. Cosine >= 0.999 is the bar.
TEST(BatchedPagedG1, N1FirstDecodeStepCosine) {
    if (!tinyLlamaAvailable()) {
        GTEST_SKIP() << "TinyLlama model not present at " << tinyLlamaPath();
    }

    std::vector<uint64_t> tokens = {1, 1462, 1554, 304, 263};

    // Reference: single-stream DecodeRunner — process all 5 tokens, capture
    // logits at step 4 (the last input token).
    mlc::runtime::DecodeOptions opts;
    opts.tokens = tokens;
    opts.max_steps = tokens.size();
    opts.start_position = 0;
    opts.top_k = 0;
    mlc::runtime::DecodeRunner runner(tinyLlamaPath());
    auto ref = runner.run(opts);
    ASSERT_TRUE(ref.success);
    ASSERT_EQ(ref.steps.size(), tokens.size());
    const auto& ref_logits = ref.steps.back().logits;
    ASSERT_FALSE(ref_logits.empty());

    // Test: BatchedExecutor with paged storage, processing the same tokens
    // via run_decode_paged one token at a time.
    mlc::runtime::Session session(tinyLlamaPath());
    mlc::runtime::BatchedExecutor exec(session);

    constexpr uint32_t REQ = 100;
    constexpr uint32_t PAGE_SIZE_TOKENS = 64;
    constexpr uint32_t CAPACITY_PAGES = 32;

    mlc::runtime::PagePool pool(CAPACITY_PAGES);
    exec.attach_page_pool(&pool, PAGE_SIZE_TOKENS);

    // Determine model attention shape from the graph.
    const auto& graph = exec.graph();
    size_t n_kv_heads = 0, head_dim = 0;
    for (const auto& n : graph.nodes()) {
        if (n.op == mlc::runtime::ExecOpType::Attention) {
            n_kv_heads = static_cast<size_t>(n.attributes.at("kv_heads"));
            head_dim   = static_cast<size_t>(n.attributes.at("head_dim"));
            break;
        }
    }
    ASSERT_GT(n_kv_heads, 0u);
    ASSERT_GT(head_dim, 0u);
    size_t n_layers = 0;
    for (const auto& n : graph.nodes()) {
        if (n.op == mlc::runtime::ExecOpType::Attention) ++n_layers;
    }
    ASSERT_GT(n_layers, 0u);

    mlc::runtime::PagedKVStorage paged(CAPACITY_PAGES, n_layers,
                                       PAGE_SIZE_TOKENS, n_kv_heads, head_dim,
                                       /*dtype_bytes=*/2);  // fp16
    ASSERT_TRUE(paged.initialize());
    exec.attach_paged_storage(&paged);

    // Run decode token-by-token via the paged path. last_logits = logits at
    // the final position.
    std::vector<float> last_logits;
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::vector<mlc::runtime::RequestSlot> slots = {{REQ, tokens[i], i}};
        auto out = exec.run_decode_paged(slots);
        ASSERT_EQ(out.per_request.size(), 1u);
        ASSERT_TRUE(out.per_request[0].success) << "step=" << i;
        last_logits = out.per_request[0].logits;
    }

    ASSERT_FALSE(last_logits.empty());
    ASSERT_EQ(last_logits.size(), ref_logits.size());

    double cos = cosineSimilarity(last_logits, ref_logits);
    EXPECT_GE(cos, 0.999) << "cosine=" << cos;

    // Greedy match (for diagnostic — different attention impl may pick a
    // different top-1 even at cos=0.9999, that's acceptable per the brief).
    uint64_t paged_top = argmax(last_logits);
    uint64_t ref_top = argmax(ref_logits);
    if (paged_top != ref_top) {
        std::cerr << "[diag] paged top-1=" << paged_top << " ref top-1=" << ref_top
                  << " (acceptable per design — different reduction order)\n";
    }
}

// G1: at N=2, each request's logits should still match its single-stream
// baseline within cosine >= 0.999. The batched paged-flash dispatches all
// N requests' attention in one call.
TEST(BatchedPagedG1, N2EachRequestCosineMatch) {
    if (!tinyLlamaAvailable()) {
        GTEST_SKIP() << "TinyLlama model not present at " << tinyLlamaPath();
    }

    std::vector<uint64_t> prompt_a = {1, 1462, 1554, 304, 263};
    std::vector<uint64_t> prompt_b = {1, 17320, 366, 460, 263};
    constexpr size_t kSteps = 5;

    auto runRef = [&](const std::vector<uint64_t>& tokens) {
        mlc::runtime::DecodeOptions opts;
        opts.tokens = tokens;
        opts.max_steps = kSteps;
        opts.start_position = 0;
        opts.top_k = 0;
        mlc::runtime::DecodeRunner r(tinyLlamaPath());
        return r.run(opts);
    };
    auto ref_a = runRef(prompt_a);
    auto ref_b = runRef(prompt_b);
    ASSERT_TRUE(ref_a.success && ref_b.success);

    mlc::runtime::Session session(tinyLlamaPath());
    mlc::runtime::BatchedExecutor exec(session);

    constexpr uint32_t REQ_A = 11;
    constexpr uint32_t REQ_B = 22;
    constexpr uint32_t PAGE_SIZE_TOKENS = 64;
    constexpr uint32_t CAPACITY_PAGES = 32;

    mlc::runtime::PagePool pool(CAPACITY_PAGES);
    exec.attach_page_pool(&pool, PAGE_SIZE_TOKENS);

    const auto& graph = exec.graph();
    size_t n_kv_heads = 0, head_dim = 0, n_layers = 0;
    for (const auto& n : graph.nodes()) {
        if (n.op == mlc::runtime::ExecOpType::Attention) {
            if (n_kv_heads == 0) {
                n_kv_heads = static_cast<size_t>(n.attributes.at("kv_heads"));
                head_dim   = static_cast<size_t>(n.attributes.at("head_dim"));
            }
            ++n_layers;
        }
    }
    mlc::runtime::PagedKVStorage paged(CAPACITY_PAGES, n_layers, PAGE_SIZE_TOKENS,
                                       n_kv_heads, head_dim, 2);
    ASSERT_TRUE(paged.initialize());
    exec.attach_paged_storage(&paged);

    // Lock-step decode: at each step, both requests get their next token
    // batched into a single run_decode_paged call.
    for (size_t i = 0; i < kSteps; ++i) {
        std::vector<mlc::runtime::RequestSlot> slots = {
            {REQ_A, prompt_a[i], i},
            {REQ_B, prompt_b[i], i},
        };
        auto out = exec.run_decode_paged(slots);
        ASSERT_EQ(out.per_request.size(), 2u);

        const std::vector<float>* got_a = nullptr;
        const std::vector<float>* got_b = nullptr;
        for (const auto& per : out.per_request) {
            if (per.request_id == REQ_A) got_a = &per.logits;
            else if (per.request_id == REQ_B) got_b = &per.logits;
        }
        ASSERT_NE(got_a, nullptr);
        ASSERT_NE(got_b, nullptr);
        ASSERT_FALSE(got_a->empty()) << "step=" << i;
        ASSERT_FALSE(got_b->empty()) << "step=" << i;

        double cos_a = cosineSimilarity(*got_a, ref_a.steps[i].logits);
        double cos_b = cosineSimilarity(*got_b, ref_b.steps[i].logits);
        EXPECT_GE(cos_a, 0.999) << "request A step=" << i << " cos=" << cos_a;
        EXPECT_GE(cos_b, 0.999) << "request B step=" << i << " cos=" << cos_b;
    }
}

// G1: at N=4, each request's logits should still match its single-stream
// baseline within cosine >= 0.999.
TEST(BatchedPagedG1, N4EachRequestCosineMatch) {
    if (!tinyLlamaAvailable()) {
        GTEST_SKIP() << "TinyLlama model not present at " << tinyLlamaPath();
    }

    std::vector<std::vector<uint64_t>> prompts = {
        {1, 1462, 1554, 304, 263},
        {1, 17320, 366, 460, 263},
        {1, 415, 7483, 310, 3444},
        {1, 17320, 590, 1024, 5181},
    };
    constexpr size_t kSteps = 5;

    // References per prompt.
    std::vector<mlc::runtime::DecodeResult> refs;
    refs.reserve(prompts.size());
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

    mlc::runtime::Session session(tinyLlamaPath());
    mlc::runtime::BatchedExecutor exec(session);

    constexpr uint32_t PAGE_SIZE_TOKENS = 64;
    constexpr uint32_t CAPACITY_PAGES = 64;
    mlc::runtime::PagePool pool(CAPACITY_PAGES);
    exec.attach_page_pool(&pool, PAGE_SIZE_TOKENS);

    const auto& graph = exec.graph();
    size_t n_kv_heads = 0, head_dim = 0, n_layers = 0;
    for (const auto& n : graph.nodes()) {
        if (n.op == mlc::runtime::ExecOpType::Attention) {
            if (n_kv_heads == 0) {
                n_kv_heads = static_cast<size_t>(n.attributes.at("kv_heads"));
                head_dim   = static_cast<size_t>(n.attributes.at("head_dim"));
            }
            ++n_layers;
        }
    }
    mlc::runtime::PagedKVStorage paged(CAPACITY_PAGES, n_layers, PAGE_SIZE_TOKENS,
                                       n_kv_heads, head_dim, 2);
    ASSERT_TRUE(paged.initialize());
    exec.attach_paged_storage(&paged);

    std::vector<uint32_t> ids = {31, 32, 33, 34};
    for (size_t i = 0; i < kSteps; ++i) {
        std::vector<mlc::runtime::RequestSlot> slots;
        for (size_t r = 0; r < prompts.size(); ++r) {
            slots.push_back({ids[r], prompts[r][i], i});
        }
        auto out = exec.run_decode_paged(slots);
        ASSERT_EQ(out.per_request.size(), prompts.size());

        for (size_t r = 0; r < prompts.size(); ++r) {
            // Find this request's per output (order in `out` may match slots).
            const std::vector<float>* got = nullptr;
            for (const auto& per : out.per_request) {
                if (per.request_id == ids[r]) { got = &per.logits; break; }
            }
            ASSERT_NE(got, nullptr) << "req=" << r << " step=" << i;
            ASSERT_FALSE(got->empty()) << "req=" << r << " step=" << i;
            double cos = cosineSimilarity(*got, refs[r].steps[i].logits);
            EXPECT_GE(cos, 0.999) << "req=" << r << " step=" << i << " cos=" << cos;
        }
    }
}
