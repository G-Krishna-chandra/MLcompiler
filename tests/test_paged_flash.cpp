#include <gtest/gtest.h>

#include "runtime/metal_runtime.hpp"
#include "runtime/float_convert.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace {

// CPU reference: paged flash attention (online softmax, fp32 throughout).
// Inputs match the kernel's shapes. K/V pages are fp16; we cast to fp32
// inside the reference for fidelity.
//
// Returns O[batch * num_heads * head_dim] (fp32).
std::vector<float> cpuPagedFlashAttention(
    const std::vector<float>&    q_flat,           // [batch * num_heads * head_dim]
    const std::vector<uint16_t>& k_pages_f16,
    const std::vector<uint16_t>& v_pages_f16,
    const std::vector<uint32_t>& page_tables_flat,
    const std::vector<uint32_t>& page_table_offsets, // [batch + 1]
    const std::vector<uint32_t>& seq_lens,           // [batch]
    const std::vector<uint32_t>& q_positions,        // [batch]
    uint32_t batch,
    uint32_t num_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t page_size_tokens,
    bool     apply_causal)
{
    std::vector<float> out(batch * num_heads * head_dim, 0.0f);
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const uint32_t elements_per_page = page_size_tokens * kv_heads * head_dim;

    auto load_kv = [&](const std::vector<uint16_t>& pages, uint32_t req,
                       uint32_t kv_h, uint32_t pos, uint32_t d) {
        uint32_t page_idx = pos / page_size_tokens;
        uint32_t slot     = pos % page_size_tokens;
        uint32_t page_id  = page_tables_flat[page_table_offsets[req] + page_idx];
        size_t base = static_cast<size_t>(page_id) * elements_per_page +
                      static_cast<size_t>(slot) * (kv_heads * head_dim) +
                      static_cast<size_t>(kv_h) * head_dim + d;
        return mlc::runtime::fp16ToFloat(pages[base]);
    };

    for (uint32_t r = 0; r < batch; ++r) {
        const uint32_t S = seq_lens[r];
        const uint32_t qpos = q_positions[r];
        for (uint32_t h = 0; h < num_heads; ++h) {
            uint32_t kv_h = (h * kv_heads) / num_heads;

            // Compute scores for every kv_pos in [0, S), apply causal mask,
            // softmax, then accumulate weighted V.
            std::vector<float> scores(S, 0.0f);
            float max_score = -INFINITY;
            for (uint32_t pos = 0; pos < S; ++pos) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    float qval = q_flat[(r * num_heads + h) * head_dim + d];
                    float kval = load_kv(k_pages_f16, r, kv_h, pos, d);
                    dot += qval * kval;
                }
                float scaled = dot * inv_sqrt_d;
                if (apply_causal && pos > qpos) scaled = -INFINITY;
                scores[pos] = scaled;
                if (scaled > max_score) max_score = scaled;
            }
            // Stable softmax.
            std::vector<float> ws(S, 0.0f);
            float sum = 0.0f;
            for (uint32_t pos = 0; pos < S; ++pos) {
                float e = (scores[pos] == -INFINITY) ? 0.0f
                                                     : std::exp(scores[pos] - max_score);
                ws[pos] = e;
                sum += e;
            }
            float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
            // Accumulate.
            for (uint32_t d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (uint32_t pos = 0; pos < S; ++pos) {
                    float vval = load_kv(v_pages_f16, r, kv_h, pos, d);
                    acc += ws[pos] * vval;
                }
                out[(r * num_heads + h) * head_dim + d] = acc * inv_sum;
            }
        }
    }
    return out;
}

double cosine(const float* a, const float* b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

struct ScenarioParams {
    uint32_t batch;
    uint32_t num_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t page_size_tokens;
    uint32_t capacity_pages;
    std::vector<uint32_t> seq_lens;     // size = batch
    bool apply_causal;
};

// Run the scenario through the Metal kernel and compare against the CPU
// reference. Returns true if every (request, head) pair has cosine >= 0.999.
void RunScenario(const ScenarioParams& sp) {
    auto& exec = mlc::runtime::MetalExecutor::Instance();
    if (!exec.isAvailable()) {
        GTEST_SKIP() << "Metal unavailable";
    }

    ASSERT_EQ(sp.seq_lens.size(), sp.batch);
    const uint32_t D = sp.head_dim;
    const uint32_t H = sp.num_heads;

    std::mt19937 rng(0xBEEF);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // Random Q.
    std::vector<float> q_flat(sp.batch * H * D);
    for (auto& v : q_flat) v = dist(rng);

    // Build per-page random K/V (fp32 → fp16).
    size_t page_floats = sp.capacity_pages * sp.page_size_tokens * sp.kv_heads * D;
    std::vector<float> k_f32(page_floats), v_f32(page_floats);
    for (auto& v : k_f32) v = dist(rng);
    for (auto& v : v_f32) v = dist(rng);
    std::vector<uint16_t> k_f16(page_floats), v_f16(page_floats);
    mlc::runtime::castF32toF16(k_f32.data(), k_f16.data(), page_floats);
    mlc::runtime::castF32toF16(v_f32.data(), v_f16.data(), page_floats);

    // Page tables: assign pages round-robin so requests get disjoint pages.
    std::vector<uint32_t> page_tables_flat;
    std::vector<uint32_t> page_table_offsets(sp.batch + 1, 0);
    uint32_t next_page_id = 0;
    for (uint32_t r = 0; r < sp.batch; ++r) {
        page_table_offsets[r] = page_tables_flat.size();
        uint32_t pages_needed = (sp.seq_lens[r] + sp.page_size_tokens - 1) / sp.page_size_tokens;
        for (uint32_t p = 0; p < pages_needed; ++p) {
            ASSERT_LT(next_page_id, sp.capacity_pages) << "test setup: pool too small";
            page_tables_flat.push_back(next_page_id++);
        }
    }
    page_table_offsets[sp.batch] = page_tables_flat.size();

    // q_positions: place each request's Q at its last token (for causal).
    std::vector<uint32_t> q_positions(sp.batch);
    for (uint32_t r = 0; r < sp.batch; ++r) {
        q_positions[r] = sp.seq_lens[r] > 0 ? sp.seq_lens[r] - 1 : 0;
    }

    // CPU reference.
    auto ref = cpuPagedFlashAttention(q_flat, k_f16, v_f16,
                                      page_tables_flat, page_table_offsets,
                                      sp.seq_lens, q_positions,
                                      sp.batch, H, sp.kv_heads, D,
                                      sp.page_size_tokens, sp.apply_causal);

    // Allocate Metal buffers.
    void* q_buf = exec.allocateScratchBuffer(q_flat.size() * sizeof(float));
    void* k_buf = exec.allocateScratchBuffer(k_f16.size() * sizeof(uint16_t));
    void* v_buf = exec.allocateScratchBuffer(v_f16.size() * sizeof(uint16_t));
    void* o_buf = exec.allocateScratchBuffer(sp.batch * H * D * sizeof(float));
    ASSERT_NE(q_buf, nullptr);
    ASSERT_NE(k_buf, nullptr);
    ASSERT_NE(v_buf, nullptr);
    ASSERT_NE(o_buf, nullptr);

    exec.uploadToBuffer(q_buf, q_flat.data(), q_flat.size() * sizeof(float));
    exec.uploadToBuffer(k_buf, k_f16.data(), k_f16.size() * sizeof(uint16_t));
    exec.uploadToBuffer(v_buf, v_f16.data(), v_f16.size() * sizeof(uint16_t));
    std::vector<float> zeros(sp.batch * H * D, 0.0f);
    exec.uploadToBuffer(o_buf, zeros.data(), zeros.size() * sizeof(float));

    bool ok = exec.runPagedFlashAttention(q_buf, k_buf, v_buf, o_buf,
                                          page_tables_flat, page_table_offsets,
                                          sp.seq_lens, q_positions,
                                          sp.batch, H, sp.kv_heads, D,
                                          sp.page_size_tokens, sp.apply_causal);
    ASSERT_TRUE(ok);

    std::vector<float> got(sp.batch * H * D, 0.0f);
    exec.downloadFromBuffer(o_buf, got.data(), got.size() * sizeof(float));

    // Per (request, head) cosine.
    for (uint32_t r = 0; r < sp.batch; ++r) {
        for (uint32_t h = 0; h < H; ++h) {
            const float* a = ref.data() + (r * H + h) * D;
            const float* b = got.data() + (r * H + h) * D;
            double c = cosine(a, b, D);
            EXPECT_GE(c, 0.999) << "req=" << r << " head=" << h << " cos=" << c;
        }
    }

    exec.releaseScratchBuffer(q_buf);
    exec.releaseScratchBuffer(k_buf);
    exec.releaseScratchBuffer(v_buf);
    exec.releaseScratchBuffer(o_buf);
}

} // namespace

TEST(PagedFlashF1, SingleRequestSmall) {
    ScenarioParams sp;
    sp.batch = 1; sp.num_heads = 2; sp.kv_heads = 2; sp.head_dim = 64;
    sp.page_size_tokens = 4; sp.capacity_pages = 8;
    sp.seq_lens = {8};
    sp.apply_causal = true;
    RunScenario(sp);
}

TEST(PagedFlashF1, TwoRequestsDisjointPages) {
    ScenarioParams sp;
    sp.batch = 2; sp.num_heads = 2; sp.kv_heads = 2; sp.head_dim = 64;
    sp.page_size_tokens = 4; sp.capacity_pages = 16;
    sp.seq_lens = {10, 12};
    sp.apply_causal = true;
    RunScenario(sp);
}

TEST(PagedFlashF1, FourRequestsGQA) {
    ScenarioParams sp;
    sp.batch = 4; sp.num_heads = 8; sp.kv_heads = 2; sp.head_dim = 64;
    sp.page_size_tokens = 4; sp.capacity_pages = 32;
    sp.seq_lens = {20, 14, 8, 18};
    sp.apply_causal = true;
    RunScenario(sp);
}

TEST(PagedFlashF1, EightRequestsVariableLength) {
    ScenarioParams sp;
    sp.batch = 8; sp.num_heads = 8; sp.kv_heads = 2; sp.head_dim = 64;
    sp.page_size_tokens = 8; sp.capacity_pages = 64;
    sp.seq_lens = {4, 12, 20, 32, 7, 16, 28, 11};
    sp.apply_causal = true;
    RunScenario(sp);
}

TEST(PagedFlashF1, MultiplePagesPerRequest) {
    ScenarioParams sp;
    sp.batch = 4; sp.num_heads = 4; sp.kv_heads = 2; sp.head_dim = 64;
    sp.page_size_tokens = 16; sp.capacity_pages = 32;
    sp.seq_lens = {64, 50, 40, 32};  // all use multiple pages
    sp.apply_causal = true;
    RunScenario(sp);
}

TEST(PagedFlashF1, TinyLlamaLikeShape) {
    ScenarioParams sp;
    sp.batch = 4; sp.num_heads = 32; sp.kv_heads = 4; sp.head_dim = 64;
    sp.page_size_tokens = 64; sp.capacity_pages = 8;
    sp.seq_lens = {32, 48, 64, 96};
    sp.apply_causal = true;
    RunScenario(sp);
}

TEST(PagedFlashF1, NoCausal) {
    ScenarioParams sp;
    sp.batch = 2; sp.num_heads = 4; sp.kv_heads = 2; sp.head_dim = 64;
    sp.page_size_tokens = 8; sp.capacity_pages = 16;
    sp.seq_lens = {16, 24};
    sp.apply_causal = false;
    RunScenario(sp);
}
