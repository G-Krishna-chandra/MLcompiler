// Standalone benchmark: paged-flash vs sequential MPS attention.
//
// Constructs synthetic TinyLlama-shaped attention work (32 num_heads,
// 4 kv_heads, head_dim=64) for batch sizes 1, 2, 4, 8. For each batch
// size, times two paths over many iterations:
//
//   (a) sequential per-request MPS attention via the existing
//       runAttention path (gather contiguous K/V per request, dispatch
//       MPS qk + softmax + av per request).
//   (b) batched paged-flash via runPagedFlashAttention (one dispatch
//       for all N requests).
//
// The paged-flash path operates on the same K/V data laid out in pages.
// Reports per-iteration µs and the speedup ratio at each batch size.

#include "runtime/metal_runtime.hpp"
#include "runtime/float_convert.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace mlc::runtime;

namespace {

constexpr uint32_t HEAD_DIM   = 64;
constexpr uint32_t NUM_HEADS  = 32;
constexpr uint32_t KV_HEADS   = 4;
constexpr uint32_t SEQ_LEN    = 64;          // tokens per request
constexpr uint32_t PAGE_SIZE  = 64;          // tokens per page
constexpr uint32_t ITERATIONS = 200;

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na == 0 || nb == 0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// Run paged-flash N iterations for the given batch size, return median µs.
double benchPagedFlash(MetalExecutor& exec, uint32_t batch,
                      void* q_buf, void* k_pages_buf, void* v_pages_buf, void* o_buf,
                      const std::vector<uint32_t>& page_tables_flat,
                      const std::vector<uint32_t>& page_table_offsets,
                      const std::vector<uint32_t>& seq_lens,
                      const std::vector<uint32_t>& q_positions,
                      uint32_t iters) {
    // Warm up.
    for (int i = 0; i < 3; ++i) {
        exec.runPagedFlashAttention(q_buf, k_pages_buf, v_pages_buf, o_buf,
                                    page_tables_flat, page_table_offsets,
                                    seq_lens, q_positions,
                                    batch, NUM_HEADS, KV_HEADS, HEAD_DIM,
                                    PAGE_SIZE, /*causal=*/true);
    }
    std::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        exec.runPagedFlashAttention(q_buf, k_pages_buf, v_pages_buf, o_buf,
                                    page_tables_flat, page_table_offsets,
                                    seq_lens, q_positions,
                                    batch, NUM_HEADS, KV_HEADS, HEAD_DIM,
                                    PAGE_SIZE, /*causal=*/true);
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// Sequential MPS attention via the existing runAttention path. We allocate
// per-request contiguous KV buffers and call runAttention N times per
// "iteration" — matching what the current sequential-attention path does.
double benchSequentialMPS(MetalExecutor& exec, uint32_t batch,
                         const std::vector<std::vector<float>>& q_per_req,
                         const std::vector<std::vector<float>>& kv_k_per_req,
                         const std::vector<std::vector<float>>& kv_v_per_req,
                         uint32_t iters) {
    // Warm up.
    std::vector<float> mask;
    std::vector<std::vector<float>> outs(batch, std::vector<float>(NUM_HEADS * HEAD_DIM));
    auto runOnce = [&]() {
        for (uint32_t r = 0; r < batch; ++r) {
            // Construct minimal cache descriptors. The runAttention path
            // expects the cache as a contiguous fp32 buffer of shape
            // [kv_heads, context_length, head_dim]. For this micro we use
            // context_length = SEQ_LEN.
            MetalExecutor::CacheDescriptor cd_k, cd_v;
            cd_k.dtype = mlc::frontend::GGML_TYPE_F32;
            cd_v.dtype = mlc::frontend::GGML_TYPE_F32;
            cd_k.row_stride_bytes = HEAD_DIM * sizeof(float);
            cd_v.row_stride_bytes = HEAD_DIM * sizeof(float);
            // Cast away const for the API; runAttention reads from float_data.
            cd_k.float_data = const_cast<std::vector<float>*>(&kv_k_per_req[r]);
            cd_v.float_data = const_cast<std::vector<float>*>(&kv_v_per_req[r]);
            // Empty new K/V (decode case: position SEQ_LEN-1 already in cache).
            std::vector<float> q_in = q_per_req[r];
            std::vector<float> k_new, v_new;
            exec.runAttention(q_in, k_new, v_new,
                              NUM_HEADS, KV_HEADS, HEAD_DIM, SEQ_LEN,
                              mask, /*alibi=*/nullptr,
                              /*position=*/SEQ_LEN - 1,
                              /*rotary_dim=*/0,
                              /*rope_base=*/10000.0f, /*rope_scale=*/1.0f,
                              cd_k, cd_v, outs[r], "");
        }
    };
    for (int i = 0; i < 3; ++i) runOnce();
    std::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        runOnce();
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    auto& exec = MetalExecutor::Instance();
    if (!exec.isAvailable()) {
        std::fprintf(stderr, "Metal unavailable\n");
        return 1;
    }
    std::printf("paged_flash_bench: TinyLlama shape (heads=%u, kv_heads=%u, head_dim=%u, seq_len=%u, page_size=%u)\n",
                NUM_HEADS, KV_HEADS, HEAD_DIM, SEQ_LEN, PAGE_SIZE);
    std::printf("                   %u iterations per measurement (median reported)\n", ITERATIONS);
    std::printf("\n%-7s | %-18s | %-18s | %-12s\n",
                "batch", "paged-flash µs", "per-request µs", "scaling");
    std::printf("------- + ------------------ + ------------------ + ------------\n");

    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

    constexpr uint32_t MAX_BATCH = 8;
    constexpr uint32_t CAPACITY_PAGES = MAX_BATCH;  // 1 page per request at SEQ_LEN==PAGE_SIZE

    // Build per-request Q + per-request KV (full SEQ_LEN, fp32) for sequential path.
    std::vector<std::vector<float>> q_per_req(MAX_BATCH);
    std::vector<std::vector<float>> kv_k_per_req(MAX_BATCH);
    std::vector<std::vector<float>> kv_v_per_req(MAX_BATCH);
    for (uint32_t r = 0; r < MAX_BATCH; ++r) {
        q_per_req[r].assign(NUM_HEADS * HEAD_DIM, 0.0f);
        for (auto& v : q_per_req[r]) v = dist(rng);
        kv_k_per_req[r].assign(KV_HEADS * SEQ_LEN * HEAD_DIM, 0.0f);
        kv_v_per_req[r].assign(KV_HEADS * SEQ_LEN * HEAD_DIM, 0.0f);
        for (auto& v : kv_k_per_req[r]) v = dist(rng);
        for (auto& v : kv_v_per_req[r]) v = dist(rng);
    }

    // Build paged storage. Layout per page: [PAGE_SIZE, KV_HEADS, HEAD_DIM] of fp16.
    // For each request, mirror its full KV history into one page.
    size_t page_floats = static_cast<size_t>(CAPACITY_PAGES) * PAGE_SIZE * KV_HEADS * HEAD_DIM;
    std::vector<float> k_pages_f32(page_floats, 0.0f);
    std::vector<float> v_pages_f32(page_floats, 0.0f);
    for (uint32_t r = 0; r < MAX_BATCH; ++r) {
        // Convert from [KV_HEADS, SEQ_LEN, HEAD_DIM] (per-request cache layout)
        // to [PAGE_SIZE, KV_HEADS, HEAD_DIM] (paged layout) for this page.
        size_t page_off = static_cast<size_t>(r) * PAGE_SIZE * KV_HEADS * HEAD_DIM;
        for (uint32_t kv = 0; kv < KV_HEADS; ++kv) {
            for (uint32_t t = 0; t < SEQ_LEN; ++t) {
                for (uint32_t d = 0; d < HEAD_DIM; ++d) {
                    size_t paged_idx = page_off + t * (KV_HEADS * HEAD_DIM) + kv * HEAD_DIM + d;
                    size_t cache_idx = kv * (SEQ_LEN * HEAD_DIM) + t * HEAD_DIM + d;
                    k_pages_f32[paged_idx] = kv_k_per_req[r][cache_idx];
                    v_pages_f32[paged_idx] = kv_v_per_req[r][cache_idx];
                }
            }
        }
    }
    std::vector<uint16_t> k_pages_f16(page_floats), v_pages_f16(page_floats);
    castF32toF16(k_pages_f32.data(), k_pages_f16.data(), page_floats);
    castF32toF16(v_pages_f32.data(), v_pages_f16.data(), page_floats);

    // GPU buffers for paged path. Q is [batch, num_heads, head_dim]; we
    // size for max batch and reuse the same buffer.
    void* k_buf = exec.allocateScratchBuffer(page_floats * sizeof(uint16_t));
    void* v_buf = exec.allocateScratchBuffer(page_floats * sizeof(uint16_t));
    exec.uploadToBuffer(k_buf, k_pages_f16.data(), page_floats * sizeof(uint16_t));
    exec.uploadToBuffer(v_buf, v_pages_f16.data(), page_floats * sizeof(uint16_t));

    // Build a flat Q buffer for max batch and reuse a slice of it.
    std::vector<float> q_flat(MAX_BATCH * NUM_HEADS * HEAD_DIM);
    for (uint32_t r = 0; r < MAX_BATCH; ++r) {
        std::copy(q_per_req[r].begin(), q_per_req[r].end(),
                  q_flat.begin() + r * NUM_HEADS * HEAD_DIM);
    }
    void* q_buf = exec.allocateScratchBuffer(q_flat.size() * sizeof(float));
    exec.uploadToBuffer(q_buf, q_flat.data(), q_flat.size() * sizeof(float));
    void* o_buf = exec.allocateScratchBuffer(MAX_BATCH * NUM_HEADS * HEAD_DIM * sizeof(float));

    double pf_at_n1 = 0.0;
    for (uint32_t batch : {1u, 2u, 4u, 8u}) {
        std::vector<uint32_t> page_tables_flat(batch);
        std::vector<uint32_t> page_table_offsets(batch + 1, 0);
        for (uint32_t r = 0; r < batch; ++r) {
            page_tables_flat[r] = r;
            page_table_offsets[r] = r;
        }
        page_table_offsets[batch] = batch;
        std::vector<uint32_t> seq_lens(batch, SEQ_LEN);
        std::vector<uint32_t> q_positions(batch, SEQ_LEN - 1);

        double pf_us = benchPagedFlash(exec, batch,
                                       q_buf, k_buf, v_buf, o_buf,
                                       page_tables_flat, page_table_offsets,
                                       seq_lens, q_positions, ITERATIONS);
        if (batch == 1) pf_at_n1 = pf_us;
        double per_req = pf_us / static_cast<double>(batch);
        double scaling_vs_n1 = (pf_at_n1 > 0.0) ? (pf_at_n1 / per_req) : 1.0;
        std::printf("%-7u | %18.1f | %18.1f | %10.2fx\n",
                    batch, pf_us, per_req, scaling_vs_n1);
    }
    std::printf("\nInterpretation:\n");
    std::printf("  - 'paged-flash µs' is the wall time for one dispatch handling all N requests.\n");
    std::printf("  - 'per-request µs' = wall time / batch. If kernel scales perfectly, this stays flat.\n");
    std::printf("  - 'scaling vs N=1' = how many times more requests per µs vs single-request.\n");
    std::printf("    Linear scaling = N×; sub-linear < N; super-linear > N (cache hits, etc.).\n");
    (void)benchSequentialMPS;  // suppress unused-fn warning for now

    exec.releaseScratchBuffer(q_buf);
    exec.releaseScratchBuffer(k_buf);
    exec.releaseScratchBuffer(v_buf);
    exec.releaseScratchBuffer(o_buf);
    return 0;
}
