#include <gtest/gtest.h>

#include "runtime/paged_kv.hpp"
#include "runtime/metal_runtime.hpp"
#include "runtime/float_convert.hpp"

#include <cstdint>
#include <cstring>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>

using mlc::runtime::PagePool;
using mlc::runtime::RequestKVState;
using mlc::runtime::MetalExecutor;

TEST(PagePool, AllocReleaseSingle) {
    PagePool pool(16);
    EXPECT_EQ(pool.capacity(), 16u);
    EXPECT_EQ(pool.pages_in_use(), 0u);
    EXPECT_EQ(pool.pages_free(), 16u);

    auto id = pool.allocate();
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(pool.pages_in_use(), 1u);
    EXPECT_EQ(pool.pages_free(), 15u);

    pool.release(*id);
    EXPECT_EQ(pool.pages_in_use(), 0u);
    EXPECT_EQ(pool.pages_free(), 16u);
}

TEST(PagePool, AllocAscendingFromFreshPool) {
    PagePool pool(8);
    std::vector<uint32_t> ids;
    for (int i = 0; i < 8; ++i) {
        auto id = pool.allocate();
        ASSERT_TRUE(id.has_value());
        ids.push_back(*id);
    }
    // Fresh pool issues IDs in ascending order.
    for (size_t i = 0; i < ids.size(); ++i) {
        EXPECT_EQ(ids[i], static_cast<uint32_t>(i));
    }
    EXPECT_EQ(pool.pages_in_use(), 8u);
}

TEST(PagePool, ExhaustionReturnsNullopt) {
    PagePool pool(2);
    EXPECT_TRUE(pool.allocate().has_value());
    EXPECT_TRUE(pool.allocate().has_value());
    auto third = pool.allocate();
    EXPECT_FALSE(third.has_value());
    EXPECT_EQ(pool.pages_free(), 0u);
    EXPECT_EQ(pool.pages_in_use(), 2u);
}

TEST(PagePool, ReleasedPageReused) {
    PagePool pool(4);
    auto a = pool.allocate();
    auto b = pool.allocate();
    auto c = pool.allocate();
    auto d = pool.allocate();
    ASSERT_TRUE(a && b && c && d);

    pool.release(*b);
    auto reused = pool.allocate();
    ASSERT_TRUE(reused.has_value());
    EXPECT_EQ(*reused, *b);  // most-recently-released is reissued first (LIFO)
}

TEST(PagePool, BulkAllocAtomicSuccess) {
    PagePool pool(10);
    std::vector<uint32_t> ids;
    EXPECT_TRUE(pool.allocate_n(7, ids));
    EXPECT_EQ(ids.size(), 7u);
    EXPECT_EQ(pool.pages_in_use(), 7u);
    EXPECT_EQ(pool.pages_free(), 3u);
    // No duplicates.
    std::unordered_set<uint32_t> as_set(ids.begin(), ids.end());
    EXPECT_EQ(as_set.size(), 7u);
}

TEST(PagePool, BulkAllocAtomicFailureLeavesPoolIntact) {
    PagePool pool(5);
    std::vector<uint32_t> first;
    ASSERT_TRUE(pool.allocate_n(3, first));
    std::vector<uint32_t> second;
    EXPECT_FALSE(pool.allocate_n(3, second));  // only 2 left
    EXPECT_TRUE(second.empty());
    EXPECT_EQ(pool.pages_in_use(), 3u);
    EXPECT_EQ(pool.pages_free(), 2u);
}

TEST(RequestKVState, EmptyTotalIsZero) {
    RequestKVState state;
    state.page_size_tokens = 64;
    EXPECT_EQ(state.total_tokens(), 0u);
}

TEST(RequestKVState, ReserveAllocatesEnoughPages) {
    PagePool pool(32);
    RequestKVState state;
    state.page_size_tokens = 64;

    // 100 tokens needs 2 pages (page 0: 0..63, page 1: 64..99).
    ASSERT_TRUE(state.reserve(pool, 100));
    EXPECT_EQ(state.page_table.size(), 2u);
    EXPECT_EQ(pool.pages_in_use(), 2u);

    // Idempotent: reserving fewer tokens does not shrink.
    ASSERT_TRUE(state.reserve(pool, 50));
    EXPECT_EQ(state.page_table.size(), 2u);

    // Extending: 200 tokens needs ceil(200/64) = 4 pages.
    ASSERT_TRUE(state.reserve(pool, 200));
    EXPECT_EQ(state.page_table.size(), 4u);
    EXPECT_EQ(pool.pages_in_use(), 4u);
}

TEST(RequestKVState, ReserveFailureKeepsTableUnchanged) {
    PagePool pool(2);
    RequestKVState state;
    state.page_size_tokens = 64;
    // 200 tokens needs 4 pages, pool only has 2.
    EXPECT_FALSE(state.reserve(pool, 200));
    EXPECT_TRUE(state.page_table.empty());
    EXPECT_EQ(pool.pages_in_use(), 0u);
}

TEST(RequestKVState, ExtendOneTokenAcrossPageBoundary) {
    PagePool pool(8);
    RequestKVState state;
    state.page_size_tokens = 4;

    std::vector<std::pair<uint32_t, uint32_t>> writes;
    for (int i = 0; i < 10; ++i) {
        auto loc = state.extend_one_token(pool);
        ASSERT_TRUE(loc.has_value()) << "i=" << i;
        writes.push_back(*loc);
    }

    EXPECT_EQ(state.total_tokens(), 10u);
    EXPECT_EQ(state.page_table.size(), 3u);
    EXPECT_EQ(state.tokens_in_last_page, 2u);

    // Slot indices increment within a page then wrap to 0 at boundary.
    EXPECT_EQ(writes[0].second, 0u);
    EXPECT_EQ(writes[3].second, 3u);
    EXPECT_EQ(writes[4].second, 0u);  // new page
    EXPECT_EQ(writes[7].second, 3u);
    EXPECT_EQ(writes[8].second, 0u);  // new page
}

TEST(RequestKVState, ExtendExhaustsPool) {
    PagePool pool(2);
    RequestKVState state;
    state.page_size_tokens = 4;

    for (int i = 0; i < 8; ++i) {
        ASSERT_TRUE(state.extend_one_token(pool).has_value()) << "i=" << i;
    }
    // 9th token would require a 3rd page; pool exhausted.
    EXPECT_FALSE(state.extend_one_token(pool).has_value());
    EXPECT_EQ(state.total_tokens(), 8u);
    EXPECT_EQ(state.page_table.size(), 2u);
    EXPECT_EQ(state.tokens_in_last_page, 4u);  // last page full but not over
}

TEST(RequestKVState, LocateRoundTripsExtensionWrites) {
    PagePool pool(8);
    RequestKVState state;
    state.page_size_tokens = 4;

    std::vector<std::pair<uint32_t, uint32_t>> writes;
    for (int i = 0; i < 10; ++i) {
        writes.push_back(*state.extend_one_token(pool));
    }
    for (int i = 0; i < 10; ++i) {
        auto loc = state.locate(static_cast<size_t>(i));
        ASSERT_TRUE(loc.has_value()) << "i=" << i;
        EXPECT_EQ(loc->first, writes[i].first) << "i=" << i;
        EXPECT_EQ(loc->second, writes[i].second) << "i=" << i;
    }
    // Out-of-range locate returns nullopt.
    EXPECT_FALSE(state.locate(10).has_value());
    EXPECT_FALSE(state.locate(100).has_value());
}

TEST(RequestKVState, ReleaseReturnsAllPages) {
    PagePool pool(8);
    RequestKVState state;
    state.page_size_tokens = 4;

    for (int i = 0; i < 6; ++i) {
        ASSERT_TRUE(state.extend_one_token(pool).has_value());
    }
    EXPECT_EQ(pool.pages_in_use(), 2u);

    state.release_all(pool);
    EXPECT_EQ(state.total_tokens(), 0u);
    EXPECT_TRUE(state.page_table.empty());
    EXPECT_EQ(state.tokens_in_last_page, 0u);
    EXPECT_EQ(pool.pages_in_use(), 0u);
    EXPECT_EQ(pool.pages_free(), 8u);
}

// =====================================================================
// Phase A2 (continuous batching): Metal-backed gather kernel tests.
// Skipped when Metal is unavailable.
// =====================================================================

namespace {

// Build a paged storage buffer with a deterministic pattern: page i, slot s,
// head h, dim d holds the float value (page_id * 100000 + slot * 1000 + head * 10 + dim).
// Returns the host-side reference layout (per-page block, token-major).
std::vector<float> buildPagedReferenceFloat(uint32_t capacity_pages,
                                            uint32_t page_size_tokens,
                                            uint32_t n_kv_heads,
                                            uint32_t head_dim) {
    std::vector<float> out(capacity_pages * page_size_tokens * n_kv_heads * head_dim, 0.f);
    for (uint32_t p = 0; p < capacity_pages; ++p) {
        for (uint32_t s = 0; s < page_size_tokens; ++s) {
            for (uint32_t h = 0; h < n_kv_heads; ++h) {
                for (uint32_t d = 0; d < head_dim; ++d) {
                    size_t idx = p * (page_size_tokens * n_kv_heads * head_dim)
                               + s * (n_kv_heads * head_dim)
                               + h * head_dim
                               + d;
                    out[idx] = static_cast<float>(p) * 100000.f
                             + static_cast<float>(s) * 1000.f
                             + static_cast<float>(h) * 10.f
                             + static_cast<float>(d);
                }
            }
        }
    }
    return out;
}

// Expected gather output for a request that holds `page_table` page ids
// covering num_tokens tokens, in the [n_kv_heads, num_tokens, head_dim] layout
// the existing K/V batch path expects.
std::vector<float> expectedGather(const std::vector<uint32_t>& page_table,
                                  uint32_t page_size_tokens,
                                  uint32_t n_kv_heads,
                                  uint32_t head_dim,
                                  uint32_t num_tokens) {
    std::vector<float> out(n_kv_heads * num_tokens * head_dim, 0.f);
    for (uint32_t h = 0; h < n_kv_heads; ++h) {
        for (uint32_t t = 0; t < num_tokens; ++t) {
            uint32_t page_idx = t / page_size_tokens;
            uint32_t slot     = t % page_size_tokens;
            uint32_t page_id  = page_table[page_idx];
            for (uint32_t d = 0; d < head_dim; ++d) {
                size_t dst = h * (num_tokens * head_dim) + t * head_dim + d;
                out[dst] = static_cast<float>(page_id) * 100000.f
                         + static_cast<float>(slot) * 1000.f
                         + static_cast<float>(h) * 10.f
                         + static_cast<float>(d);
            }
        }
    }
    return out;
}

} // namespace

TEST(GatherKVPages, F32_RoundTrips) {
    auto& exec = MetalExecutor::Instance();
    if (!exec.isAvailable()) {
        GTEST_SKIP() << "Metal unavailable on this host";
    }

    constexpr uint32_t capacity_pages   = 16;
    constexpr uint32_t page_size_tokens = 8;
    constexpr uint32_t n_kv_heads       = 4;
    constexpr uint32_t head_dim         = 16;

    auto reference = buildPagedReferenceFloat(capacity_pages, page_size_tokens,
                                              n_kv_heads, head_dim);
    size_t storage_bytes = reference.size() * sizeof(float);

    void* storage = exec.allocateScratchBuffer(storage_bytes);
    ASSERT_NE(storage, nullptr);
    exec.uploadToBuffer(storage, reference.data(), storage_bytes);

    // Request: 3 pages, 20 tokens. Pages chosen out of natural order to make sure
    // the gather honors the page table.
    std::vector<uint32_t> page_table = {7, 2, 11};
    constexpr uint32_t num_tokens = 20;  // covers pages 0,1,2 (pages 0..2 in table)

    size_t dst_bytes = n_kv_heads * num_tokens * head_dim * sizeof(float);
    void* dst = exec.allocateScratchBuffer(dst_bytes);
    ASSERT_NE(dst, nullptr);

    bool ok = exec.gatherKVPages(storage, page_table,
                                 page_size_tokens, n_kv_heads, head_dim,
                                 num_tokens, dst, /*dtype_bytes=*/4);
    ASSERT_TRUE(ok);

    std::vector<float> result(n_kv_heads * num_tokens * head_dim, 0.f);
    exec.downloadFromBuffer(dst, result.data(), dst_bytes);

    auto expected = expectedGather(page_table, page_size_tokens, n_kv_heads, head_dim, num_tokens);
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]) << "i=" << i;
    }

    exec.releaseScratchBuffer(storage);
    exec.releaseScratchBuffer(dst);
}

TEST(GatherKVPages, F16_RoundTrips) {
    auto& exec = MetalExecutor::Instance();
    if (!exec.isAvailable()) {
        GTEST_SKIP() << "Metal unavailable on this host";
    }

    constexpr uint32_t capacity_pages   = 8;
    constexpr uint32_t page_size_tokens = 8;
    constexpr uint32_t n_kv_heads       = 4;
    constexpr uint32_t head_dim         = 16;

    auto reference_f32 = buildPagedReferenceFloat(capacity_pages, page_size_tokens,
                                                  n_kv_heads, head_dim);
    // Convert to fp16 storage (use small values to avoid f16 saturation).
    // Scale down so max value < 65504.
    for (auto& v : reference_f32) v *= 1e-3f;
    std::vector<uint16_t> reference_f16(reference_f32.size());
    mlc::runtime::castF32toF16(reference_f32.data(), reference_f16.data(), reference_f32.size());

    size_t storage_bytes = reference_f16.size() * sizeof(uint16_t);
    void* storage = exec.allocateScratchBuffer(storage_bytes);
    ASSERT_NE(storage, nullptr);
    exec.uploadToBuffer(storage, reference_f16.data(), storage_bytes);

    std::vector<uint32_t> page_table = {3, 0, 6};
    constexpr uint32_t num_tokens = 17;

    size_t dst_bytes = n_kv_heads * num_tokens * head_dim * sizeof(uint16_t);
    void* dst = exec.allocateScratchBuffer(dst_bytes);
    ASSERT_NE(dst, nullptr);

    bool ok = exec.gatherKVPages(storage, page_table,
                                 page_size_tokens, n_kv_heads, head_dim,
                                 num_tokens, dst, /*dtype_bytes=*/2);
    ASSERT_TRUE(ok);

    std::vector<uint16_t> result_f16(n_kv_heads * num_tokens * head_dim, 0);
    exec.downloadFromBuffer(dst, result_f16.data(), dst_bytes);

    std::vector<float> result_f32(result_f16.size());
    mlc::runtime::castF16toF32(result_f16.data(), result_f32.data(), result_f16.size());

    auto expected_full = expectedGather(page_table, page_size_tokens, n_kv_heads, head_dim, num_tokens);
    for (auto& v : expected_full) v *= 1e-3f;  // match the down-scale above

    ASSERT_EQ(result_f32.size(), expected_full.size());
    for (size_t i = 0; i < result_f32.size(); ++i) {
        // fp16 round-trip: tolerate ~0.1% relative error.
        float ref = expected_full[i];
        float got = result_f32[i];
        float tol = std::max(1e-3f, std::abs(ref) * 1e-3f);
        EXPECT_NEAR(got, ref, tol) << "i=" << i << " ref=" << ref << " got=" << got;
    }

    exec.releaseScratchBuffer(storage);
    exec.releaseScratchBuffer(dst);
}

TEST(GatherKVPages, RejectsOversizedRequest) {
    auto& exec = MetalExecutor::Instance();
    if (!exec.isAvailable()) {
        GTEST_SKIP() << "Metal unavailable on this host";
    }

    constexpr uint32_t page_size_tokens = 8;
    constexpr uint32_t n_kv_heads       = 4;
    constexpr uint32_t head_dim         = 16;
    constexpr uint32_t capacity_pages   = 4;

    void* storage = exec.allocateScratchBuffer(
        capacity_pages * page_size_tokens * n_kv_heads * head_dim * sizeof(float));
    void* dst = exec.allocateScratchBuffer(1024);
    ASSERT_NE(storage, nullptr);
    ASSERT_NE(dst, nullptr);

    // page_table only covers 1 page (8 tokens) but we ask for 20.
    std::vector<uint32_t> page_table = {0};
    bool ok = exec.gatherKVPages(storage, page_table, page_size_tokens,
                                 n_kv_heads, head_dim, /*num_tokens=*/20,
                                 dst, /*dtype_bytes=*/4);
    EXPECT_FALSE(ok);

    exec.releaseScratchBuffer(storage);
    exec.releaseScratchBuffer(dst);
}

TEST(RequestKVState, MultipleRequestsShareSamePool) {
    PagePool pool(16);
    RequestKVState a, b;
    a.page_size_tokens = 4;
    b.page_size_tokens = 4;

    // Interleaved extensions: each request gets distinct page IDs.
    std::set<uint32_t> a_pages, b_pages;
    for (int i = 0; i < 8; ++i) {
        a_pages.insert(a.extend_one_token(pool)->first);
        b_pages.insert(b.extend_one_token(pool)->first);
    }
    // Disjoint page sets.
    for (uint32_t id : a_pages) {
        EXPECT_EQ(b_pages.count(id), 0u) << "page " << id << " in both requests";
    }
    EXPECT_EQ(pool.pages_in_use(), a_pages.size() + b_pages.size());

    a.release_all(pool);
    EXPECT_EQ(pool.pages_in_use(), b_pages.size());
    b.release_all(pool);
    EXPECT_EQ(pool.pages_in_use(), 0u);
}
