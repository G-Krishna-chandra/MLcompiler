#include <gtest/gtest.h>

#include "runtime/paged_kv.hpp"

#include <set>
#include <unordered_set>

using mlc::runtime::PagePool;
using mlc::runtime::RequestKVState;

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
