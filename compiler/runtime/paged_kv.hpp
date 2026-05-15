#pragma once

// Paged KV cache infrastructure (Phase A of continuous batching).
//
// This file defines the *logical* layer: page-id allocation and per-request
// page tables. The Metal-backed page storage (MTLBuffer per page) and the
// gather/scatter kernels live in subsequent commits (A2, A3).
//
// Decoupling logical page IDs from GPU storage lets us unit-test allocation
// behavior without a Metal device, and lets the Metal layer reuse the same
// allocator across both production decode and parity-harness paths.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace mlc {
namespace runtime {

// PagePool — logical free-list allocator over a fixed capacity of page IDs.
// Page ID 0 is a valid ID; there is no sentinel value. Use std::optional /
// the bool return values to detect exhaustion.
//
// Thread-safety: not thread-safe. The scheduler is single-threaded; the
// batched executor pulls slots into a single forward pass on one thread.
// If we add async admission later, we'll add a mutex.
class PagePool {
public:
    explicit PagePool(uint32_t capacity);

    // Returns nullopt if the pool is exhausted.
    std::optional<uint32_t> allocate();

    // Bulk allocate exactly `n` pages atomically. On success, appends `n` IDs
    // to `out` and returns true. On failure (insufficient pages), returns
    // false and `out` is unchanged.
    bool allocate_n(size_t n, std::vector<uint32_t>& out);

    // Release a previously-allocated page back to the free list.
    // Releasing a page not currently allocated is undefined behavior in
    // release builds; debug builds may assert.
    void release(uint32_t page_id);

    uint32_t capacity() const { return capacity_; }
    size_t pages_in_use() const { return in_use_; }
    size_t pages_free() const { return free_list_.size(); }

private:
    uint32_t capacity_;
    std::vector<uint32_t> free_list_;  // LIFO
    size_t in_use_ = 0;
};

// RequestKVState — per-request page table and token count.
//
// Each request owns an ordered list of page IDs (its KV page chain) and a
// counter of how many tokens are filled in the *last* page. Total tokens
// stored = (page_table.size() - 1) * page_size_tokens + tokens_in_last_page.
//
// The page chain is shared across all transformer layers — one allocation
// per page covers all 22 layers' K and V data for the same token range.
// (Actual storage is laid out per-layer in A2; the page IDs are layer-
// agnostic indices into per-layer storage arrays.)
struct RequestKVState {
    uint32_t request_id = 0;
    uint32_t page_size_tokens = 64;

    std::vector<uint32_t> page_table;   // ordered list of page IDs
    uint32_t tokens_in_last_page = 0;   // 0..page_size_tokens

    // Logical total = how many tokens of K/V history this request holds.
    size_t total_tokens() const {
        if (page_table.empty()) return 0;
        return (page_table.size() - 1) * static_cast<size_t>(page_size_tokens)
               + static_cast<size_t>(tokens_in_last_page);
    }

    // Reserve enough pages to hold `target_tokens` total. Allocates new
    // pages from `pool` as needed. Returns true on success, false if the
    // pool would be exhausted (no partial allocation; the page table is
    // unchanged on failure).
    bool reserve(PagePool& pool, size_t target_tokens);

    // Append one token's worth of slot at the end of the chain. Allocates
    // a new page if the current last page is full. Returns the (page_id,
    // slot_in_page) pair where the new K/V row should be written, or
    // nullopt if pool exhausted.
    std::optional<std::pair<uint32_t, uint32_t>> extend_one_token(PagePool& pool);

    // Resolve the (page_id, slot_in_page) for absolute token position `pos`
    // (0-based). Returns nullopt if pos is beyond total_tokens().
    std::optional<std::pair<uint32_t, uint32_t>> locate(size_t pos) const;

    // Release every page in the table back to the pool and clear state.
    void release_all(PagePool& pool);
};

// PagedKVStorage — per-layer Metal-backed bulk page storage.
//
// One MTLBuffer per layer holding `capacity_pages * page_bytes` bytes, where
// page_bytes = page_size_tokens * n_kv_heads * head_dim * dtype_bytes.
// Per-page layout (matches gather_kv_pages_f16 / scatter_kv_paged_f16):
//   [page_size_tokens, n_kv_heads, head_dim] of half (or float when
//   dtype_bytes == 4)
//
// This class owns the storage. Page-id allocation is delegated to a
// PagePool (request-level page tables hold the page IDs).
class PagedKVStorage {
public:
    PagedKVStorage(uint32_t capacity_pages,
                   uint32_t n_layers,
                   uint32_t page_size_tokens,
                   uint32_t n_kv_heads,
                   uint32_t head_dim,
                   uint32_t dtype_bytes);
    ~PagedKVStorage();
    PagedKVStorage(const PagedKVStorage&) = delete;
    PagedKVStorage& operator=(const PagedKVStorage&) = delete;

    bool initialize();  // Allocates MTLBuffers; returns false if Metal unavailable.

    void* k_buffer(size_t layer) const;  // void* bridged from id<MTLBuffer>
    void* v_buffer(size_t layer) const;

    uint32_t capacity_pages() const { return capacity_pages_; }
    uint32_t page_size_tokens() const { return page_size_tokens_; }
    uint32_t n_layers() const { return n_layers_; }
    uint32_t n_kv_heads() const { return n_kv_heads_; }
    uint32_t head_dim() const { return head_dim_; }
    uint32_t dtype_bytes() const { return dtype_bytes_; }
    size_t page_bytes() const {
        return static_cast<size_t>(page_size_tokens_) * n_kv_heads_ * head_dim_ * dtype_bytes_;
    }

private:
    uint32_t capacity_pages_;
    uint32_t n_layers_;
    uint32_t page_size_tokens_;
    uint32_t n_kv_heads_;
    uint32_t head_dim_;
    uint32_t dtype_bytes_;
    std::vector<void*> k_buffers_;  // size = n_layers
    std::vector<void*> v_buffers_;
};

} // namespace runtime
} // namespace mlc
