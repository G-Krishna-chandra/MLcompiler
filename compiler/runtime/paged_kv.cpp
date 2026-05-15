#include "runtime/paged_kv.hpp"

#include <cassert>
#include <stdexcept>

namespace mlc {
namespace runtime {

PagePool::PagePool(uint32_t capacity) : capacity_(capacity) {
    free_list_.reserve(capacity);
    // Initial free list contains every page ID, with ID 0 at the bottom so
    // allocate() returns IDs in ascending order on a fresh pool.
    for (uint32_t i = capacity; i-- > 0; ) {
        free_list_.push_back(i);
    }
}

std::optional<uint32_t> PagePool::allocate() {
    if (free_list_.empty()) return std::nullopt;
    uint32_t id = free_list_.back();
    free_list_.pop_back();
    ++in_use_;
    return id;
}

bool PagePool::allocate_n(size_t n, std::vector<uint32_t>& out) {
    if (n > free_list_.size()) return false;
    for (size_t i = 0; i < n; ++i) {
        out.push_back(free_list_.back());
        free_list_.pop_back();
    }
    in_use_ += n;
    return true;
}

void PagePool::release(uint32_t page_id) {
    assert(page_id < capacity_ && "release: page_id out of range");
    assert(in_use_ > 0 && "release: pool is empty");
    free_list_.push_back(page_id);
    --in_use_;
}

bool RequestKVState::reserve(PagePool& pool, size_t target_tokens) {
    if (page_size_tokens == 0) return false;
    size_t needed_pages = (target_tokens + page_size_tokens - 1) / page_size_tokens;
    if (needed_pages <= page_table.size()) return true;
    size_t to_alloc = needed_pages - page_table.size();
    return pool.allocate_n(to_alloc, page_table);
}

std::optional<std::pair<uint32_t, uint32_t>>
RequestKVState::extend_one_token(PagePool& pool) {
    if (page_size_tokens == 0) return std::nullopt;
    // If the last page is full (or there is no page yet), allocate a new one.
    if (page_table.empty() || tokens_in_last_page == page_size_tokens) {
        auto id = pool.allocate();
        if (!id) return std::nullopt;
        page_table.push_back(*id);
        tokens_in_last_page = 0;
    }
    uint32_t page_id = page_table.back();
    uint32_t slot = tokens_in_last_page;
    ++tokens_in_last_page;
    return std::make_pair(page_id, slot);
}

std::optional<std::pair<uint32_t, uint32_t>>
RequestKVState::locate(size_t pos) const {
    if (page_size_tokens == 0) return std::nullopt;
    if (pos >= total_tokens()) return std::nullopt;
    size_t idx = pos / page_size_tokens;
    size_t slot = pos % page_size_tokens;
    return std::make_pair(page_table[idx], static_cast<uint32_t>(slot));
}

void RequestKVState::release_all(PagePool& pool) {
    for (uint32_t id : page_table) pool.release(id);
    page_table.clear();
    tokens_in_last_page = 0;
}

} // namespace runtime
} // namespace mlc
