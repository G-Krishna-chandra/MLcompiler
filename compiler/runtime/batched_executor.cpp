#include "runtime/batched_executor.hpp"

#include "runtime/execution_plan_builder.hpp"
#include "runtime/kernel_registry.hpp"
#include "runtime/operator_backend.hpp"

namespace mlc {
namespace runtime {

namespace {
constexpr const char* kLogitsTensorName = "logits";
const std::vector<std::string>& tokenInputNames() {
    static const std::vector<std::string> names = {"tokens", "token_ids"};
    return names;
}
}

BatchedExecutor::BatchedExecutor(const Session& session)
    : session_(session),
      graph_(ExecutionPlanBuilder::BuildFromLoader(session.loader())) {}

void BatchedExecutor::reset() {
    if (page_pool_) {
        for (auto& kv : requests_) {
            if (kv.second.page_state) kv.second.page_state->release_all(*page_pool_);
        }
    }
    requests_.clear();
}

void BatchedExecutor::release_request(uint32_t request_id) {
    auto it = requests_.find(request_id);
    if (it == requests_.end()) return;
    if (page_pool_ && it->second.page_state) {
        it->second.page_state->release_all(*page_pool_);
    }
    requests_.erase(it);
}

void BatchedExecutor::attach_page_pool(PagePool* pool, uint32_t page_size_tokens) {
    page_pool_ = pool;
    page_size_tokens_ = page_size_tokens > 0 ? page_size_tokens : 64;
}

const RequestKVState* BatchedExecutor::page_state(uint32_t request_id) const {
    auto it = requests_.find(request_id);
    if (it == requests_.end()) return nullptr;
    return it->second.page_state.get();
}

bool BatchedExecutor::advance_page_state(PerRequest& req) {
    if (!page_pool_) return true;
    if (!req.page_state) {
        req.page_state = std::make_unique<RequestKVState>();
        req.page_state->page_size_tokens = page_size_tokens_;
    }
    return req.page_state->extend_one_token(*page_pool_).has_value();
}

size_t BatchedExecutor::known_position(uint32_t request_id) const {
    auto it = requests_.find(request_id);
    return it == requests_.end() ? 0 : it->second.next_position;
}

BatchedExecutor::PerRequest& BatchedExecutor::ensure_request(uint32_t request_id) {
    auto it = requests_.find(request_id);
    if (it != requests_.end()) return it->second;
    PerRequest pr;
    pr.context  = std::make_unique<ExecutionContext>(session_, &graph_);
    pr.executor = std::make_unique<ExecutionExecutor>(graph_,
                                                      &BackendRegistry::Default(),
                                                      pr.context.get());
    // Match chat-repl's startup: zero KV state and invalidate any cached
    // MetalBuffer wrappers. Without this, fuse-mode reads stale GPU
    // buffers cached by the executor singleton during a prior request.
    pr.context->clearStateTensors();
    auto inserted = requests_.emplace(request_id, std::move(pr));
    return inserted.first->second;
}

void BatchedExecutor::prime_token_tensors(ExecutionContext& ctx, uint64_t token) const {
    for (const auto& name : tokenInputNames()) {
        if (graph_.tensors().count(name)) {
            ctx.setTensor(name, {static_cast<float>(token)});
        }
    }
}

BatchedDecodeOutput BatchedExecutor::run_decode(const std::vector<RequestSlot>& slots) {
    BatchedDecodeOutput out;
    out.per_request.reserve(slots.size());
    if (slots.empty()) return out;

    for (const auto& slot : slots) {
        BatchedDecodeOutput::PerRequest per;
        per.request_id = slot.request_id;

        auto& req = ensure_request(slot.request_id);
        ExecutionContext& ctx = *req.context;

        ctx.setToken(slot.input_token);
        ctx.setSequencePosition(slot.sequence_position);
        prime_token_tensors(ctx, slot.input_token);

        auto exec_res = req.executor->run();
        per.success = exec_res.success;
        if (per.success) {
            if (const auto* logits = ctx.getTensor(kLogitsTensorName)) {
                per.logits = *logits;
            }
            req.next_position = slot.sequence_position + 1;
            // B3: extend the request's paged-KV table by one slot. Failure
            // here (pool exhaustion) is reported as a soft failure on the
            // slot — the per-request KV in the contiguous context remains
            // intact, but the scheduler should treat this as a stop signal.
            if (!advance_page_state(req)) {
                per.success = false;
                per.logits.clear();
            }
        }
        out.per_request.push_back(std::move(per));
    }
    return out;
}

std::vector<float> BatchedExecutor::run_prefill(uint32_t request_id,
                                                const std::vector<uint64_t>& tokens) {
    std::vector<float> last_logits;
    if (tokens.empty()) return last_logits;

    auto& req = ensure_request(request_id);
    ExecutionContext& ctx = *req.context;
    size_t cursor = known_position(request_id);

    // Multi-token batched prefill (matches chat-repl). When seq_len > 1 the
    // executor's fuse path is disabled internally and per-token fallthrough
    // builds the KV cache correctly. Looping single-token decode-with-fuse
    // from a zero KV cache silently produces all-zero logits — the encode
    // path's K/V writes don't take effect until the cache has been seeded
    // by the non-fuse multi-token path. (Confirmed by mlc decode having
    // the same bug; chat-repl avoids it via multi-token prefill.)
    ctx.setTokens(tokens);
    ctx.setSequencePosition(cursor);
    ctx.setActiveSequenceId(0);
    static const std::vector<std::string> kTokenInputs = {"tokens", "token_ids"};
    for (const auto& name : kTokenInputs) {
        if (graph_.tensors().count(name)) {
            std::vector<float> tf;
            tf.reserve(tokens.size());
            for (uint64_t t : tokens) tf.push_back(static_cast<float>(t));
            ctx.setTensor(name, std::move(tf));
        }
    }
    auto exec_res = req.executor->run();
    if (!exec_res.success) {
        return last_logits;
    }
    if (const auto* logits = ctx.getTensor("logits")) {
        last_logits = *logits;
    }
    req.next_position = cursor + tokens.size();
    // Reset seq_len so subsequent run_decode (single-token) sees the right shape.
    ctx.setSeqLen(1);

    // B3 paged-KV: extend the page table by one slot per prefilled token.
    // Failure here releases logits and signals failure.
    if (page_pool_) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (!advance_page_state(req)) {
                last_logits.clear();
                return last_logits;
            }
        }
    }
    return last_logits;
}

} // namespace runtime
} // namespace mlc
