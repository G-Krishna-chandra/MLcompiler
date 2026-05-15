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
    requests_.clear();
}

void BatchedExecutor::release_request(uint32_t request_id) {
    requests_.erase(request_id);
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
        ctx.setActiveSequenceId(0);
        prime_token_tensors(ctx, slot.input_token);

        auto exec_res = req.executor->run();
        per.success = exec_res.success;
        if (per.success) {
            if (const auto* logits = ctx.getTensor(kLogitsTensorName)) {
                per.logits = *logits;
            }
            req.next_position = slot.sequence_position + 1;
        }
        out.per_request.push_back(std::move(per));
    }
    return out;
}

std::vector<float> BatchedExecutor::run_prefill(uint32_t request_id,
                                                const std::vector<uint64_t>& tokens) {
    std::vector<float> last_logits;
    if (tokens.empty()) return last_logits;

    size_t cursor = known_position(request_id);
    for (uint64_t tok : tokens) {
        std::vector<RequestSlot> slot{{request_id, tok, cursor}};
        auto out = run_decode(slot);
        if (out.per_request.empty() || !out.per_request.front().success) {
            last_logits.clear();
            return last_logits;
        }
        last_logits = std::move(out.per_request.front().logits);
        ++cursor;
    }
    return last_logits;
}

} // namespace runtime
} // namespace mlc
