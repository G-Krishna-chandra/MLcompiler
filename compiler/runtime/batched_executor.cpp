#include "runtime/batched_executor.hpp"

#include "runtime/execution_plan_builder.hpp"
#include "runtime/kernel_registry.hpp"
#include "runtime/operator_backend.hpp"

namespace mlc {
namespace runtime {

namespace {
constexpr const char* kLogitsTensorName = "logits";
}

BatchedExecutor::BatchedExecutor(const Session& session)
    : session_(session),
      graph_(ExecutionPlanBuilder::BuildFromLoader(session.loader())),
      context_(session, &graph_),
      executor_(graph_, &BackendRegistry::Default(), &context_) {}

void BatchedExecutor::reset() {
    context_.clearStateTensors();
    request_positions_.clear();
}

size_t BatchedExecutor::known_position(uint32_t request_id) const {
    auto it = request_positions_.find(request_id);
    return it == request_positions_.end() ? 0 : it->second;
}

BatchedDecodeOutput BatchedExecutor::run_decode(const std::vector<RequestSlot>& slots) {
    BatchedDecodeOutput out;
    if (slots.empty()) return out;

    // B1: only N=1 is supported. The single-stream path runs unchanged
    // under the wrapper. N>1 returns a failure marker per slot so callers
    // can detect that the batched path isn't yet wired (B2 fills this in).
    if (slots.size() > 1) {
        for (const auto& slot : slots) {
            out.per_request.push_back({slot.request_id, /*success=*/false, {}});
        }
        return out;
    }

    const RequestSlot& slot = slots.front();
    BatchedDecodeOutput::PerRequest per;
    per.request_id = slot.request_id;

    context_.setToken(slot.input_token);
    context_.setSequencePosition(slot.sequence_position);
    context_.setActiveSequenceId(0);
    // Propagate the input token into any token-input tensor the graph defines.
    static const std::vector<std::string> kTokenInputs = {"tokens", "token_ids"};
    for (const auto& name : kTokenInputs) {
        if (graph_.tensors().count(name)) {
            context_.setTensor(name, {static_cast<float>(slot.input_token)});
        }
    }

    auto exec_res = executor_.run();
    per.success = exec_res.success;
    if (per.success) {
        if (const auto* logits = context_.getTensor(kLogitsTensorName)) {
            per.logits = *logits;
        }
        request_positions_[slot.request_id] = slot.sequence_position + 1;
    }
    out.per_request.push_back(std::move(per));
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
