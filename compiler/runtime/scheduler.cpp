#include "runtime/scheduler.hpp"

#include <algorithm>
#include <cassert>
#include <utility>

namespace mlc {
namespace runtime {

Scheduler::Scheduler(IBatchedExecutor* executor, size_t max_batch_size)
    : executor_(executor),
      max_batch_size_(max_batch_size > 0 ? max_batch_size : 1) {}

uint32_t Scheduler::add_request(std::vector<uint64_t> prompt_tokens,
                                GenerationParams params) {
    uint32_t id = next_id_++;
    Request req;
    req.id = id;
    req.prompt_tokens = std::move(prompt_tokens);
    req.params = params;
    req.state = State::Queued;
    queued_.push_back(std::move(req));
    return id;
}

void Scheduler::set_token_callback(TokenCallback cb) {
    token_cb_ = std::move(cb);
}

void Scheduler::set_complete_callback(CompleteCallback cb) {
    complete_cb_ = std::move(cb);
}

bool Scheduler::empty() const {
    return queued_.empty() && active_.empty();
}

void Scheduler::admit_new() {
    while (!queued_.empty() && active_.size() < max_batch_size_) {
        Request req = std::move(queued_.front());
        queued_.pop_front();
        req.state = req.prompt_tokens.empty() ? State::Decoding : State::Prefilling;
        active_.push_back(std::move(req));
    }
}

void Scheduler::prefill_one_active() {
    // v1 policy: prefill one request per tick (no mixed-batch prefill).
    for (auto& req : active_) {
        if (req.state != State::Prefilling) continue;
        if (req.prompt_tokens.empty()) {
            req.state = State::Decoding;
            return;
        }
        auto logits = executor_->run_prefill(req.id, req.prompt_tokens);
        if (logits.empty()) {
            req.state = State::Failed;
            req.failure_reason = "prefill failed";
            return;
        }
        // After prefill, the model has consumed every prompt token; sequence
        // position = prompt_tokens.size(). Sample first decode token from
        // the prefill's last-position logits.
        req.position = req.prompt_tokens.size();
        uint64_t tok = argmax(logits);
        req.last_token = tok;
        req.generated_tokens.push_back(tok);
        ++stats_.total_tokens_generated;
        if (token_cb_) token_cb_(req.id, tok);
        // EOS or budget can complete the request immediately after prefill.
        if (req.params.eos_token_id >= 0 &&
            tok == static_cast<uint64_t>(req.params.eos_token_id)) {
            req.state = State::Complete;
        } else if (req.generated_tokens.size() >= req.params.max_new_tokens) {
            req.state = State::Complete;
        } else {
            req.state = State::Decoding;
        }
        return;
    }
}

void Scheduler::decode_active() {
    std::vector<RequestSlot> slots;
    slots.reserve(active_.size());
    std::vector<size_t> active_indices;
    active_indices.reserve(active_.size());

    for (size_t i = 0; i < active_.size(); ++i) {
        Request& req = active_[i];
        if (req.state != State::Decoding) continue;
        if (req.generated_tokens.size() >= req.params.max_new_tokens) {
            req.state = State::Complete;
            continue;
        }
        slots.push_back({req.id, req.last_token, req.position});
        active_indices.push_back(i);
    }
    if (slots.empty()) return;

    auto out = executor_->run_decode(slots);
    ++stats_.total_decode_steps;
    if (out.per_request.size() != slots.size()) {
        // Unexpected: mark all dispatched requests failed.
        for (size_t idx : active_indices) {
            active_[idx].state = State::Failed;
            active_[idx].failure_reason = "decode output size mismatch";
        }
        return;
    }
    for (size_t k = 0; k < out.per_request.size(); ++k) {
        const auto& per = out.per_request[k];
        size_t i = active_indices[k];
        Request& req = active_[i];
        if (!per.success || per.logits.empty()) {
            req.state = State::Failed;
            req.failure_reason = "decode failed";
            continue;
        }
        uint64_t tok = argmax(per.logits);
        req.last_token = tok;
        ++req.position;
        req.generated_tokens.push_back(tok);
        ++stats_.total_tokens_generated;
        if (token_cb_) token_cb_(req.id, tok);
        if (req.params.eos_token_id >= 0 &&
            tok == static_cast<uint64_t>(req.params.eos_token_id)) {
            req.state = State::Complete;
            continue;
        }
        if (req.generated_tokens.size() >= req.params.max_new_tokens) {
            req.state = State::Complete;
        }
    }
}

void Scheduler::reap_completed() {
    auto split = std::stable_partition(active_.begin(), active_.end(),
                                       [](const Request& r) {
                                           return r.state != State::Complete &&
                                                  r.state != State::Failed;
                                       });
    for (auto it = split; it != active_.end(); ++it) {
        if (executor_) executor_->release_request(it->id);
        if (complete_cb_) complete_cb_(*it);
        ++stats_.total_requests_completed;
        completed_.push_back(std::move(*it));
    }
    active_.erase(split, active_.end());
}

bool Scheduler::tick() {
    bool any = false;
    size_t before_active = active_.size();
    admit_new();
    if (active_.size() > before_active) any = true;
    prefill_one_active();
    size_t before_steps = stats_.total_decode_steps;
    decode_active();
    if (stats_.total_decode_steps > before_steps) any = true;
    size_t before_completed = completed_.size();
    reap_completed();
    if (completed_.size() > before_completed) any = true;
    return any || !empty();
}

size_t Scheduler::run_until_idle() {
    size_t initial_steps = stats_.total_decode_steps;
    while (!empty()) {
        if (!tick()) break;
    }
    return stats_.total_decode_steps - initial_steps;
}

std::vector<Scheduler::Request> Scheduler::drain_completed() {
    auto out = std::move(completed_);
    completed_.clear();
    return out;
}

uint64_t Scheduler::argmax(const std::vector<float>& logits) {
    auto it = std::max_element(logits.begin(), logits.end());
    return static_cast<uint64_t>(std::distance(logits.begin(), it));
}

} // namespace runtime
} // namespace mlc
