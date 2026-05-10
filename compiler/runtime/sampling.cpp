#include "runtime/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace mlc {
namespace runtime {

namespace {
struct ScoredIndex {
    float logit;
    uint64_t index;
};
} // namespace

uint64_t sampleLogits(const std::vector<float>& logits,
                      const SamplerOptions& options,
                      std::mt19937& rng) {
    if (logits.empty()) return std::numeric_limits<uint64_t>::max();

    float temperature = options.temperature;
    if (temperature <= 0.0f) temperature = 1.0f;

    std::vector<ScoredIndex> scored;
    scored.reserve(logits.size());
    for (uint64_t i = 0; i < logits.size(); ++i) {
        scored.push_back({logits[i] / temperature, i});
    }

    if (options.top_k > 0 && scored.size() > options.top_k) {
        std::nth_element(scored.begin(),
                         scored.begin() + options.top_k,
                         scored.end(),
                         [](const ScoredIndex& a, const ScoredIndex& b) {
                             return a.logit > b.logit;
                         });
        scored.resize(options.top_k);
    }

    // Softmax with optional top-p truncation.
    float max_logit = std::max_element(scored.begin(), scored.end(),
                                       [](const ScoredIndex& a, const ScoredIndex& b) {
                                           return a.logit < b.logit;
                                       })
                          ->logit;
    std::vector<float> probs(scored.size(), 0.0f);
    float sum = 0.0f;
    for (size_t i = 0; i < scored.size(); ++i) {
        float val = std::exp(scored[i].logit - max_logit);
        probs[i] = val;
        sum += val;
    }
    if (sum <= 0.0f) return scored.front().index;
    for (float& p : probs) p /= sum;

    // Apply top-p cumulative cutoff if requested.
    if (options.top_p > 0.0f && options.top_p < 1.0f) {
        std::vector<size_t> order(scored.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return probs[a] > probs[b];
        });
        float cumulative = 0.0f;
        std::vector<bool> keep(scored.size(), false);
        for (size_t idx : order) {
            cumulative += probs[idx];
            keep[idx] = true;
            if (cumulative >= options.top_p) break;
        }
        float new_sum = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i) {
            if (!keep[i]) {
                probs[i] = 0.0f;
            } else {
                new_sum += probs[i];
            }
        }
        if (new_sum > 0.0f) {
            for (float& p : probs) p /= new_sum;
        }
    }

    std::discrete_distribution<uint64_t> dist(probs.begin(), probs.end());
    uint64_t chosen = dist(rng);
    if (chosen >= scored.size()) {
        return scored.back().index;
    }
    return scored[chosen].index;
}

} // namespace runtime
} // namespace mlc
