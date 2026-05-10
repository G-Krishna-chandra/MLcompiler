#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace mlc {
namespace runtime {

struct SamplerOptions {
    float temperature = 1.0f;
    size_t top_k = 40;   // 0 = disabled
    float top_p = 0.9f;  // 0 = disabled
};

// Samples the next token id from logits using temperature + top-k/top-p filtering.
// Returns UINT64_MAX if logits are empty or invalid.
uint64_t sampleLogits(const std::vector<float>& logits,
                      const SamplerOptions& options,
                      std::mt19937& rng);

} // namespace runtime
} // namespace mlc
