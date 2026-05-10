#pragma once

#include "runtime/execution_plan_builder.hpp"
#include "runtime/execution_executor.hpp"
#include "runtime/execution_context.hpp"
#include "runtime/session.hpp"

#include <string>
#include <vector>

namespace mlc {
namespace runtime {

struct DecodeOptions {
    std::vector<uint64_t> tokens;
    size_t start_position = 0;
    size_t max_steps = 0; // 0 = run all tokens
    size_t top_k = 0;     // 0 = no top-k preview
    bool evict_on_full = true;
    bool cache_report = false;
};

struct DecodeStep {
    uint64_t token = 0;
    size_t position = 0;
    std::vector<float> logits;
    std::vector<uint64_t> top_indices;
    std::vector<float> top_probs;
    std::vector<std::string> trace;
    bool success = false;
    std::string error;
    bool cache_evicted = false;
};

struct CacheReportEntry {
    std::string name;
    uint32_t dtype = frontend::GGML_TYPE_F32;
    uint32_t quant_version = 1;
    size_t row_stride_bytes = 0;
    size_t byte_size = 0;
};

struct DecodeResult {
    bool success = false;
    std::vector<DecodeStep> steps;
    std::vector<CacheReportEntry> cache_report;
};

struct DecodeBatchOptions {
    std::vector<std::vector<uint64_t>> sequences;
    size_t start_position = 0;
    size_t max_steps = 0;
    size_t top_k = 0;
    bool evict_on_full = true;
    bool cache_report = false;
};

struct DecodeBatchResult {
    bool success = false;
    std::vector<DecodeResult> results;
};

// Utility to summarize KV caches for reporting/CLI.
std::vector<CacheReportEntry> BuildCacheReport(const ExecutionGraph& graph,
                                               const ExecutionContext& context,
                                               const frontend::GGUFLoader& loader);

// DecodeRunner drives a full execution graph token-by-token, handling sequence
// position updates and KV cache overwrite (wrap) using the model's context_length.
class DecodeRunner {
public:
    explicit DecodeRunner(const std::string& gguf_path);

    // Runs the decode loop over the provided token sequence.
    DecodeResult run(const DecodeOptions& options);

    // Runs multiple sequences independently, sharing a single model graph.
    DecodeBatchResult runBatch(const DecodeBatchOptions& options);

    const frontend::GGUFLoader& loader() const { return session_.loader(); }

private:
    Session session_;
};

} // namespace runtime
} // namespace mlc
