#pragma once

#include "runtime/execution_graph.hpp"
#include "runtime/execution_executor.hpp"
#include "runtime/session.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mlc {
namespace runtime {

struct RunConfig {
    uint64_t token_id = 0;
    size_t preview_length = 16;
    bool try_logits = true;
    bool simulate_plan = false;
    size_t simulate_limit = 0;
    bool execute_plan = false;
    size_t sequence_position = 0;
};

struct DryRunResult {
    ExecutionGraph plan;
    size_t num_layers = 0;
    size_t hidden_size = 0;
    std::vector<std::string> schedule;
    std::vector<std::string> execution_trace;

    std::string embedding_tensor;
    uint64_t token_id = 0;
    size_t embedding_dim = 0;
    std::vector<float> embedding_preview;

    std::string logits_tensor;
    size_t logits_dim = 0;
    std::vector<float> logits_preview;
    std::string logits_error;

    bool execution_ran = false;
    bool execution_success = false;
    std::string execution_error;
    std::vector<float> execution_output_preview;
};

class ModelRunner {
public:
    explicit ModelRunner(const std::string& gguf_path);

    DryRunResult dryRun(const RunConfig& config) const;

private:
    const frontend::GGUFLoader& loader() const { return session_.loader(); }

    static size_t readSize(const frontend::GGUFLoader& loader,
                           const std::string& key,
                           size_t fallback);
    static std::string selectEmbeddingTensor(const frontend::GGUFLoader& loader,
                                             size_t hidden_size);
    static std::string selectHeadTensor(const frontend::GGUFLoader& loader,
                                        size_t hidden_size);
    static bool tensorMatchesHidden(const frontend::GGUFTensorInfo& tensor,
                                    size_t hidden_size);
    static bool dtypeSupportedForLinear(uint32_t dtype);
    static std::vector<float> previewVector(const std::vector<float>& data,
                                            size_t preview_length);

    Session session_;
};

} // namespace runtime
} // namespace mlc
