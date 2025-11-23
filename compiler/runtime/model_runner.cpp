#include "runtime/model_runner.hpp"

#include "frontends/ggml_types.hpp"
#include "runtime/execution_plan_builder.hpp"
#include "runtime/execution_context.hpp"
#include "runtime/operator_backend.hpp"

#include <algorithm>
#include <stdexcept>

namespace mlc {
namespace runtime {

namespace {
const std::vector<std::string> kEmbeddingTensorCandidates = {
    "tok_embeddings.weight",
    "token_embd.weight",
    "token_emb.weight",
    "token_embeddings.weight",
    "embed_tokens.weight",
    "model.embed_tokens.weight",
    "transformer.wte.weight"
};

const std::vector<std::string> kHeadTensorCandidates = {
    "output.weight",
    "lm_head.weight",
    "model.output.weight",
    "model.lm_head.weight"
};
} // namespace

ModelRunner::ModelRunner(const std::string& gguf_path)
    : session_(gguf_path) {
}

DryRunResult ModelRunner::dryRun(const RunConfig& config) const {
    if (config.preview_length == 0) {
        throw std::runtime_error("preview_length must be non-zero");
    }

    DryRunResult result;
    const auto& loader_ref = loader();

    size_t layers = readSize(loader_ref, "llama.block_count", 0);
    size_t hidden = readSize(loader_ref, "llama.embedding_length", 0);
    if (layers == 0 || hidden == 0) {
        throw std::runtime_error("Model metadata missing llama.block_count or llama.embedding_length");
    }

    result.num_layers = layers;
    result.hidden_size = hidden;

    result.plan = ExecutionPlanBuilder::BuildFromLoader(loader_ref);
    result.schedule = result.plan.topologicalOrder();

    std::string embedding_tensor = selectEmbeddingTensor(loader_ref, hidden);
    result.embedding_tensor = embedding_tensor;
    result.token_id = config.token_id;

    auto embedding = session_.getEmbedding(embedding_tensor, config.token_id);
    result.embedding_dim = embedding.size();
    result.embedding_preview = previewVector(embedding, config.preview_length);

    if (config.try_logits) {
        std::string head_tensor = selectHeadTensor(loader_ref, hidden);
        if (!head_tensor.empty()) {
            result.logits_tensor = head_tensor;
            try {
                auto logits = session_.runLinear(head_tensor, embedding);
                result.logits_dim = logits.size();
                result.logits_preview = previewVector(logits, config.preview_length);
            } catch (const std::exception& e) {
                result.logits_error = e.what();
            }
        }
    }

    if (config.simulate_plan) {
        ExecutionExecutor executor(result.plan);
        auto sim = executor.run(config.simulate_limit);
        for (const auto& entry : sim.trace) {
            result.execution_trace.push_back(formatTraceEntry(entry));
        }
    }

    if (config.execute_plan) {
        ExecutionContext context(session_, &result.plan);
        context.setToken(config.token_id);
        context.setSequencePosition(config.sequence_position);
        std::vector<std::string> token_inputs = {"token_ids", "tokens"};
        for (const auto& name : token_inputs) {
            if (result.plan.tensors().count(name)) {
                context.setTensor(name,
                                  {static_cast<float>(config.token_id)});
            }
        }
        ExecutionExecutor executor(result.plan,
                                   &BackendRegistry::Default(),
                                   &context);
        auto exec_result = executor.run();
        result.execution_ran = true;
        result.execution_success = exec_result.success;
        if (!exec_result.success) {
            if (!exec_result.trace.empty()) {
                result.execution_error = formatTraceEntry(exec_result.trace.back());
            } else {
                result.execution_error = "Execution failed";
            }
        } else {
            const auto* logits = context.getTensor("logits");
            if (logits) {
                result.execution_output_preview = previewVector(*logits, config.preview_length);
            } else {
                result.execution_success = false;
                result.execution_error = "Logits tensor not produced";
            }
        }
    }

    return result;
}

size_t ModelRunner::readSize(const frontend::GGUFLoader& loader,
                             const std::string& key,
                             size_t fallback) {
    const auto& kv = loader.kvMetadata();
    auto it = kv.find(key);
    if (it == kv.end()) return fallback;

    switch (it->second.type) {
        case frontend::GGUFValueType::UINT32:
            return static_cast<size_t>(std::get<uint32_t>(it->second.data));
        case frontend::GGUFValueType::UINT64:
            return static_cast<size_t>(std::get<uint64_t>(it->second.data));
        default:
            return fallback;
    }
}

bool ModelRunner::tensorMatchesHidden(const frontend::GGUFTensorInfo& tensor,
                                      size_t hidden_size) {
    if (tensor.shape.size() != 2) return false;
    return tensor.shape[1] == hidden_size || tensor.shape[0] == hidden_size;
}

bool ModelRunner::dtypeSupportedForLinear(uint32_t dtype) {
    switch (dtype) {
        case frontend::GGML_TYPE_F32:
        case frontend::GGML_TYPE_F16:
        case frontend::GGML_TYPE_BF16:
        case frontend::GGML_TYPE_I8:
        case frontend::GGML_TYPE_Q4_0:
        case frontend::GGML_TYPE_Q4_1:
        case frontend::GGML_TYPE_Q5_0:
        case frontend::GGML_TYPE_Q5_1:
        case frontend::GGML_TYPE_Q8_0:
        case frontend::GGML_TYPE_Q8_1:
        case frontend::GGML_TYPE_Q2_K:
        case frontend::GGML_TYPE_Q3_K:
        case frontend::GGML_TYPE_Q4_K:
        case frontend::GGML_TYPE_Q5_K:
        case frontend::GGML_TYPE_Q6_K:
        case frontend::GGML_TYPE_Q8_K:
            return true;
        default:
            return false;
    }
}

std::string ModelRunner::selectEmbeddingTensor(const frontend::GGUFLoader& loader_ref,
                                               size_t hidden_size) {
    const auto& tensors = loader_ref.tensors();
    for (const auto& name : kEmbeddingTensorCandidates) {
        auto it = tensors.find(name);
        if (it != tensors.end() && tensorMatchesHidden(it->second, hidden_size)) {
            return name;
        }
    }

    for (const auto& kv : tensors) {
        if (tensorMatchesHidden(kv.second, hidden_size)) {
            return kv.first;
        }
    }

    throw std::runtime_error("Unable to locate embedding tensor matching hidden size");
}

std::string ModelRunner::selectHeadTensor(const frontend::GGUFLoader& loader_ref,
                                          size_t hidden_size) {
    const auto& tensors = loader_ref.tensors();
    auto pick = [&](const std::string& name) -> std::string {
        auto it = tensors.find(name);
        if (it == tensors.end()) return {};
        if (!tensorMatchesHidden(it->second, hidden_size)) return {};
        if (!dtypeSupportedForLinear(it->second.dtype)) return {};
        return name;
    };

    for (const auto& name : kHeadTensorCandidates) {
        auto chosen = pick(name);
        if (!chosen.empty()) {
            return chosen;
        }
    }

    for (const auto& kv : tensors) {
        if (kv.second.shape.size() != 2) continue;
        if (!tensorMatchesHidden(kv.second, hidden_size)) continue;
        if (!dtypeSupportedForLinear(kv.second.dtype)) continue;
        return kv.first;
    }

    return {};
}

std::vector<float> ModelRunner::previewVector(const std::vector<float>& data,
                                              size_t preview_length) {
    size_t count = std::min(preview_length, data.size());
    return std::vector<float>(data.begin(), data.begin() + count);
}

} // namespace runtime
} // namespace mlc
