#include "runtime/execution_plan_builder.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "pipeline/ir_pipeline.hpp"

namespace mlc {
namespace runtime {

ExecutionGraph ExecutionPlanBuilder::BuildFromLoader(const frontend::GGUFLoader& loader) {
    pipeline::IRPipeline pipeline;
    auto result = pipeline.Run(loader);
    return result.exec_graph;
}

ExecutionGraph ExecutionPlanBuilder::BuildToy(size_t num_layers, size_t hidden_size) {
    if (num_layers == 0 || hidden_size == 0) {
        throw std::runtime_error("Toy execution graph requires non-zero dimensions");
    }
    ModelConfig config;
    config.num_layers = num_layers;
    config.hidden_size = hidden_size;
    config.head_count = 8;
    config.kv_head_count = 8;
    config.context_length = 128;
    config.vocab_size = 32000;
    config.head_dim = hidden_size / std::max<size_t>(1, config.head_count);
    return Build(config);
}

ExecutionGraph ExecutionPlanBuilder::Build(const ModelConfig& base_config) {
    if (base_config.num_layers == 0 || base_config.hidden_size == 0) {
        throw std::runtime_error("Execution plan requires non-zero layers and hidden size");
    }

    ModelConfig config = base_config;
    if (config.head_count == 0) config.head_count = 1;
    if (config.kv_head_count == 0) config.kv_head_count = config.head_count;
    if (config.head_dim == 0) {
        config.head_dim = config.hidden_size / std::max<size_t>(1, config.head_count);
    }
    if (config.head_dim == 0) config.head_dim = 1;
    if (config.context_length == 0) config.context_length = 1;
    if (config.vocab_size == 0) config.vocab_size = config.hidden_size;
    if (config.rotary_dim == 0) {
        config.rotary_dim = config.head_dim;
    }
    if (config.rope_freq_base <= 0.0f) {
        config.rope_freq_base = 10000.0f;
    }
    if (config.rope_freq_scale <= 0.0f) {
        config.rope_freq_scale = 1.0f;
    }

    ExecutionGraph graph;
    graph.setModelConfig(config);

    auto& tokens = graph.addTensor("tokens", {1}, ir::DataType::I4);
    tokens.metadata["role"] = "input_tokens";

    auto& hidden0 = graph.addTensor("hidden_state_0",
                                    {static_cast<int64_t>(config.hidden_size)},
                                    ir::DataType::F32);
    hidden0.metadata["role"] = "hidden_state";

    auto& embed_node = graph.addNode("embedding_lookup",
                                     ExecOpType::Embedding,
                                     {"tokens"},
                                     {"hidden_state_0"},
                                     BackendKind::Metal);
    embed_node.attributes["hidden_size"] = static_cast<float>(config.hidden_size);

    std::string current = "hidden_state_0";
    int64_t hidden_shape = static_cast<int64_t>(config.hidden_size);
    int64_t kv_context = static_cast<int64_t>(config.context_length);
    int64_t head_dim = static_cast<int64_t>(config.head_dim);
    int64_t n_heads = static_cast<int64_t>(config.head_count);
    int64_t n_kv_heads = static_cast<int64_t>(config.kv_head_count);

    for (size_t i = 0; i < config.num_layers; ++i) {
        std::string attn_out = "layer_" + std::to_string(i) + "_attn";
        std::string ffn_out = "layer_" + std::to_string(i) + "_ffn";
        std::string k_cache = "layer_" + std::to_string(i) + "_kv_cache_k";
        std::string v_cache = "layer_" + std::to_string(i) + "_kv_cache_v";

        auto& attn_tensor = graph.addTensor(attn_out, {hidden_shape}, ir::DataType::F32);
        attn_tensor.metadata["role"] = "attention_output";
        attn_tensor.metadata["layer"] = std::to_string(i);

        auto& ffn_tensor = graph.addTensor(ffn_out, {hidden_shape}, ir::DataType::F32);
        ffn_tensor.metadata["role"] = "ffn_output";
        ffn_tensor.metadata["layer"] = std::to_string(i);

        auto& k_tensor = graph.addTensor(k_cache,
                                         {n_kv_heads, kv_context, head_dim},
                                         ir::DataType::F32);
        k_tensor.is_state = true;
        k_tensor.metadata["role"] = "kv_cache";
        k_tensor.metadata["kv_kind"] = "k";
        k_tensor.metadata["layer"] = std::to_string(i);

        auto& v_tensor = graph.addTensor(v_cache,
                                         {n_kv_heads, kv_context, head_dim},
                                         ir::DataType::F32);
        v_tensor.is_state = true;
        v_tensor.metadata["role"] = "kv_cache";
        v_tensor.metadata["kv_kind"] = "v";
        v_tensor.metadata["layer"] = std::to_string(i);

        auto& attn_node = graph.addNode("layer_" + std::to_string(i) + "_attention",
                                        ExecOpType::Attention,
                                        {current},
                                        {attn_out, k_cache, v_cache},
                                        BackendKind::Metal);
        attn_node.attributes["layer_index"] = static_cast<float>(i);
        attn_node.attributes["heads"] = static_cast<float>(n_heads);
        attn_node.attributes["kv_heads"] = static_cast<float>(n_kv_heads);
        attn_node.attributes["head_dim"] = static_cast<float>(head_dim);
        attn_node.annotations["kv_cache_k"] = k_cache;
        attn_node.annotations["kv_cache_v"] = v_cache;

        auto& ffn_node = graph.addNode("layer_" + std::to_string(i) + "_feedforward",
                                       ExecOpType::FeedForward,
                                       {attn_out},
                                       {ffn_out},
                                       BackendKind::Auto);
        ffn_node.attributes["layer_index"] = static_cast<float>(i);

        current = ffn_out;
    }

    graph.addTensor("norm_out", {hidden_shape}, ir::DataType::F32);
    auto& norm_node = graph.addNode("final_norm",
                                    ExecOpType::Norm,
                                    {current},
                                    {"norm_out"},
                                    BackendKind::CPU);
    norm_node.attributes["hidden_size"] = static_cast<float>(config.hidden_size);

    auto& logits = graph.addTensor("logits",
                                   {static_cast<int64_t>(config.vocab_size)},
                                   ir::DataType::F32);
    logits.metadata["role"] = "logits";
    auto& head_node = graph.addNode("lm_head",
                                    ExecOpType::Output,
                                    {"norm_out"},
                                    {"logits"},
                                    BackendKind::CPU);
    head_node.attributes["vocab_size"] = static_cast<float>(config.vocab_size);

    return graph;
}

bool ExecutionPlanBuilder::valueToSize(const frontend::GGUFValue& value, size_t& out) {
    switch (value.type) {
        case frontend::GGUFValueType::UINT32:
            out = static_cast<size_t>(std::get<uint32_t>(value.data));
            return true;
        case frontend::GGUFValueType::UINT64:
            out = static_cast<size_t>(std::get<uint64_t>(value.data));
            return true;
        case frontend::GGUFValueType::INT32: {
            int32_t v = std::get<int32_t>(value.data);
            if (v < 0) return false;
            out = static_cast<size_t>(v);
            return true;
        }
        case frontend::GGUFValueType::INT64: {
            int64_t v = std::get<int64_t>(value.data);
            if (v < 0) return false;
            out = static_cast<size_t>(v);
            return true;
        }
        case frontend::GGUFValueType::FLOAT32: {
            float v = std::get<float>(value.data);
            if (v < 0.0f) return false;
            out = static_cast<size_t>(v);
            return true;
        }
        case frontend::GGUFValueType::FLOAT64: {
            double v = std::get<double>(value.data);
            if (v < 0.0) return false;
            out = static_cast<size_t>(v);
            return true;
        }
        default:
            return false;
    }
}

std::string ExecutionPlanBuilder::toLower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

size_t ExecutionPlanBuilder::readSize(const frontend::GGUFLoader& loader,
                                      const std::vector<std::string>& keys,
                                      size_t default_value) {
    const auto& kv = loader.kvMetadata();
    for (const auto& key : keys) {
        auto it = kv.find(key);
        if (it == kv.end()) continue;
        size_t value = 0;
        if (valueToSize(it->second, value)) {
            return value;
        }
    }
    return default_value;
}

size_t ExecutionPlanBuilder::findSizeByKeywords(const frontend::GGUFLoader& loader,
                                                const std::vector<std::string>& keywords) {
    if (keywords.empty()) return 0;
    const auto& kv = loader.kvMetadata();
    std::vector<std::string> lowered_keywords;
    lowered_keywords.reserve(keywords.size());
    for (const auto& kw : keywords) {
        lowered_keywords.push_back(toLower(kw));
    }

    for (const auto& [key, value] : kv) {
        std::string lower_key = toLower(key);
        bool matches = true;
        for (const auto& kw : lowered_keywords) {
            if (lower_key.find(kw) == std::string::npos) {
                matches = false;
                break;
            }
        }
        if (!matches) continue;
        size_t size_value = 0;
        if (valueToSize(value, size_value)) {
            return size_value;
        }
    }
    return 0;
}

size_t ExecutionPlanBuilder::inferVocabFromTokens(const frontend::GGUFLoader& loader) {
    const auto& kv = loader.kvMetadata();
    auto it = kv.find("tokenizer.ggml.tokens");
    if (it == kv.end()) return 0;
    if (it->second.type != frontend::GGUFValueType::ARRAY) return 0;
    const auto& arr = std::get<std::vector<frontend::GGUFValue>>(it->second.data);
    return arr.size();
}

} // namespace runtime
} // namespace mlc
