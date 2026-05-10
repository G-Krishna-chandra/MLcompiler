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
    // Propagate sliding window / architecture hints if present.
    auto config = result.exec_graph.modelConfig();
    // Sliding window hint can be carried via kv metadata; keep it in model config for mask generation.
    const auto& kv = loader.kvMetadata();
    auto it = kv.find("attention.sliding_window");
    if (it != kv.end()) {
        if (it->second.type == frontend::GGUFValueType::UINT32) {
            config.context_length = std::max<size_t>(config.context_length,
                                                     static_cast<size_t>(std::get<uint32_t>(it->second.data)));
        } else if (it->second.type == frontend::GGUFValueType::UINT64) {
            config.context_length = std::max<size_t>(config.context_length,
                                                     static_cast<size_t>(std::get<uint64_t>(it->second.data)));
        }
    }
    // Activation type hint (e.g., Gemma uses GeGLU); default remains SiLU.
    auto act_it = kv.find("activation_type");
    if (act_it != kv.end() && act_it->second.type == frontend::GGUFValueType::STRING) {
        config.activation = std::get<std::string>(act_it->second.data);
    }

    if (config.hidden_size == 0) {
        throw std::runtime_error("Missing hidden_size in model config after IR pipeline");
    }

    // Head weight / bias selection and vocab inference.
    const std::string head_name = selectHeadTensor(loader, config.hidden_size);
    if (head_name.empty()) {
        throw std::runtime_error("Failed to locate lm_head weight matching hidden size");
    }
    const auto& tensors = loader.tensors();
    const auto& head_info = tensors.at(head_name);
    size_t vocab = 0;
    if (head_info.shape.size() == 2) {
        vocab = (static_cast<size_t>(head_info.shape[0]) == config.hidden_size)
                    ? static_cast<size_t>(head_info.shape[1])
                    : static_cast<size_t>(head_info.shape[0]);
    }
    if (vocab == 0) {
        vocab = inferVocabFromTokens(loader);
    }
    if (vocab == 0) {
        throw std::runtime_error("Unable to infer vocabulary size from head weight or tokens");
    }
    if (config.vocab_size == 0) {
        config.vocab_size = vocab;
    }
    config.head_weight_name = head_name;
    config.head_bias_name = selectHeadBias(loader, head_name, config.hidden_size);

    result.exec_graph.setModelConfig(config);
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
    // BuildToy is for tests/diagnostics that have no real GGUF weights; satisfy
    // Build()'s precondition with a synthetic head name. The lm_head node it
    // creates references this name but no kernel will actually find a parameter
    // for it — toy callers only inspect graph structure, not run inference.
    config.head_weight_name = "lm_head.weight";
    return Build(config);
}

ExecutionGraph ExecutionPlanBuilder::BuildForTests(const ModelConfig& config) {
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

    if (config.head_weight_name.empty()) {
        throw std::runtime_error("Execution plan requires a head_weight_name");
    }
    if (config.vocab_size == 0) {
        throw std::runtime_error("Execution plan requires non-zero vocab_size");
    }

    ExecutionGraph graph;
    graph.setModelConfig(config);

    auto& tokens = graph.addTensor("tokens", {1}, ir::DataType::I4);
    tokens.metadata["role"] = "input_tokens";

    // Track grouped-query intent and per-family activation defaults for downstream kernels.
    bool grouped_query = (config.kv_head_count > 0) && (config.head_count > config.kv_head_count);
    std::string ffn_activation = config.activation;
    if (ffn_activation.empty()) {
        if (config.family == ArchitectureFamily::Gemma) {
            ffn_activation = "geglu";
        } else {
            ffn_activation = "silu";
        }
    }

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

    bool prefer_layer_norm = false;
    if (config.family != ArchitectureFamily::Unknown) {
        prefer_layer_norm = (config.family != ArchitectureFamily::Llama &&
                             config.family != ArchitectureFamily::Mistral);
    } else {
        std::string arch_lower = config.architecture;
        std::transform(arch_lower.begin(), arch_lower.end(), arch_lower.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (arch_lower.find("gpt") != std::string::npos &&
            arch_lower.find("llama") == std::string::npos &&
            arch_lower.find("mistral") == std::string::npos) {
            prefer_layer_norm = true;
        }
    }

    for (size_t i = 0; i < config.num_layers; ++i) {
        std::string prefix = "layer_" + std::to_string(i);
        std::string norm1_out = prefix + "_norm1";
        std::string attn_out = prefix + "_attn";
        std::string resid1_out = prefix + "_resid1";
        std::string norm2_out = prefix + "_norm2";
        std::string ffn_out = prefix + "_ffn";
        std::string resid2_out = prefix + "_resid2";
        std::string k_cache = prefix + "_kv_cache_k";
        std::string v_cache = prefix + "_kv_cache_v";

        auto& norm1_tensor = graph.addTensor(norm1_out, {hidden_shape}, ir::DataType::F32);
        norm1_tensor.metadata["role"] = "norm1_output";
        norm1_tensor.metadata["layer"] = std::to_string(i);

        auto& attn_tensor = graph.addTensor(attn_out, {hidden_shape}, ir::DataType::F32);
        attn_tensor.metadata["role"] = "attention_output";
        attn_tensor.metadata["layer"] = std::to_string(i);

        auto& resid1_tensor = graph.addTensor(resid1_out, {hidden_shape}, ir::DataType::F32);
        resid1_tensor.metadata["role"] = "residual_1";
        resid1_tensor.metadata["layer"] = std::to_string(i);

        auto& norm2_tensor = graph.addTensor(norm2_out, {hidden_shape}, ir::DataType::F32);
        norm2_tensor.metadata["role"] = "norm2_output";
        norm2_tensor.metadata["layer"] = std::to_string(i);

        auto& ffn_tensor = graph.addTensor(ffn_out, {hidden_shape}, ir::DataType::F32);
        ffn_tensor.metadata["role"] = "ffn_output";
        ffn_tensor.metadata["layer"] = std::to_string(i);

        auto& resid2_tensor = graph.addTensor(resid2_out, {hidden_shape}, ir::DataType::F32);
        resid2_tensor.metadata["role"] = "residual_2";
        resid2_tensor.metadata["layer"] = std::to_string(i);

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

        auto& norm1_node = graph.addNode(prefix + "_norm1",
                                         ExecOpType::Norm,
                                         {current},
                                         {norm1_out},
                                         BackendKind::Auto);
        norm1_node.attributes["layer_index"] = static_cast<float>(i);
        norm1_node.annotations["norm_kind"] = prefer_layer_norm ? "layer" : "rms";

        auto& attn_node = graph.addNode(prefix + "_attention",
                                        ExecOpType::Attention,
                                        {norm1_out},
                                        {attn_out, k_cache, v_cache},
                                        BackendKind::Metal);
        attn_node.attributes["layer_index"] = static_cast<float>(i);
        attn_node.attributes["heads"] = static_cast<float>(n_heads);
        attn_node.attributes["kv_heads"] = static_cast<float>(n_kv_heads);
        attn_node.attributes["head_dim"] = static_cast<float>(head_dim);
        attn_node.annotations["kv_cache_k"] = k_cache;
        attn_node.annotations["kv_cache_v"] = v_cache;
        attn_node.annotations["grouped_query"] = grouped_query ? "true" : "false";
        attn_node.grouped_query = grouped_query;

        auto& add1_node = graph.addNode(prefix + "_residual_add1",
                                        ExecOpType::Add,
                                        {current, attn_out},
                                        {resid1_out},
                                        BackendKind::Auto);
        add1_node.attributes["layer_index"] = static_cast<float>(i);

        auto& norm2_node = graph.addNode(prefix + "_norm2",
                                         ExecOpType::Norm,
                                         {resid1_out},
                                         {norm2_out},
                                         BackendKind::Auto);
        norm2_node.attributes["layer_index"] = static_cast<float>(i);
        norm2_node.annotations["norm_kind"] = prefer_layer_norm ? "layer" : "rms";

        auto& ffn_node = graph.addNode(prefix + "_feedforward",
                                       ExecOpType::FeedForward,
                                       {norm2_out},
                                       {ffn_out},
                                       BackendKind::Auto);
        ffn_node.attributes["layer_index"] = static_cast<float>(i);
        ffn_node.annotations["activation"] = ffn_activation;

        auto& add2_node = graph.addNode(prefix + "_residual_add2",
                                        ExecOpType::Add,
                                        {resid1_out, ffn_out},
                                        {resid2_out},
                                        BackendKind::Auto);
        add2_node.attributes["layer_index"] = static_cast<float>(i);
        // Propagate FFN bias annotation if present so fused add+bias kernels can be used.
        std::string ffn_bias = "ffn_down_bias_" + std::to_string(i);
        if (graph.getTensor(ffn_bias) != nullptr) {
            add2_node.annotations["bias"] = ffn_bias;
        }

        current = resid2_out;
    }

    graph.addTensor("norm_out", {hidden_shape}, ir::DataType::F32);
    auto& norm_node = graph.addNode("final_norm",
                                    ExecOpType::Norm,
                                    {current},
                                    {"norm_out"},
                                    BackendKind::CPU);
    norm_node.attributes["hidden_size"] = static_cast<float>(config.hidden_size);
    norm_node.annotations["norm_kind"] = prefer_layer_norm ? "layer" : "rms";

    auto& logits = graph.addTensor("logits",
                                   {static_cast<int64_t>(config.vocab_size)},
                                   ir::DataType::F32);
    logits.metadata["role"] = "logits";
    auto& head_node = graph.addNode("lm_head",
                                    ExecOpType::Linear,
                                    {"norm_out"},
                                    {"logits"},
                                    BackendKind::Metal);
    head_node.attributes["vocab_size"] = static_cast<float>(config.vocab_size);
    head_node.annotations["weight"] = config.head_weight_name;
    if (!config.head_bias_name.empty()) {
        head_node.annotations["bias"] = config.head_bias_name;
    }

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

bool ExecutionPlanBuilder::dtypeSupportedForLinear(uint32_t dtype) {
    using namespace frontend;
    switch (dtype) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_I8:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
            return true;
        default:
            return false;
    }
}

bool ExecutionPlanBuilder::tensorMatchesHidden(const frontend::GGUFTensorInfo& info, size_t hidden) {
    if (info.shape.size() != 2) return false;
    return static_cast<size_t>(info.shape[0]) == hidden || static_cast<size_t>(info.shape[1]) == hidden;
}

std::string ExecutionPlanBuilder::selectHeadTensor(const frontend::GGUFLoader& loader, size_t hidden_size) {
    static const std::vector<std::string> candidates = {
        "output.weight",
        "lm_head.weight",
        "model.output.weight",
        "model.lm_head.weight",
        "head.weight"
    };

    const auto& tensors = loader.tensors();
    auto is_valid = [&](const frontend::GGUFTensorInfo& info) {
        return dtypeSupportedForLinear(info.dtype) && tensorMatchesHidden(info, hidden_size);
    };

    for (const auto& name : candidates) {
        auto it = tensors.find(name);
        if (it == tensors.end()) continue;
        if (is_valid(it->second)) {
            return name;
        }
    }

    for (const auto& [name, info] : tensors) {
        if (is_valid(info)) return name;
    }
    return "";
}

std::string ExecutionPlanBuilder::selectHeadBias(const frontend::GGUFLoader& loader,
                                                 const std::string& weight_name,
                                                 size_t hidden_size) {
    const auto& tensors = loader.tensors();
    auto weight_it = tensors.find(weight_name);
    if (weight_it == tensors.end()) return "";
    const auto& weight_info = weight_it->second;
    if (weight_info.shape.size() != 2) return "";
    size_t vocab_dim = (static_cast<size_t>(weight_info.shape[0]) == hidden_size)
                           ? static_cast<size_t>(weight_info.shape[1])
                           : static_cast<size_t>(weight_info.shape[0]);
    auto biasMatches = [&](const frontend::GGUFTensorInfo& info) {
        if (!dtypeSupportedForLinear(info.dtype)) return false;
        if (info.shape.size() == 1) {
            return static_cast<size_t>(info.shape[0]) == vocab_dim;
        }
        if (info.shape.size() == 2) {
            return static_cast<size_t>(info.shape[0]) == vocab_dim && info.shape[1] == 1;
        }
        return false;
    };

    std::vector<std::string> candidates;
    if (!weight_name.empty()) {
        auto pos = weight_name.rfind(".weight");
        if (pos != std::string::npos) {
            std::string derived = weight_name;
            derived.replace(pos, std::string(".weight").size(), ".bias");
            candidates.push_back(derived);
        }
    }
    candidates.insert(candidates.end(),
                      {"output.bias",
                       "lm_head.bias",
                       "model.output.bias",
                       "model.lm_head.bias",
                       "head.bias"});

    for (const auto& name : candidates) {
        auto it = tensors.find(name);
        if (it == tensors.end()) continue;
        if (biasMatches(it->second)) return name;
    }
    return "";
}

} // namespace runtime
} // namespace mlc
