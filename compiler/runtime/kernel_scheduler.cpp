#include "runtime/kernel_scheduler.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

#include "frontends/gguf_utils.hpp"
#include "runtime/kernel_registry.hpp"

namespace mlc {
namespace runtime {

namespace {

size_t readSize(const frontend::GGUFLoader& loader,
                const std::vector<std::string>& keys,
                size_t fallback) {
    const auto& kv = loader.kvMetadata();
    for (const auto& key : keys) {
        auto it = kv.find(key);
        if (it == kv.end()) continue;
        const auto& value = it->second;
        switch (value.type) {
            case frontend::GGUFValueType::UINT32:
                return static_cast<size_t>(std::get<uint32_t>(value.data));
            case frontend::GGUFValueType::UINT64:
                return static_cast<size_t>(std::get<uint64_t>(value.data));
            case frontend::GGUFValueType::INT32: {
                int32_t v = std::get<int32_t>(value.data);
                if (v > 0) return static_cast<size_t>(v);
                break;
            }
            case frontend::GGUFValueType::INT64: {
                int64_t v = std::get<int64_t>(value.data);
                if (v > 0) return static_cast<size_t>(v);
                break;
            }
            default:
                break;
        }
    }
    return fallback;
}

std::string readString(const frontend::GGUFLoader& loader,
                       const std::vector<std::string>& keys) {
    const auto& kv = loader.kvMetadata();
    for (const auto& key : keys) {
        auto it = kv.find(key);
        if (it == kv.end()) continue;
        if (it->second.type == frontend::GGUFValueType::STRING) {
            return std::get<std::string>(it->second.data);
        }
    }
    return {};
}

float readFloat(const frontend::GGUFLoader& loader,
                const std::vector<std::string>& keys,
                float fallback) {
    const auto& kv = loader.kvMetadata();
    for (const auto& key : keys) {
        auto it = kv.find(key);
        if (it == kv.end()) continue;
        switch (it->second.type) {
            case frontend::GGUFValueType::FLOAT32:
                return std::get<float>(it->second.data);
            case frontend::GGUFValueType::FLOAT64:
                return static_cast<float>(std::get<double>(it->second.data));
            case frontend::GGUFValueType::UINT32:
                return static_cast<float>(std::get<uint32_t>(it->second.data));
            case frontend::GGUFValueType::UINT64:
                return static_cast<float>(std::get<uint64_t>(it->second.data));
            case frontend::GGUFValueType::INT32:
                return static_cast<float>(std::get<int32_t>(it->second.data));
            case frontend::GGUFValueType::INT64:
                return static_cast<float>(std::get<int64_t>(it->second.data));
            default:
                break;
        }
    }
    return fallback;
}

ExecOpType mapOp(ir::OpKind kind) {
    switch (kind) {
        case ir::OpKind::Embedding: return ExecOpType::Embedding;
        case ir::OpKind::Attention: return ExecOpType::Attention;
        case ir::OpKind::FeedForward: return ExecOpType::FeedForward;
        case ir::OpKind::MatMul: return ExecOpType::MatMul;
        case ir::OpKind::Linear: return ExecOpType::Linear;
        case ir::OpKind::LayerNorm:
        case ir::OpKind::Norm: return ExecOpType::Norm;
        case ir::OpKind::Softmax: return ExecOpType::Softmax;
        case ir::OpKind::Add: return ExecOpType::Add;
        case ir::OpKind::Slice: return ExecOpType::Slice;
        case ir::OpKind::Transpose: return ExecOpType::MatMul;
        default:
            return ExecOpType::Unknown;
    }
}

BackendKind selectBackend(ExecOpType op,
                          const ir::Node* node,
                          const ModelConfig& config) {
    switch (op) {
        case ExecOpType::Embedding:
            // Small embeddings are fine on CPU; otherwise prefer Metal.
            return (config.head_count * config.head_dim < 256) ? BackendKind::CPU : BackendKind::Metal;
        case ExecOpType::Attention: {
            bool is_familyspecial = (config.family == ArchitectureFamily::Gemma ||
                                     config.family == ArchitectureFamily::Mistral);
            // Gemma/Mistral: allow Metal when heads are reasonably wide or grouped-query is requested.
            if (is_familyspecial) {
                if (config.head_dim >= 64 || config.grouped_query_attention) {
                    return BackendKind::Metal;
                }
                return BackendKind::CPU;
            }
            // Heuristic: push to Metal when head_dim is at least 64 or tokens are multi.
            int64_t tokens = (node && !node->outputs.empty() && node->outputs[0])
                                 ? (node->outputs[0]->shape.empty() ? 1 : node->outputs[0]->shape[0])
                                 : 1;
            if (config.head_dim >= 64 || tokens > 1) return BackendKind::Metal;
            return BackendKind::CPU;
        }
        case ExecOpType::MatMul:
        case ExecOpType::Linear: {
            if (!node || node->outputs.empty() || !node->outputs[0]) {
                return BackendKind::Metal;
            }
            const auto& shape = node->outputs[0]->shape;
            int64_t rows = !shape.empty() ? shape[0] : 0;
            // Prefer Metal for mid/large rows; CPU for very small.
            return rows >= 64 ? BackendKind::Metal : BackendKind::CPU;
        }
        case ExecOpType::FeedForward: {
            // Prefer Metal when hidden is moderate/large or when using GeGLU (Gemma-style).
            if (config.hidden_size >= 512) return BackendKind::Metal;
            std::string act = config.activation;
            std::transform(act.begin(), act.end(), act.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (act == "geglu" || act == "gelu") return BackendKind::Metal;
            return BackendKind::CPU;
        }
        case ExecOpType::Norm:
        case ExecOpType::Add:
            return BackendKind::Metal;
        case ExecOpType::Slice:
            return BackendKind::CPU;
        default:
            return BackendKind::Auto;
    }
}

bool readBool(const frontend::GGUFLoader& loader,
              const std::vector<std::string>& keys,
              bool fallback) {
    const auto& kv = loader.kvMetadata();
    for (const auto& key : keys) {
        auto it = kv.find(key);
        if (it == kv.end()) continue;
        if (it->second.type == frontend::GGUFValueType::BOOL) {
            return std::get<bool>(it->second.data);
        }
    }
    return fallback;
}

std::vector<float> generateAlibiSlopes(size_t heads) {
    // From Fairseq/LLAMA-style ALiBi slopes generation
    std::vector<float> slopes;
    if (heads == 0) return slopes;
    auto get_pow = [](int64_t a, int64_t b) {
        double r = std::pow(static_cast<double>(a), 1.0 / static_cast<double>(b));
        return r;
    };
    size_t closest_power_of_2 = 1;
    while (closest_power_of_2 * 2 <= heads) closest_power_of_2 *= 2;
    double m = get_pow(2, static_cast<int64_t>(closest_power_of_2));
    double start = m;
    slopes.reserve(heads);
    for (size_t i = 0; i < closest_power_of_2; ++i) {
        double slope = std::pow(m, static_cast<double>(i) / static_cast<double>(closest_power_of_2));
        slopes.push_back(static_cast<float>(slope));
    }
    if (closest_power_of_2 < heads) {
        double m2 = get_pow(2, static_cast<int64_t>(2 * closest_power_of_2));
        size_t extra = heads - closest_power_of_2;
        for (size_t i = 1; i <= extra; ++i) {
            double slope = std::pow(m2, static_cast<double>(2 * i - 1) / static_cast<double>(2 * extra));
            slopes.push_back(static_cast<float>(slope));
        }
    }
    return slopes;
}

ArchitectureFamily detectFamily(const std::string& arch) {
    if (arch.empty()) return ArchitectureFamily::Unknown;
    std::string lower = arch;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lower.find("llama") != std::string::npos) return ArchitectureFamily::Llama;
    if (lower.find("gemma") != std::string::npos) return ArchitectureFamily::Gemma;
    if (lower.find("mistral") != std::string::npos) return ArchitectureFamily::Mistral;
    return ArchitectureFamily::Unknown;
}

struct FamilyKeySet {
    std::vector<std::string> head_count;
    std::vector<std::string> kv_head_count;
    std::vector<std::string> context_length;
    std::vector<std::string> block_count;
    std::vector<std::string> hidden_size;
    std::vector<std::string> rope_dim;
    std::vector<std::string> rope_freq_base;
    std::vector<std::string> rope_freq_scale;
    std::vector<std::string> vocab_size;
};

FamilyKeySet keySetForFamily(ArchitectureFamily family) {
    FamilyKeySet ks;
    switch (family) {
    case ArchitectureFamily::Gemma:
        ks.head_count = {"gemma.attention.head_count", "attention.head_count", "num_attention_heads"};
        ks.kv_head_count = {"gemma.attention.head_count_kv", "attention.head_count_kv"};
        ks.context_length = {"gemma.context_length", "context_length"};
        ks.block_count = {"gemma.block_count", "block_count", "num_hidden_layers"};
        ks.hidden_size = {"gemma.embedding_length", "embedding_length", "n_embd"};
        ks.rope_dim = {"gemma.rope.dimension_count", "rope.dimension_count"};
        ks.rope_freq_base = {"gemma.rope.freq_base", "rope.freq_base"};
        ks.rope_freq_scale = {"gemma.rope.freq_scale", "rope.freq_scale"};
        ks.vocab_size = {"gemma.vocab_size", "tokenizer.ggml.vocab_size", "vocab_size"};
        break;
    case ArchitectureFamily::Mistral:
        ks.head_count = {"mistral.attention.head_count", "attention.head_count", "num_attention_heads"};
        ks.kv_head_count = {"mistral.attention.head_count_kv", "attention.head_count_kv"};
        ks.context_length = {"mistral.context_length", "context_length"};
        ks.block_count = {"mistral.block_count", "block_count", "num_hidden_layers"};
        ks.hidden_size = {"mistral.embedding_length", "embedding_length", "n_embd"};
        ks.rope_dim = {"mistral.rope.dimension_count", "rope.dimension_count"};
        ks.rope_freq_base = {"mistral.rope.freq_base", "rope.freq_base"};
        ks.rope_freq_scale = {"mistral.rope.freq_scale", "rope.freq_scale"};
        ks.vocab_size = {"mistral.vocab_size", "tokenizer.ggml.vocab_size", "vocab_size"};
        break;
    case ArchitectureFamily::Llama:
    default:
        ks.head_count = {"llama.attention.head_count", "attention.head_count", "num_attention_heads"};
        ks.kv_head_count = {"llama.attention.head_count_kv", "attention.head_count_kv"};
        ks.context_length = {"llama.context_length", "context_length"};
        ks.block_count = {"llama.block_count", "llama.layers", "block_count", "n_layer", "num_hidden_layers"};
        ks.hidden_size = {"llama.embedding_length", "llama.n_embd", "embedding_length", "n_embd"};
        ks.rope_dim = {"llama.rope.dimension_count", "rope.dimension_count"};
        ks.rope_freq_base = {"llama.rope.freq_base", "rope.freq_base"};
        ks.rope_freq_scale = {"llama.rope.freq_scale", "rope.freq_scale"};
        ks.vocab_size = {"general.vocab_size", "tokenizer.ggml.vocab_size", "vocab_size"};
        break;
    }
    return ks;
}

void annotateTensor(const ir::Tensor* source,
                    ExecutionTensor& target,
                    uint32_t quant_version) {
    if (!source) return;
    for (const auto& [key, value] : source->metadata) {
        target.metadata[key] = value;
    }
    target.quant_version = quant_version;
    if (source->layout_transposed) {
        target.layout.transposed = true;
    }
    auto it = source->metadata.find("needs_transpose");
    if (it != source->metadata.end()) {
        target.layout.transposed = (it->second == "true");
    }
    auto gguf = source->metadata.find("gguf_dtype");
    if (gguf != source->metadata.end()) {
        uint32_t dtype = target.ggml_dtype;
        if (frontend::tryParseGGUFDtypeString(gguf->second, dtype)) {
            target.ggml_dtype = dtype;
            target.has_ggml_dtype = true;
            target.quantized = frontend::isQuantizedGGMLType(dtype);
        }
    }
}

size_t readSizeAttr(const std::unordered_map<std::string, float>& attrs,
                    const std::string& key,
                    size_t fallback) {
    auto it = attrs.find(key);
    if (it == attrs.end()) return fallback;
    if (it->second <= 0.0f) return fallback;
    return static_cast<size_t>(it->second);
}

void recordKernelSelection(ExecutionNode& node,
                           const KernelDescriptor* descriptor) {
    if (!descriptor) {
        node.kernel_id.clear();
        node.annotations["kernel"] = "unassigned";
        return;
    }
    node.kernel_id = descriptor->id;
    node.backend = descriptor->backend;
    node.annotations["kernel"] = descriptor->id;
    std::ostringstream oss;
    oss << descriptor->tile_m << "x" << descriptor->tile_n << "x" << descriptor->tile_k;
    node.annotations["kernel_tile"] = oss.str();
}

KernelSelectionQuery buildQuery(const ExecutionNode& node,
                                BackendKind preferred,
                                const ModelConfig& config) {
    KernelSelectionQuery query;
    query.op = node.op;
    query.preferred_backend = preferred;
    query.activation_dtype = node.activation_dtype;
    query.weight_dtype = node.weight_dtype;
    query.activation_quantized = node.activation_quantized;
    query.weight_quantized = node.weight_quantized;
    query.batch = readSizeAttr(node.attributes, "batch", 1);
    query.context = config.context_length;
    query.head_count = readSizeAttr(node.attributes, "heads", config.head_count);
    query.kv_head_count = readSizeAttr(node.attributes, "kv_heads", config.kv_head_count);
    query.head_dim = readSizeAttr(node.attributes, "head_dim", config.head_dim);
    query.grouped_query = node.grouped_query;
    return query;
}

void annotateNodeQuantization(ExecutionNode& node,
                              const ir::Tensor* activation,
                              const ir::Tensor* weight,
                              ExecutionGraph& exec) {
    if (activation) {
        node.activation_dtype = activation->dtype;
        if (auto* tensor = exec.getTensor(activation->name)) {
            node.activation_quantized = tensor->quantized;
        }
    }
    if (weight) {
        node.weight_dtype = weight->dtype;
        if (auto* tensor = exec.getTensor(weight->name)) {
            node.weight_quantized = tensor->quantized;
        }
    }
}

} // namespace

ModelConfig KernelScheduler::BuildModelConfig(const frontend::GGUFLoader& loader) {
    ModelConfig config;
    config.architecture = readString(loader,
                                     {"general.architecture", "architecture"});
    config.family = detectFamily(config.architecture);
    FamilyKeySet ks = keySetForFamily(config.family);
    config.num_layers = readSize(loader, ks.block_count, 0);
    config.hidden_size = readSize(loader, ks.hidden_size, 0);
    config.head_count = readSize(loader, ks.head_count, 8);
    config.kv_head_count = readSize(loader, ks.kv_head_count, config.head_count);
    config.context_length = readSize(loader, ks.context_length, 128);
    config.vocab_size = readSize(loader, ks.vocab_size, 0);
    config.rotary_dim = readSize(loader, ks.rope_dim, 0);
    config.rope_freq_base = readFloat(loader,
                                      ks.rope_freq_base,
                                      10000.0f);
    config.rope_freq_scale = readFloat(loader,
                                       ks.rope_freq_scale,
                                       1.0f);
    config.sliding_window = readSize(loader,
                                     {"attention.sliding_window", "mistral.sliding_window"},
                                     0);
    config.activation = readString(loader,
                                   {"activation_type", "model.activation_type"});
    if (config.activation.empty() && config.family == ArchitectureFamily::Gemma) {
        config.activation = "geglu";
    } else if (config.activation.empty()) {
        config.activation = "silu";
    }
    config.use_alibi = readBool(loader,
                                {"attention.alibi", "alibi"},
                                false);
    if (config.use_alibi) {
        config.alibi_slopes = generateAlibiSlopes(config.head_count > 0 ? config.head_count : 1);
    }
    config.grouped_query_attention = config.kv_head_count > 0 && config.head_count > config.kv_head_count;
    if (config.head_count > 0 && config.hidden_size > 0) {
        config.head_dim = config.hidden_size / config.head_count;
    }
    return config;
}

ExecutionGraph KernelScheduler::Schedule(const ir::Graph& graph,
                                         const frontend::GGUFLoader& loader) {
    ExecutionGraph exec;
    ModelConfig config = BuildModelConfig(loader);
    if (config.num_layers == 0) {
        config.num_layers = graph.nodes().size();
    }
    if (config.hidden_size == 0 && !graph.tensors().empty() && graph.tensors().front()) {
        const auto* t = graph.tensors().front();
        if (!t->shape.empty()) config.hidden_size = std::abs(t->shape[0]);
        if (config.head_count > 0) config.head_dim = config.hidden_size / config.head_count;
    }
    exec.setModelConfig(config);

    uint32_t quant_version = loader.quantizationVersion();

    for (const ir::Tensor* tensor : graph.tensors()) {
        if (!tensor) continue;
        if (tensor->metadata.count("role") && tensor->metadata.at("role") == "input_tokens") {
            auto& added = exec.addTensor(tensor->name, tensor->shape, tensor->dtype);
            annotateTensor(tensor, added, quant_version);
        }
    }

    const auto& kernel_registry = KernelDescriptorRegistry::Instance();

    for (const ir::Node* node : graph.nodes()) {
        if (!node) continue;
        ExecOpType op = mapOp(node->kind);
        if (op == ExecOpType::Unknown) {
            // Skip utility nodes (e.g., reshape/token source) that do not map to runtime ops.
            // Their tensors have already been registered above so downstream nodes can consume them.
            continue;
        }
        if (node->outputs.empty()) continue;
        std::vector<std::string> inputs;
        for (const ir::Tensor* input_tensor : node->activation_inputs) {
            if (!input_tensor) continue;
            auto& tensor = exec.addTensor(input_tensor->name, input_tensor->shape, input_tensor->dtype);
            annotateTensor(input_tensor, tensor, quant_version);
            inputs.push_back(input_tensor->name);
        }
        std::vector<std::string> outputs;
        for (const ir::Tensor* out : node->outputs) {
            if (!out) continue;
            auto& tensor = exec.addTensor(out->name, out->shape, out->dtype);
            annotateTensor(out, tensor, quant_version);
            outputs.push_back(out->name);
        }
        if (outputs.empty()) continue;
        BackendKind backend = selectBackend(op, node, config);
        auto& exec_node = exec.addNode(node->name, op, inputs, outputs, backend);
        exec_node.attributes = node->attributes;
        for (const auto& [key, value] : node->metadata) {
            exec_node.annotations[key] = value;
        }
        size_t param_index = 0;
        for (const ir::Tensor* param : node->tensor_inputs) {
            if (!param) continue;
            auto& tensor = exec.addTensor(param->name, param->shape, param->dtype);
            annotateTensor(param, tensor, quant_version);
            exec_node.annotations["param" + std::to_string(param_index++)] = param->name;
        }

        const ir::Tensor* first_activation = !node->activation_inputs.empty() ? node->activation_inputs.front() : nullptr;
        const ir::Tensor* first_weight = !node->tensor_inputs.empty() ? node->tensor_inputs.front() : nullptr;
        annotateNodeQuantization(exec_node, first_activation, first_weight, exec);

        if (op == ExecOpType::Attention) {
            exec_node.grouped_query = config.grouped_query_attention;
            exec_node.annotations["grouped_query"] = exec_node.grouped_query ? "true" : "false";

            // Attach KV cache tensors (state) and core attention attributes.
            std::string layer_tag;
            auto it_layer = node->metadata.find("layer");
            if (it_layer != node->metadata.end()) layer_tag = it_layer->second;
            size_t layer_index = 0;
            if (!layer_tag.empty()) {
                try {
                    layer_index = static_cast<size_t>(std::stoul(layer_tag));
                } catch (...) {
                    layer_index = 0;
                }
            }

            size_t heads = config.head_count > 0 ? config.head_count : 1;
            auto it_heads = node->metadata.find("heads");
            if (it_heads != node->metadata.end()) {
                try {
                    heads = static_cast<size_t>(std::stoul(it_heads->second));
                } catch (...) { /* keep default */ }
            }
            size_t kv_heads = config.kv_head_count > 0 ? config.kv_head_count : heads;
            size_t head_dim = config.head_dim > 0 ? config.head_dim
                                                  : (config.hidden_size > 0 && heads > 0
                                                         ? config.hidden_size / heads
                                                         : 1);
            exec_node.attributes["heads"] = static_cast<float>(heads);
            exec_node.attributes["kv_heads"] = static_cast<float>(kv_heads);
            exec_node.attributes["head_dim"] = static_cast<float>(head_dim);

            auto add_cache = [&](const std::string& base) -> std::string {
                std::ostringstream oss;
                oss << base << "." << layer_index;
                std::string name = oss.str();
                auto& t = exec.addTensor(name,
                                         {static_cast<int64_t>(kv_heads),
                                          static_cast<int64_t>(config.context_length),
                                          static_cast<int64_t>(head_dim)},
                                         ir::DataType::F32);
                t.is_state = true;
                t.metadata["role"] = "kv_cache";
                t.metadata["layer"] = std::to_string(layer_index);
                return name;
            };
            std::string cache_k = add_cache("kv_cache_k");
            std::string cache_v = add_cache("kv_cache_v");
            exec_node.annotations["kv_cache_k"] = cache_k;
            exec_node.annotations["kv_cache_v"] = cache_v;
        }

        KernelSelectionQuery query = buildQuery(exec_node, backend, config);
        const KernelDescriptor* descriptor = kernel_registry.select(query);
        recordKernelSelection(exec_node, descriptor);
    }

    return exec;
}

} // namespace runtime
} // namespace mlc
