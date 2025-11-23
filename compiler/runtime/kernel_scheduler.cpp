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
        case ir::OpKind::Transpose: return ExecOpType::MatMul;
        default:
            return ExecOpType::Unknown;
    }
}

BackendKind selectBackend(ExecOpType op,
                          const ir::Node* node) {
    switch (op) {
        case ExecOpType::Embedding:
        case ExecOpType::Attention:
            return BackendKind::Metal;
        case ExecOpType::MatMul:
        case ExecOpType::Linear: {
            if (!node || node->outputs.empty() || !node->outputs[0]) {
                return BackendKind::Metal;
            }
            const auto& shape = node->outputs[0]->shape;
            int64_t rows = !shape.empty() ? shape[0] : 0;
            return rows >= 128 ? BackendKind::Metal : BackendKind::CPU;
        }
        case ExecOpType::FeedForward:
        case ExecOpType::Norm:
        case ExecOpType::Add:
            return BackendKind::Metal;
        default:
            return BackendKind::Auto;
    }
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

ExecutionGraph KernelScheduler::Schedule(const ir::Graph& graph,
                                         const frontend::GGUFLoader& loader) {
    ExecutionGraph exec;
    ModelConfig config;
    config.num_layers = readSize(loader,
                                 {"llama.block_count", "llama.layers", "n_layer"},
                                 graph.nodes().size());
    config.hidden_size = readSize(loader,
                                  {"llama.embedding_length", "llama.n_embd", "n_embd"},
                                  graph.tensors().empty() ? 0
                                                          : std::abs(graph.tensors().front()->shape.empty()
                                                                         ? 0
                                                                         : graph.tensors().front()->shape[0]));
    config.head_count = readSize(loader,
                                 {"llama.attention.head_count", "attention.head_count"},
                                 8);
    config.kv_head_count = readSize(loader,
                                    {"llama.attention.head_count_kv", "attention.head_count_kv"},
                                    config.head_count);
    config.context_length = readSize(loader,
                                     {"llama.context_length", "context_length"},
                                     128);
    config.vocab_size = readSize(loader,
                                 {"general.vocab_size", "tokenizer.ggml.vocab_size"},
                                 0);
    config.rotary_dim = readSize(loader,
                                 {"llama.rope.dimension_count", "rope.dimension_count"},
                                 0);
    config.rope_freq_base = readFloat(loader,
                                      {"llama.rope.freq_base", "rope.freq_base"},
                                      10000.0f);
    config.rope_freq_scale = readFloat(loader,
                                       {"llama.rope.freq_scale", "rope.freq_scale"},
                                       1.0f);
    config.architecture = readString(loader,
                                     {"general.architecture", "architecture"});
    config.family = detectFamily(config.architecture);
    config.grouped_query_attention = config.kv_head_count > 0 && config.head_count > config.kv_head_count;
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
        BackendKind backend = selectBackend(op, node);
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
        }

        KernelSelectionQuery query = buildQuery(exec_node, backend, config);
        const KernelDescriptor* descriptor = kernel_registry.select(query);
        recordKernelSelection(exec_node, descriptor);
    }

    return exec;
}

} // namespace runtime
} // namespace mlc
