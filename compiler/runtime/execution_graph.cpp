#include "runtime/execution_graph.hpp"

#include <queue>
#include <sstream>
#include <stdexcept>

namespace mlc {
namespace runtime {

namespace {
std::string join(const std::vector<std::string>& items) {
    std::ostringstream oss;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << items[i];
    }
    return oss.str();
}
}

ExecutionTensor& ExecutionGraph::addTensor(const std::string& name,
                                           const std::vector<int64_t>& shape,
                                           ir::DataType dtype) {
    auto [it, inserted] = tensors_.emplace(name, ExecutionTensor{});
    if (inserted) {
        it->second.name = name;
    }
    it->second.shape = shape;
    it->second.dtype = dtype;
    return it->second;
}

ExecutionNode& ExecutionGraph::addNode(const std::string& name,
                                       ExecOpType op,
                                       const std::vector<std::string>& inputs,
                                       const std::vector<std::string>& outputs,
                                       BackendKind backend) {
    if (node_index_.count(name)) {
        throw std::runtime_error("Execution node '" + name + "' already exists");
    }
    for (const auto& input : inputs) {
        ensureTensorExists(input);
    }
    for (const auto& output : outputs) {
        if (!tensors_.count(output)) {
            addTensor(output, {}, ir::DataType::F32);
        }
        tensor_producer_[output] = name;
    }
    nodes_.push_back(ExecutionNode{});
    auto& node = nodes_.back();
    node.name = name;
    node.op = op;
    node.inputs = inputs;
    node.outputs = outputs;
    node.backend = backend;
    node.kernel_id = "";
    node.activation_dtype = ir::DataType::F32;
    node.weight_dtype = ir::DataType::F32;
    node.activation_quantized = false;
    node.weight_quantized = false;
    node.grouped_query = false;
    node_index_[name] = nodes_.size() - 1;
    return node;
}

void ExecutionGraph::setModelConfig(const ModelConfig& config) {
    model_config_ = config;
}

ExecutionTensor* ExecutionGraph::getTensor(const std::string& name) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    return &it->second;
}

const ExecutionTensor* ExecutionGraph::getTensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    return &it->second;
}

void ExecutionGraph::ensureTensorExists(const std::string& name) const {
    if (!name.empty() && !tensors_.count(name)) {
        throw std::runtime_error("Tensor '" + name + "' not registered in execution graph");
    }
}

std::vector<std::string> ExecutionGraph::topologicalOrder() const {
    std::unordered_map<std::string, int> indegree;
    std::unordered_map<std::string, std::vector<std::string>> adjacency;

    for (const auto& node : nodes_) {
        indegree[node.name] = 0;
    }
    for (const auto& node : nodes_) {
        for (const auto& input : node.inputs) {
            auto it = tensor_producer_.find(input);
            if (it == tensor_producer_.end()) continue;
            adjacency[it->second].push_back(node.name);
            indegree[node.name]++;
        }
    }

    std::queue<std::string> ready;
    for (const auto& [name, deg] : indegree) {
        if (deg == 0) ready.push(name);
    }

    std::vector<std::string> order;
    while (!ready.empty()) {
        auto current = ready.front(); ready.pop();
        order.push_back(current);
        for (const auto& dep : adjacency[current]) {
            if (--indegree[dep] == 0) {
                ready.push(dep);
            }
        }
    }

    if (order.size() != nodes_.size()) {
        throw std::runtime_error("ExecutionGraph contains cycles; cannot schedule");
    }
    return order;
}

std::string ExecutionGraph::dump() const {
    std::ostringstream oss;
    if (model_config_.num_layers > 0 || model_config_.hidden_size > 0) {
        oss << "ModelConfig: layers=" << model_config_.num_layers
            << " hidden=" << model_config_.hidden_size
            << " heads=" << model_config_.head_count
            << " kv_heads=" << model_config_.kv_head_count
            << " head_dim=" << model_config_.head_dim
            << " context=" << model_config_.context_length
            << " vocab=" << model_config_.vocab_size;
        if (!model_config_.architecture.empty()) {
            oss << " arch=" << model_config_.architecture;
        }
        if (model_config_.family != ArchitectureFamily::Unknown) {
            oss << " family=" << toString(model_config_.family);
        }
        oss << "\n";
    }
    for (const auto& node : nodes_) {
        oss << node.name << " [" << toString(node.op) << "] "
            << "inputs(" << join(node.inputs) << ") -> outputs(" << join(node.outputs) << ") "
            << "backend=" << toString(node.backend);
        if (!node.kernel_id.empty()) {
            oss << " kernel=" << node.kernel_id;
        }
        oss << "\n";
        if (!node.attributes.empty()) {
            oss << "  attrs: ";
            bool first = true;
            for (const auto& [k, v] : node.attributes) {
                if (!first) oss << ", ";
                oss << k << "=" << v;
                first = false;
            }
            oss << "\n";
        }
        if (!node.annotations.empty()) {
            oss << "  tags: ";
            bool first = true;
            for (const auto& [k, v] : node.annotations) {
                if (!first) oss << ", ";
                oss << k << "=" << v;
                first = false;
            }
            oss << "\n";
        }
    }
    return oss.str();
}

std::string toString(ExecOpType op) {
    switch (op) {
        case ExecOpType::Embedding: return "Embedding";
        case ExecOpType::Attention: return "Attention";
        case ExecOpType::FeedForward: return "FeedForward";
        case ExecOpType::MatMul: return "MatMul";
        case ExecOpType::Add: return "Add";
        case ExecOpType::Norm: return "Norm";
        case ExecOpType::Softmax: return "Softmax";
        case ExecOpType::Linear: return "Linear";
        case ExecOpType::Output: return "Output";
        default: return "Unknown";
    }
}

std::string toString(BackendKind backend) {
    switch (backend) {
        case BackendKind::CPU: return "CPU";
        case BackendKind::Metal: return "Metal";
        default: return "Auto";
    }
}

std::string toString(ArchitectureFamily family) {
    switch (family) {
        case ArchitectureFamily::Llama: return "llama";
        case ArchitectureFamily::Gemma: return "gemma";
        case ArchitectureFamily::Mistral: return "mistral";
        default: return "unknown";
    }
}

} // namespace runtime
} // namespace mlc
