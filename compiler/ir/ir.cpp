#include "ir.hpp"
#include <sstream>
#include <algorithm>

namespace mlc {
namespace ir {

Graph::Graph() = default;

Graph::~Graph() {
    // Clean up nodes
    for (Node* node : nodes_) {
        delete node;
    }
    
    // Clean up tensors
    for (Tensor* tensor : tensors_) {
        delete tensor;
    }
}

Node* Graph::addNode(OpKind kind, const std::string& name) {
    Node* node = new Node(kind, name);
    nodes_.push_back(node);
    return node;
}

Tensor* Graph::addTensor(const std::string& name, const std::vector<int64_t>& shape, DataType dtype) {
    Tensor* tensor = new Tensor(name, shape, dtype);
    tensors_.push_back(tensor);
    tensor_lookup_[name] = tensor;
    return tensor;
}

Tensor* Graph::findTensor(const std::string& name) const {
    auto it = tensor_lookup_.find(name);
    if (it == tensor_lookup_.end()) return nullptr;
    return it->second;
}

std::string Graph::dumpGraph() const {
    std::ostringstream oss;
    
    oss << "=== Graph Dump ===\n";
    oss << "Tensors (" << tensors_.size() << "):\n";
    for (size_t i = 0; i < tensors_.size(); ++i) {
        const Tensor* t = tensors_[i];
        oss << "  [" << i << "] " << t->name 
            << " shape=[";
        for (size_t j = 0; j < t->shape.size(); ++j) {
            oss << t->shape[j];
            if (j < t->shape.size() - 1) oss << ", ";
        }
        oss << "] dtype=" << dataTypeToString(t->dtype)
            << " offset=" << t->byteOffset << "\n";
    }
    
    oss << "\nNodes (" << nodes_.size() << "):\n";
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const Node* n = nodes_[i];
        oss << "  [" << i << "] " << n->name 
            << " kind=" << opKindToString(n->kind)
            << " inputs=[";
        for (size_t j = 0; j < n->inputs.size(); ++j) {
            if (n->inputs[j] != nullptr) {
                oss << n->inputs[j]->name;
            } else {
                oss << "null";
            }
            if (j < n->inputs.size() - 1) oss << ", ";
        }
        oss << "] outputs=[";
        for (size_t j = 0; j < n->outputs.size(); ++j) {
            if (n->outputs[j] != nullptr) {
                oss << n->outputs[j]->name;
            } else {
                oss << "null";
            }
            if (j < n->outputs.size() - 1) oss << ", ";
        }
        oss << "]";
        if (!n->tensor_inputs.empty()) {
            oss << " params=[";
            for (size_t j = 0; j < n->tensor_inputs.size(); ++j) {
                oss << (n->tensor_inputs[j] ? n->tensor_inputs[j]->name : "null");
                if (j < n->tensor_inputs.size() - 1) oss << ", ";
            }
            oss << "]";
        }
        if (!n->metadata.empty()) {
            oss << " meta={";
            bool first = true;
            for (const auto& [key, value] : n->metadata) {
                if (!first) oss << ", ";
                oss << key << "=" << value;
                first = false;
            }
            oss << "}";
        }
        
        if (!n->attributes.empty()) {
            oss << " attrs={";
            bool first = true;
            for (const auto& [key, value] : n->attributes) {
                if (!first) oss << ", ";
                oss << key << "=" << value;
                first = false;
            }
            oss << "}";
        }
        oss << "\n";
    }
    
    return oss.str();
}

std::string dataTypeToString(DataType dtype) {
    switch (dtype) {
        case DataType::F32: return "F32";
        case DataType::F16: return "F16";
        case DataType::BF16: return "BF16";
        case DataType::I8: return "I8";
        case DataType::I4: return "I4";
        default: return "UNKNOWN";
    }
}

std::string opKindToString(OpKind kind) {
    switch (kind) {
        case OpKind::MatMul: return "MatMul";
        case OpKind::Add: return "Add";
        case OpKind::Mul: return "Mul";
        case OpKind::Linear: return "Linear";
        case OpKind::Norm: return "Norm";
        case OpKind::Softmax: return "Softmax";
        case OpKind::Reshape: return "Reshape";
        case OpKind::Transpose: return "Transpose";
        case OpKind::Embedding: return "Embedding";
        case OpKind::Attention: return "Attention";
        case OpKind::FeedForward: return "FeedForward";
        case OpKind::LayerNorm: return "LayerNorm";
        default: return "UNKNOWN";
    }
}

bool IsLMHead(const std::string& name) {
    // Exact matches
    if (name == "output.weight" || name == "lm_head.weight") {
        return true;
    }
    
    // Names ending with "output.weight"
    if (name.length() >= 13 && name.substr(name.length() - 13) == "output.weight") {
        return true;
    }
    
    // Names ending with "output.weight.T"
    if (name.length() >= 15 && name.substr(name.length() - 15) == "output.weight.T") {
        return true;
    }
    
    return false;
}

bool IsLikelyMatmulWeight(const std::string& name) {
    // Check for common matmul weight patterns
    std::string lower_name = name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    
    // Attention weights
    if (lower_name.find("wq") != std::string::npos ||
        lower_name.find("wk") != std::string::npos ||
        lower_name.find("wv") != std::string::npos ||
        lower_name.find("wo") != std::string::npos ||
        lower_name.find("q_proj") != std::string::npos ||
        lower_name.find("k_proj") != std::string::npos ||
        lower_name.find("v_proj") != std::string::npos ||
        lower_name.find("o_proj") != std::string::npos) {
        return true;
    }
    
    // MLP weights
    if (lower_name.find("mlp_up") != std::string::npos ||
        lower_name.find("mlp_down") != std::string::npos ||
        lower_name.find("gate_proj") != std::string::npos ||
        lower_name.find("up_proj") != std::string::npos ||
        lower_name.find("down_proj") != std::string::npos) {
        return true;
    }
    
    // Generic weight patterns
    if (lower_name.find(".weight") != std::string::npos && 
        (lower_name.find("linear") != std::string::npos ||
         lower_name.find("proj") != std::string::npos ||
         lower_name.find("attn") != std::string::npos)) {
        return true;
    }
    
    return false;
}

} // namespace ir
} // namespace mlc
