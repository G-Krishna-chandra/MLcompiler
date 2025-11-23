#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlc {
namespace ir {

enum class DataType {
    F32,
    F16,
    BF16,
    I8,
    I4
};

struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<int64_t> original_shape;  // Original shape from GGUF file
    DataType dtype;
    uint64_t byteOffset;
    bool layout_transposed = false;  // True if tensor was logically transposed
    std::unordered_map<std::string, std::string> metadata;  // For storing additional info like gguf_dtype
    
    Tensor(const std::string& n, const std::vector<int64_t>& s, DataType dt, uint64_t offset = 0)
        : name(n), shape(s), original_shape(s), dtype(dt), byteOffset(offset) {}
};

enum class OpKind {
    MatMul,
    Add,
    Mul,
    Linear,
    Norm,
    Softmax,
    Reshape,
    Transpose,
    Embedding,
    Attention,
    FeedForward,
    LayerNorm
};

struct Node {
    OpKind kind;
    std::string name;
    std::vector<Node*> inputs;
    std::vector<Tensor*> outputs;
    std::vector<Tensor*> tensor_inputs;
    std::vector<Tensor*> activation_inputs;
    std::unordered_map<std::string, float> attributes;
    std::unordered_map<std::string, std::string> metadata;
    
    Node(OpKind k, const std::string& n)
        : kind(k), name(n) {}
};

class Graph {
public:
    Graph();
    ~Graph();
    
    // Disable copy constructor and assignment
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    
    Node* addNode(OpKind kind, const std::string& name);
    Tensor* addTensor(const std::string& name, const std::vector<int64_t>& shape, DataType dtype);
    
    const std::vector<Node*>& nodes() const { return nodes_; }
    const std::vector<Tensor*>& tensors() const { return tensors_; }
    Tensor* findTensor(const std::string& name) const;
    
    // Debugging helper
    std::string dumpGraph() const;

private:
    std::vector<Node*> nodes_;
    std::vector<Tensor*> tensors_;
    std::unordered_map<std::string, Tensor*> tensor_lookup_;
};

// Helper function to convert DataType to string
std::string dataTypeToString(DataType dtype);

// Helper function to convert OpKind to string
std::string opKindToString(OpKind kind);

// Check if tensor name is an LM head (output layer)
bool IsLMHead(const std::string& name);

// Check if tensor name is likely a matmul weight
bool IsLikelyMatmulWeight(const std::string& name);

} // namespace ir
} // namespace mlc
