#pragma once

#include "ir/ir.hpp"
#include "frontends/ggml_types.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace mlc {
namespace runtime {

enum class ExecOpType {
    Unknown,
    Embedding,
    Attention,
    FeedForward,
    MatMul,
    Add,
    Norm,
    Softmax,
    Linear,
    Output
};

enum class BackendKind {
    Auto,
    CPU,
    Metal
};

enum class ArchitectureFamily {
    Unknown,
    Llama,
    Gemma,
    Mistral
};

struct TensorLayoutInfo {
    bool column_major = false;
    bool transposed = false;
    bool normalized = false;
};

struct ExecutionTensor {
    std::string name;
    std::vector<int64_t> shape;
    ir::DataType dtype = ir::DataType::F32;
    bool is_state = false;
    TensorLayoutInfo layout;
    uint32_t ggml_dtype = frontend::GGML_TYPE_F32;
    bool has_ggml_dtype = false;
    bool quantized = false;
    uint32_t quant_version = 0;
    std::unordered_map<std::string, std::string> metadata;
};

struct ExecutionNode {
    std::string name;
    ExecOpType op = ExecOpType::Unknown;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    BackendKind backend = BackendKind::Auto;
    std::string kernel_id;
    ir::DataType activation_dtype = ir::DataType::F32;
    ir::DataType weight_dtype = ir::DataType::F32;
    bool activation_quantized = false;
    bool weight_quantized = false;
    bool grouped_query = false;
    std::unordered_map<std::string, float> attributes;
    std::unordered_map<std::string, std::string> annotations;
};

struct ModelConfig {
    size_t vocab_size = 0;
    size_t num_layers = 0;
    size_t hidden_size = 0;
    size_t head_count = 0;
    size_t kv_head_count = 0;
    size_t head_dim = 0;
    size_t context_length = 0;
    std::string architecture;
    ArchitectureFamily family = ArchitectureFamily::Unknown;
    size_t rotary_dim = 0;
    float rope_freq_base = 10000.0f;
    float rope_freq_scale = 1.0f;
    bool grouped_query_attention = false;
};

class ExecutionGraph {
public:
    ExecutionGraph() = default;

    ExecutionTensor& addTensor(const std::string& name,
                               const std::vector<int64_t>& shape,
                               ir::DataType dtype);

    ExecutionNode& addNode(const std::string& name,
                           ExecOpType op,
                           const std::vector<std::string>& inputs,
                           const std::vector<std::string>& outputs,
                           BackendKind backend = BackendKind::Auto);

    const std::unordered_map<std::string, ExecutionTensor>& tensors() const { return tensors_; }
    ExecutionTensor* getTensor(const std::string& name);
    const ExecutionTensor* getTensor(const std::string& name) const;
    const std::vector<ExecutionNode>& nodes() const { return nodes_; }

    void setModelConfig(const ModelConfig& config);
    const ModelConfig& modelConfig() const { return model_config_; }

    std::vector<std::string> topologicalOrder() const;
    std::string dump() const;

private:
    void ensureTensorExists(const std::string& name) const;

    std::unordered_map<std::string, ExecutionTensor> tensors_;
    std::vector<ExecutionNode> nodes_;
    std::unordered_map<std::string, size_t> node_index_;
    std::unordered_map<std::string, std::string> tensor_producer_;
    ModelConfig model_config_;
};

std::string toString(ExecOpType op);
std::string toString(BackendKind backend);
std::string toString(ArchitectureFamily family);

} // namespace runtime
} // namespace mlc
