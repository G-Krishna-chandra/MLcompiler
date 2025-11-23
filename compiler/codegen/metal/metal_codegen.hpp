#pragma once

namespace mlc {
namespace ir {
    struct Node;
    struct Tensor;
}

namespace codegen {
namespace metal {

class MetalCodeGen {
public:
    MetalCodeGen();
    ~MetalCodeGen();
    
    // Placeholder methods
    void generateCode();
    void optimize();
    
    // Generate matmul/GEMM with optional transpose flags
    // When right_weight has needs_transpose=true, use transB=true or logical transpose in kernel
    void generateMatMul(const ir::Node* node, const ir::Tensor* left, const ir::Tensor* right_weight);
};

} // namespace metal
} // namespace codegen
} // namespace mlc


