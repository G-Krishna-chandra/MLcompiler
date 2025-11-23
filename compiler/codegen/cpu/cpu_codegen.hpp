#pragma once

namespace mlc {
namespace ir {
    struct Node;
    struct Tensor;
}

namespace codegen {
namespace cpu {

class CPUCodeGen {
public:
    CPUCodeGen();
    ~CPUCodeGen();
    
    // Placeholder methods
    void generateCode();
    void optimize();
    
    // Generate matmul/GEMM with optional transpose flags
    // When right_weight has needs_transpose=true, use transB=true in GEMM
    void generateMatMul(const ir::Node* node, const ir::Tensor* left, const ir::Tensor* right_weight);
};

} // namespace cpu
} // namespace codegen
} // namespace mlc


