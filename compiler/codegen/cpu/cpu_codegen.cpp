#include "cpu_codegen.hpp"
#include "ir/ir.hpp"
#include <string>

namespace mlc {
namespace codegen {
namespace cpu {

CPUCodeGen::CPUCodeGen() = default;
CPUCodeGen::~CPUCodeGen() = default;

void CPUCodeGen::generateCode() {
    // Placeholder implementation
}

void CPUCodeGen::optimize() {
    // Placeholder implementation
}

void CPUCodeGen::generateMatMul(const ir::Node* node, const ir::Tensor* left, const ir::Tensor* right_weight) {
    // Check if right weight needs transpose
    bool transB = false;
    if (right_weight && right_weight->metadata.find("needs_transpose") != right_weight->metadata.end()) {
        if (right_weight->metadata.at("needs_transpose") == "true") {
            transB = true;
        }
    }
    
    // Generate GEMM with transB flag
    // For now, this is a placeholder - actual codegen would emit BLAS call:
    // cblas_sgemm(..., transB ? CblasTrans : CblasNoTrans, ...)
    (void)node;
    (void)left;
    (void)right_weight;
    (void)transB;
}

} // namespace cpu
} // namespace codegen
} // namespace mlc


