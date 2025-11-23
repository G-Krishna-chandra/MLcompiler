#include "gguf_to_ir.hpp"
#include "frontends/ggml_types.hpp"
#include "gguf_utils.hpp"
#include "ir/ir.hpp"
#include <algorithm>
#include <regex>

namespace mlc {
namespace frontend {

ir::DataType mapGGUFDtypeToIR(uint32_t gguf_dtype) {
    switch (gguf_dtype) {
        case GGML_TYPE_F32:
            return ir::DataType::F32;
        case GGML_TYPE_F16:
            return ir::DataType::F16;
        case GGML_TYPE_BF16:
            return ir::DataType::BF16;
        case GGML_TYPE_I8:
            return ir::DataType::I8;
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q8_K:
            return ir::DataType::I8;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_K:  // Includes Q4_K_M
        case GGML_TYPE_Q2_K:
            return ir::DataType::I4;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q6_K:
            // K-quant types map to I4
            return ir::DataType::I4;
        default:
            // Default to F32 for unknown types
            return ir::DataType::F32;
    }
}

std::unique_ptr<ir::Graph> GGUFToIR(const GGUFLoader& loader) {
    auto graph = std::make_unique<ir::Graph>();
    
    // Get all tensors from GGUF loader
    const auto& gguf_tensors = loader.tensors();
    
    // Convert each GGUF tensor to IR tensor
    for (const auto& [name, gguf_tensor] : gguf_tensors) {
        // Convert shape from uint64_t to int64_t - IMPORT EXACTLY AS STORED
        std::vector<int64_t> file_shape;
        file_shape.reserve(gguf_tensor.shape.size());
        for (uint64_t dim : gguf_tensor.shape) {
            file_shape.push_back(static_cast<int64_t>(dim));
        }
        
        // Map GGUF dtype to IR DataType
        ir::DataType ir_dtype = mapGGUFDtypeToIR(gguf_tensor.dtype);
        
        // Get original GGUF dtype string for metadata
        std::string gguf_dtype_str = ggufDtypeToString(gguf_tensor.dtype);
        
        // Create IR tensor with EXACT file shape (no reshaping)
        ir::Tensor* ir_tensor = graph->addTensor(name, file_shape, ir_dtype);
        ir_tensor->byteOffset = gguf_tensor.offset;
        ir_tensor->original_shape = file_shape;  // Same as shape since we don't reshape
        ir_tensor->layout_transposed = false;  // We don't transpose in loader
        ir_tensor->metadata["gguf_dtype"] = gguf_dtype_str;
        
        // For LLaMA models: detect output.weight with [vocab, hidden] layout
        // GGUF stores it as [vocab, hidden], but matmul expects [hidden, vocab] as right operand
        // Mark it for transpose in codegen (transB=true) rather than reshaping here
        if (name == "output.weight" && file_shape.size() == 2) {
            int64_t d0 = file_shape[0];
            int64_t d1 = file_shape[1];
            // If shape is [vocab, hidden] where vocab > hidden, mark for transpose
            if (d0 > d1) {
                ir_tensor->metadata["needs_transpose"] = "true";
            }
        }
        // Do NOT transpose tok_embeddings.weight or other tensors - only output.weight
    }
    
    return graph;
}

} // namespace frontend
} // namespace mlc
