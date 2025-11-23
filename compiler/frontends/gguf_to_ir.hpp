#pragma once

#include "gguf_loader.hpp"
#include "ir/ir.hpp"
#include <memory>

namespace mlc {
namespace frontend {

// Convert GGUF model to IR Graph
// Currently only imports tensors (weights), no operations
std::unique_ptr<ir::Graph> GGUFToIR(const GGUFLoader& loader);

// Helper function to map GGUF dtype to IR DataType
ir::DataType mapGGUFDtypeToIR(uint32_t gguf_dtype);

} // namespace frontend
} // namespace mlc







