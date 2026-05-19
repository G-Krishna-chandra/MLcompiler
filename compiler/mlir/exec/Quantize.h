#pragma once

#include "compiler/frontends/gguf_loader.hpp"

#include <cstdint>
#include <vector>

namespace mlir::mlc::exec {

// Dequantize a single GGUF tensor to a contiguous fp32 buffer, row-major in
// storage order. Uses the existing runtime's dequant routines so we match the
// hand-written runtime path byte-for-byte at the dequant boundary.
//
// The returned vector has size `prod(tensor.shape)`. Caller owns it.
std::vector<float> dequantizeToF32(const ::mlc::frontend::GGUFLoader &loader,
                                   const std::string &tensor_name);

} // namespace mlir::mlc::exec
