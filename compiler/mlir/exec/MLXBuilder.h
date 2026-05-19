#pragma once

#include "compiler/frontends/gguf_loader.hpp"

#include <mlx/array.h>

#include <string>
#include <vector>

namespace mlir::mlc::exec {

// Convert a contiguous fp32 buffer to an mx::array (fp32, given shape).
// The array takes ownership of a copy of the buffer.
mlx::core::array fp32ToMLX(const std::vector<float> &data,
                           const std::vector<int> &shape);

// Dequantize a GGUF weight tensor and upload it as an fp32 mx::array.
// Convenience for v1 — slow path, future versions will keep weights quantized
// on-device and dequantize inside a fused Metal kernel.
mlx::core::array gufWeightToMLX(const ::mlc::frontend::GGUFLoader &loader,
                                const std::string &tensor_name);

// MLX matmul wrapper. Builds the lazy graph node; caller still has to call
// mlx::core::eval() to materialize.
mlx::core::array matmul(const mlx::core::array &x, const mlx::core::array &w,
                        bool transpose_b);

// Read an evaluated MLX array back to a host fp32 vector.
std::vector<float> mlxToF32(const mlx::core::array &a);

} // namespace mlir::mlc::exec
