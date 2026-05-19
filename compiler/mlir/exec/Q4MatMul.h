#pragma once

#include "compiler/frontends/gguf_loader.hpp"

#include <mlx/array.h>

#include <string>

namespace mlir::mlc::exec {

// Build an `mx::array` of dtype uint8 holding the raw GGUF tensor bytes
// (no dequantization). Used by the Q4_0 custom kernel below as the weight
// operand.
mlx::core::array gufWeightToBytesMLX(const ::mlc::frontend::GGUFLoader &loader,
                                     const std::string &tensor_name);

// Q4_0 matmul via a custom Metal kernel.
// x: [seq, in_dim] float16/float32 activations
// w_bytes: uint8 array of length out_dim * (in_dim/32) * 18 — raw Q4_0
//          blocks for the [out, in] weight matrix.
// Returns: [seq, out_dim] in x's dtype.
mlx::core::array q4_0_matmul(const mlx::core::array &x,
                             const mlx::core::array &w_bytes, int in_dim,
                             int out_dim);

} // namespace mlir::mlc::exec
