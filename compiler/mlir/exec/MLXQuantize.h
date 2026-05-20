#pragma once

#include "compiler/frontends/gguf_loader.hpp"

#include <mlx/array.h>

#include <string>

namespace mlir::mlc::exec {

// MLX-native quantized weight derived from a GGUF Q4_0 tensor. The three
// arrays match what `mx::quantized_matmul` expects with
// `group_size=32, bits=4, mode="affine"`:
//
//   w_q     : uint32, shape [out, in/8]    (8 nibbles per uint32)
//   scales  : fp16,   shape [out, in/32]   (one per 32-element group)
//   biases  : fp16,   shape [out, in/32]   (one per 32-element group)
//
// The mapping from GGUF Q4_0 (signed-symmetric, (n-8)*d) to MLX affine
// (asymmetric, n*scale+bias) is exact: scale = d, bias = -8*d. No
// re-quantization, no precision loss.
//
// in_dim is the inner dimension of the matmul (input feature dim);
// out_dim is the number of output rows. GGUF stores [in, out] with
// shape[0]=in (innermost) — this struct reports the logical matmul dims.
struct MLXQuantWeights {
  mlx::core::array w_q;
  mlx::core::array scales;
  mlx::core::array biases;
  int in_dim;
  int out_dim;
};

// Build the (w_q, scales, biases) triple from a GGUF Q4_0 tensor.
// Bit-level conversion only; the result is evaluated on stream {} and
// ready to feed to `mx::quantized_matmul`.
MLXQuantWeights
ggufQ4_0ToMLXQuantized(const ::mlc::frontend::GGUFLoader &loader,
                       const std::string &tensor_name);

} // namespace mlir::mlc::exec
