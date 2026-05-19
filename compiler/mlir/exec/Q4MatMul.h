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

// Multi-output Q4_0 matmul. Single Metal dispatch over a concat'd
// weight matrix; each output lane writes to one of three destination
// buffers based on its output index. Equivalent to:
//   y = q4_0_matmul(x, w_concat, in_dim, n_q + n_k + n_v)
//   q = y[:, 0:n_q]; k = y[:, n_q:n_q+n_k]; v = y[:, n_q+n_k:...]
// but without the mx::slice allocations that the QKV-fused walker
// pays today.
//
// w_bytes must be the Q4_0 byte concat in Q ‖ K ‖ V order along the
// out-row axis — the same layout the executor already builds for the
// `qkv_concat_cache_`.
struct Q4MatMulQKVResult {
  mlx::core::array q, k, v;
};
Q4MatMulQKVResult q4_0_matmul_qkv(const mlx::core::array &x,
                                  const mlx::core::array &w_bytes,
                                  int in_dim, int n_q, int n_k, int n_v);

// Two-output variant for FFN gate+up. Same idea: one dispatch over a
// gate ‖ up concat'd weight; routing writes the first n_gate output
// rows to `gate`, the remaining n_up rows to `up`. Returns the pair
// so the caller can apply silu(gate) * up without an mx::slice.
struct Q4MatMulGateUpResult {
  mlx::core::array gate, up;
};
Q4MatMulGateUpResult q4_0_matmul_gate_up(const mlx::core::array &x,
                                         const mlx::core::array &w_bytes,
                                         int in_dim, int n_gate, int n_up);

} // namespace mlir::mlc::exec
