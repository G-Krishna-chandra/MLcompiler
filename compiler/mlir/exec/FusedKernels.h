#pragma once

// Fused multi-op Metal kernels for dispatch-count reduction.
//
// Each function here replaces N separate MLX lazy ops with ONE custom Metal
// kernel, reducing Metal dispatch overhead at batch=1 decode.
//
// W1 profiling showed 85% of decode step time is dispatch overhead
// (~110 dispatches × 100 µs = 11 ms of 13 ms). These kernels target
// the most common adjacent-op patterns.

#include <mlx/array.h>

namespace mlir::mlc::exec {

// ── X1: Fused RMSNorm + Q4_0 matmul ─────────────────────────────────────────
//
// Replaces:
//   tmp  = rms_norm(x, gamma, eps)       ← 1 Metal dispatch
//   y    = q4_0_matmul(tmp, w, K, N)     ← 1 Metal dispatch
// With one Metal dispatch.
//
// x:      [seq, K]   fp16 or fp32 input activations
// gamma:  [K]        fp16 or fp32 RMSNorm gain weights
// w_bytes: uint8 [N * ceil(K/32) * 18]  raw GGUF Q4_0 weight bytes
// eps:    RMSNorm epsilon (typically 1e-5)
// K, N:   input/output feature dimensions
//
// Returns: [seq, N] in x.dtype()
mlx::core::array fusedNormQ4Matmul(const mlx::core::array &x,
                                   const mlx::core::array &gamma,
                                   const mlx::core::array &w_bytes,
                                   float eps, int K, int N);

// ── X2: Fused RMSNorm + Q4_0 matmul, 3 outputs (norm+QKV) ───────────────────
//
// Replaces:
//   tmp       = rms_norm(x, gamma, eps)   ← 1 dispatch
//   [Q, K, V] = q4_0_matmul(tmp, W_qkv)  ← 1 dispatch (currently concat'd)
// With one Metal dispatch.
//
// w_bytes: Q‖K‖V concat'd Q4_0 bytes (same layout as qkv_concat_cache_).
// n_q, n_k, n_v: output dims for each projection.
//
// Returns: Q=[seq, n_q], K=[seq, n_k], V=[seq, n_v] (all in x.dtype())
struct FusedNormQKVResult {
  mlx::core::array q, k, v;
};
FusedNormQKVResult fusedNormQ4MatmulQKV(const mlx::core::array &x,
                                        const mlx::core::array &gamma,
                                        const mlx::core::array &w_bytes,
                                        float eps, int K,
                                        int n_q, int n_k, int n_v);

// ── X3a: Fused RMSNorm + FFN gate+up + SiLU×mul ─────────────────────────────
//
// Replaces:
//   tmp      = rms_norm(x, gamma, eps)   ← 1 dispatch
//   gate_raw = q4_0_matmul(tmp, W_gate)  ← 1 dispatch  (fused via gate‖up concat)
//   up_raw   = q4_0_matmul(tmp, W_up)    (same dispatch as gate via concat)
//   h        = silu(gate_raw) * up_raw   ← 1 dispatch
// With one Metal dispatch.
//
// w_gate_up_bytes: gate‖up concat'd Q4_0 bytes (same as ffn_gateup_cache_).
// ffn_dim: output size of each of gate and up (N/2 of the concat'd weight).
//
// Returns h = silu(gate) * up: [seq, ffn_dim] in x.dtype()
mlx::core::array fusedNormQ4GateUpSilu(const mlx::core::array &x,
                                        const mlx::core::array &gamma,
                                        const mlx::core::array &w_gate_up_bytes,
                                        float eps, int K, int ffn_dim);

// ── X3a′: Fused Q4_0 gate+up + SiLU×mul (no norm — x already normalized) ─────
//
// Same as fusedNormQ4GateUpSilu but skips the RMSNorm step.
// Use this when the norm was already applied (x is already x_normed).
// Replaces: gate_matmul [1 dispatch] + silu [1] + mul [1] = 3 dispatches → 1.
mlx::core::array fusedQ4GateUpSilu(const mlx::core::array &x_normed,
                                    const mlx::core::array &w_gate_up_bytes,
                                    int K, int ffn_dim);

// ── X3b: Fused Q4_0 matmul + residual add ────────────────────────────────────
//
// Replaces:
//   y_proj = q4_0_matmul(h, W_down)   ← 1 dispatch
//   out    = residual + y_proj         ← 1 dispatch
// With one Metal dispatch.
//
// h:        [seq, K_ffn]  SiLU*mul output from X3a
// residual: [seq, N]      pre-FFN residual to add
// w_bytes:  Q4_0 bytes for W_down
//
// Returns: residual + down_proj(h): [seq, N] in h.dtype()
mlx::core::array fusedQ4MatmulResidual(const mlx::core::array &h,
                                        const mlx::core::array &residual,
                                        const mlx::core::array &w_bytes,
                                        int K_ffn, int N);

} // namespace mlir::mlc::exec
