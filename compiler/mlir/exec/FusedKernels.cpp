#include "compiler/mlir/exec/FusedKernels.h"

#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/ops.h>

#include <mutex>
#include <stdexcept>

namespace mx = mlx::core;

namespace mlir::mlc::exec {

namespace {

// ── X1: fused_norm_q4_matmul ─────────────────────────────────────────────────
//
// One simdgroup (32 lanes) per output element `o` per sequence position `s`.
//
// Step 1 — RMSNorm:
//   Each lane accumulates sum_sq for its strided slice of x (i=lane, lane+32, …).
//   simd_sum collapses across the 32 lanes → total sum_sq.
//   rms_scale = rsqrt(sum_sq / n_in + 1e-5).
//
// Step 2 — Fused Q4_0 matmul (same blocking as the standalone kernel):
//   For each Q4_0 block b handled by this lane, the two x values accessed
//   (base+j and base+j+16) are first multiplied by rms_scale * gamma[…]
//   BEFORE applying the Q4_0 dot product. gamma is fp16/fp32, same type as x.
//   No intermediate x_norm buffer is written; the normalized values exist
//   only in fp32 registers.
//
// x is read twice: once for sum_sq, once inside the Q4_0 block loop.
// The extra bandwidth (~8 KB at K=2048) is negligible vs the dispatch
// overhead saved (~100 µs per dispatch).
constexpr const char *kFusedNormQ4Source = R"(
  uint o    = thread_position_in_grid.x / 32u;
  uint lane = thread_position_in_grid.x % 32u;
  uint s    = thread_position_in_grid.y;
  if (o >= (uint)n_out || s >= (uint)n_seq) return;

  const device T* x_row = x + s * (uint)n_in;
  const device T* g     = gamma;        // [n_in], broadcast over seq

  // ── Step 1: RMSNorm ──────────────────────────────────────────────────
  float sum_sq = 0.0f;
  for (uint i = lane; i < (uint)n_in; i += 32u) {
    float xi = (float)x_row[i];
    sum_sq += xi * xi;
  }
  sum_sq = simd_sum(sum_sq);
  float rms_scale = rsqrt(sum_sq / (float)n_in + 1e-5f);

  // ── Step 2: Q4_0 matmul with inline norm application ─────────────────
  const uint blocks    = (uint)n_in / 32u;
  const uint row_bytes = blocks * 18u;
  const device uchar* row = w + o * row_bytes;
  float acc = 0.0f;
  for (uint b = lane; b < blocks; b += 32u) {
    const device uchar* block = row + b * 18u;
    half   scale_h = *((const device half*)block);
    float  q4_scale = (float)scale_h;
    const device uchar* qs = block + 2;
    uint base = b * 32u;
    for (uint j = 0; j < 16u; ++j) {
      uchar byte = qs[j];
      int lo = (int)(byte & 0x0F) - 8;
      int hi = (int)((byte >> 4) & 0x0F) - 8;
      // x_norm = x * rms_scale * gamma — computed on the fly, no buffer
      float xlo = (float)x_row[base + j]      * rms_scale * (float)g[base + j];
      float xhi = (float)x_row[base + j + 16] * rms_scale * (float)g[base + j + 16];
      acc += xlo * (float)lo * q4_scale;
      acc += xhi * (float)hi * q4_scale;
    }
  }
  float total = simd_sum(acc);
  if (lane == 0u) {
    y[s * (uint)n_out + o] = (T)total;
  }
)";

mx::fast::CustomKernelFunction &kernelNormQ4() {
  static std::once_flag flag;
  static mx::fast::CustomKernelFunction kern;
  std::call_once(flag, [] {
    kern = mx::fast::metal_kernel(
        "mlc_fused_norm_q4",
        {"x", "gamma", "w"},
        {"y"},
        kFusedNormQ4Source,
        /*header=*/"",
        /*ensure_row_contiguous=*/true,
        /*atomic_outputs=*/false);
  });
  return kern;
}

// ── X2: fused_norm_q4_qkv — 3 output buffers ─────────────────────────────────
//
// Identical to X1 except lane-0 routes the result to one of y_q, y_k, y_v
// based on which region of the concat'd weight `o` falls in.
// Dispatch stays (n_total * 32, n_seq, 1) — one simdgroup per output element.
constexpr const char *kFusedNormQ4QKVSource = R"(
  uint o    = thread_position_in_grid.x / 32u;
  uint lane = thread_position_in_grid.x % 32u;
  uint s    = thread_position_in_grid.y;
  uint nq = (uint)n_q;
  uint nk = (uint)n_k;
  uint nv = (uint)n_v;
  uint n_total = nq + nk + nv;
  if (o >= n_total || s >= (uint)n_seq) return;

  const device T* x_row = x + s * (uint)n_in;
  const device T* g     = gamma;

  // RMSNorm
  float sum_sq = 0.0f;
  for (uint i = lane; i < (uint)n_in; i += 32u) {
    float xi = (float)x_row[i];
    sum_sq += xi * xi;
  }
  sum_sq = simd_sum(sum_sq);
  float rms_scale = rsqrt(sum_sq / (float)n_in + 1e-5f);

  // Fused Q4_0 matmul
  const uint blocks    = (uint)n_in / 32u;
  const uint row_bytes = blocks * 18u;
  const device uchar* row = w + o * row_bytes;
  float acc = 0.0f;
  for (uint b = lane; b < blocks; b += 32u) {
    const device uchar* block = row + b * 18u;
    half  scale_h  = *((const device half*)block);
    float q4_scale = (float)scale_h;
    const device uchar* qs = block + 2;
    uint base = b * 32u;
    for (uint j = 0; j < 16u; ++j) {
      uchar byte = qs[j];
      int lo = (int)(byte & 0x0F) - 8;
      int hi = (int)((byte >> 4) & 0x0F) - 8;
      float xlo = (float)x_row[base + j]      * rms_scale * (float)g[base + j];
      float xhi = (float)x_row[base + j + 16] * rms_scale * (float)g[base + j + 16];
      acc += xlo * (float)lo * q4_scale;
      acc += xhi * (float)hi * q4_scale;
    }
  }
  float total = simd_sum(acc);
  if (lane == 0u) {
    if (o < nq) {
      y_q[s * nq + o] = (T)total;
    } else if (o < nq + nk) {
      y_k[s * nk + (o - nq)] = (T)total;
    } else {
      y_v[s * nv + (o - nq - nk)] = (T)total;
    }
  }
)";

mx::fast::CustomKernelFunction &kernelNormQ4QKV() {
  static std::once_flag flag;
  static mx::fast::CustomKernelFunction kern;
  std::call_once(flag, [] {
    kern = mx::fast::metal_kernel(
        "mlc_fused_norm_q4_qkv",
        {"x", "gamma", "w"},
        {"y_q", "y_k", "y_v"},
        kFusedNormQ4QKVSource,
        /*header=*/"",
        /*ensure_row_contiguous=*/true,
        /*atomic_outputs=*/false);
  });
  return kern;
}

// ── X3a: fused_norm_q4_gate_up_silu ──────────────────────────────────────────
//
// Fuses: RMSNorm → gate_matmul → up_matmul → silu(gate)*up
//
// The concat'd weight W_gate‖W_up has shape [(2*ffn_dim, n_in)] in Q4_0.
// One simdgroup per output element of the concat'd result.
// Lane-0 writes gate result to y_gate (if o < ffn_dim) or up result to y_up.
// A second kernel pass (X3b) then computes silu(y_gate)*y_up + ffn_down.
//
// HOWEVER: to truly fuse silu*mul into the same kernel, we need both gate[o]
// and up[o] simultaneously. They're in different halves of the weight matrix.
// Approach: two simdgroups cooperate on the same output index — but that
// requires atomic reduction, which is complex.
//
// SIMPLER: keep gate+up in one dispatch (2 outputs = gate_raw, up_raw), then
// silu*mul as a second tiny dispatch. Saves: norm dispatch + possibly the
// silu*mul can be batched into the next op.
//
// For V1: gate+up fused with norm → 2 outputs in 1 dispatch (saves 1 norm).
// silu*mul stays as a separate (cheap) dispatch.
constexpr const char *kFusedNormQ4GateUpSource = R"(
  uint o    = thread_position_in_grid.x / 32u;
  uint lane = thread_position_in_grid.x % 32u;
  uint s    = thread_position_in_grid.y;
  uint ng = (uint)ffn_dim;
  uint nu = (uint)ffn_dim;
  uint n_total = ng + nu;
  if (o >= n_total || s >= (uint)n_seq) return;

  const device T* x_row = x + s * (uint)n_in;
  const device T* g     = gamma;

  // RMSNorm
  float sum_sq = 0.0f;
  for (uint i = lane; i < (uint)n_in; i += 32u) {
    float xi = (float)x_row[i];
    sum_sq += xi * xi;
  }
  sum_sq = simd_sum(sum_sq);
  float rms_scale = rsqrt(sum_sq / (float)n_in + 1e-5f);

  // Fused Q4_0 matmul (gate ‖ up concat along out-dim)
  const uint blocks    = (uint)n_in / 32u;
  const uint row_bytes = blocks * 18u;
  const device uchar* row = w_gate_up + o * row_bytes;
  float acc = 0.0f;
  for (uint b = lane; b < blocks; b += 32u) {
    const device uchar* block = row + b * 18u;
    half  scale_h  = *((const device half*)block);
    float q4_scale = (float)scale_h;
    const device uchar* qs = block + 2;
    uint base = b * 32u;
    for (uint j = 0; j < 16u; ++j) {
      uchar byte = qs[j];
      int lo = (int)(byte & 0x0F) - 8;
      int hi = (int)((byte >> 4) & 0x0F) - 8;
      float xlo = (float)x_row[base + j]      * rms_scale * (float)g[base + j];
      float xhi = (float)x_row[base + j + 16] * rms_scale * (float)g[base + j + 16];
      acc += xlo * (float)lo * q4_scale;
      acc += xhi * (float)hi * q4_scale;
    }
  }
  float total = simd_sum(acc);
  if (lane == 0u) {
    if (o < ng) {
      // Gate: apply SiLU inline so the caller doesn't need another dispatch
      // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
      float g_val = total;
      float s_val = 1.0f / (1.0f + exp(-g_val));
      y_gate_silu[s * ng + o] = (T)(g_val * s_val);
    } else {
      y_up[s * nu + (o - ng)] = (T)total;
    }
  }
)";

mx::fast::CustomKernelFunction &kernelNormQ4GateUpSilu() {
  static std::once_flag flag;
  static mx::fast::CustomKernelFunction kern;
  std::call_once(flag, [] {
    kern = mx::fast::metal_kernel(
        "mlc_fused_norm_q4_gate_up_silu",
        {"x", "gamma", "w_gate_up"},
        {"y_gate_silu", "y_up"},
        kFusedNormQ4GateUpSource,
        /*header=*/"",
        /*ensure_row_contiguous=*/true,
        /*atomic_outputs=*/false);
  });
  return kern;
}

// ── X3b: fused_q4_matmul_residual ────────────────────────────────────────────
//
// Fuses: ffn_down = q4_0_matmul(h, W_down) and residual_add = residual + ffn_down
// One dispatch computes both, writing the final output.
//
// Grid: (n_out * 32, n_seq, 1) — same as standalone Q4_0 matmul.
// residual must be [seq, n_out] in the same dtype as h.
constexpr const char *kFusedQ4ResidualSource = R"(
  uint o    = thread_position_in_grid.x / 32u;
  uint lane = thread_position_in_grid.x % 32u;
  uint s    = thread_position_in_grid.y;
  if (o >= (uint)n_out || s >= (uint)n_seq) return;

  const device T* h_row = h + s * (uint)n_in;

  const uint blocks    = (uint)n_in / 32u;
  const uint row_bytes = blocks * 18u;
  const device uchar* row = w + o * row_bytes;
  float acc = 0.0f;
  for (uint b = lane; b < blocks; b += 32u) {
    const device uchar* block = row + b * 18u;
    half  scale_h  = *((const device half*)block);
    float q4_scale = (float)scale_h;
    const device uchar* qs = block + 2;
    uint base = b * 32u;
    for (uint j = 0; j < 16u; ++j) {
      uchar byte = qs[j];
      int lo = (int)(byte & 0x0F) - 8;
      int hi = (int)((byte >> 4) & 0x0F) - 8;
      acc += (float)h_row[base + j]      * (float)lo * q4_scale;
      acc += (float)h_row[base + j + 16] * (float)hi * q4_scale;
    }
  }
  float total = simd_sum(acc);
  if (lane == 0u) {
    // Residual add inline — no separate dispatch needed.
    float res = (float)residual[s * (uint)n_out + o];
    y[s * (uint)n_out + o] = (T)(total + res);
  }
)";

mx::fast::CustomKernelFunction &kernelQ4Residual() {
  static std::once_flag flag;
  static mx::fast::CustomKernelFunction kern;
  std::call_once(flag, [] {
    kern = mx::fast::metal_kernel(
        "mlc_fused_q4_residual",
        {"h", "residual", "w"},
        {"y"},
        kFusedQ4ResidualSource,
        /*header=*/"",
        /*ensure_row_contiguous=*/true,
        /*atomic_outputs=*/false);
  });
  return kern;
}

// ── X3a′: gate+up+silu without norm ──────────────────────────────────────────
constexpr const char *kFusedQ4GateUpSiluSource = R"(
  uint o    = thread_position_in_grid.x / 32u;
  uint lane = thread_position_in_grid.x % 32u;
  uint s    = thread_position_in_grid.y;
  uint ng = (uint)ffn_dim;
  uint nu = (uint)ffn_dim;
  uint n_total = ng + nu;
  if (o >= n_total || s >= (uint)n_seq) return;

  const device T* x_row = x + s * (uint)n_in;

  const uint blocks    = (uint)n_in / 32u;
  const uint row_bytes = blocks * 18u;
  const device uchar* row = w_gate_up + o * row_bytes;
  float acc = 0.0f;
  for (uint b = lane; b < blocks; b += 32u) {
    const device uchar* block = row + b * 18u;
    half  scale_h  = *((const device half*)block);
    float q4_scale = (float)scale_h;
    const device uchar* qs = block + 2;
    uint base = b * 32u;
    for (uint j = 0; j < 16u; ++j) {
      uchar byte = qs[j];
      int lo = (int)(byte & 0x0F) - 8;
      int hi = (int)((byte >> 4) & 0x0F) - 8;
      acc += (float)x_row[base + j]      * (float)lo * q4_scale;
      acc += (float)x_row[base + j + 16] * (float)hi * q4_scale;
    }
  }
  float total = simd_sum(acc);
  if (lane == 0u) {
    if (o < ng) {
      float g_val = total;
      float s_val = 1.0f / (1.0f + exp(-g_val));
      y_gate_silu[s * ng + o] = (T)(g_val * s_val);
    } else {
      y_up[s * nu + (o - ng)] = (T)total;
    }
  }
)";

mx::fast::CustomKernelFunction &kernelQ4GateUpSilu() {
  static std::once_flag flag;
  static mx::fast::CustomKernelFunction kern;
  std::call_once(flag, [] {
    kern = mx::fast::metal_kernel(
        "mlc_fused_q4_gate_up_silu",
        {"x", "w_gate_up"},
        {"y_gate_silu", "y_up"},
        kFusedQ4GateUpSiluSource,
        /*header=*/"",
        /*ensure_row_contiguous=*/true,
        /*atomic_outputs=*/false);
  });
  return kern;
}

} // namespace

// ── Public API implementations ────────────────────────────────────────────────

mx::array fusedNormQ4Matmul(const mx::array &x, const mx::array &gamma,
                            const mx::array &w_bytes, float /*eps*/,
                            int K, int N) {
  if (x.ndim() != 2 || x.shape(1) != K)
    throw std::runtime_error("fusedNormQ4Matmul: x must be [seq, K]");
  int seq = x.shape(0);
  auto &kern = kernelNormQ4();
  std::tuple<int,int,int> grid{N * 32, seq, 1};
  std::tuple<int,int,int> tg{32, 1, 1};
  auto results = kern(
      {x, gamma, w_bytes},
      {mx::Shape{seq, N}},
      {x.dtype()},
      grid, tg,
      {{"T", x.dtype()}, {"n_in", K}, {"n_out", N}, {"n_seq", seq}},
      std::nullopt, false, {});
  return results[0];
}

FusedNormQKVResult fusedNormQ4MatmulQKV(const mx::array &x,
                                         const mx::array &gamma,
                                         const mx::array &w_bytes, float /*eps*/,
                                         int K, int n_q, int n_k, int n_v) {
  if (x.ndim() != 2 || x.shape(1) != K)
    throw std::runtime_error("fusedNormQ4MatmulQKV: x must be [seq, K]");
  int seq = x.shape(0);
  int n_total = n_q + n_k + n_v;
  auto &kern = kernelNormQ4QKV();
  std::tuple<int,int,int> grid{n_total * 32, seq, 1};
  std::tuple<int,int,int> tg{32, 1, 1};
  auto results = kern(
      {x, gamma, w_bytes},
      {mx::Shape{seq, n_q}, mx::Shape{seq, n_k}, mx::Shape{seq, n_v}},
      {x.dtype(), x.dtype(), x.dtype()},
      grid, tg,
      {{"T", x.dtype()}, {"n_in", K}, {"n_seq", seq},
       {"n_q", n_q}, {"n_k", n_k}, {"n_v", n_v}},
      std::nullopt, false, {});
  return {results[0], results[1], results[2]};
}

mx::array fusedNormQ4GateUpSilu(const mx::array &x, const mx::array &gamma,
                                 const mx::array &w_gate_up_bytes,
                                 float /*eps*/, int K, int ffn_dim) {
  if (x.ndim() != 2 || x.shape(1) != K)
    throw std::runtime_error("fusedNormQ4GateUpSilu: x must be [seq, K]");
  int seq    = x.shape(0);
  int n_total = ffn_dim * 2;
  auto &kern = kernelNormQ4GateUpSilu();
  std::tuple<int,int,int> grid{n_total * 32, seq, 1};
  std::tuple<int,int,int> tg{32, 1, 1};
  // Returns (y_gate_silu [seq, ffn_dim], y_up [seq, ffn_dim]).
  // The caller multiplies them: h = y_gate_silu * y_up.
  auto results = kern(
      {x, gamma, w_gate_up_bytes},
      {mx::Shape{seq, ffn_dim}, mx::Shape{seq, ffn_dim}},
      {x.dtype(), x.dtype()},
      grid, tg,
      {{"T", x.dtype()}, {"n_in", K}, {"n_seq", seq}, {"ffn_dim", ffn_dim}},
      std::nullopt, false, {});
  // Multiply gate_silu * up to produce h = silu(gate)*up in-kernel would
  // require both to be ready simultaneously (different o ranges). Since they're
  // from the same dispatch, MLX evaluates them together — just multiply lazily.
  return mx::multiply(results[0], results[1]);
}

mx::array fusedQ4GateUpSilu(const mx::array &x_normed,
                            const mx::array &w_gate_up_bytes,
                            int K, int ffn_dim) {
  if (x_normed.ndim() != 2 || x_normed.shape(1) != K)
    throw std::runtime_error("fusedQ4GateUpSilu: x must be [seq, K]");
  int seq = x_normed.shape(0);
  int n_total = ffn_dim * 2;
  auto &kern = kernelQ4GateUpSilu();
  std::tuple<int,int,int> grid{n_total * 32, seq, 1};
  std::tuple<int,int,int> tg{32, 1, 1};
  auto results = kern(
      {x_normed, w_gate_up_bytes},
      {mx::Shape{seq, ffn_dim}, mx::Shape{seq, ffn_dim}},
      {x_normed.dtype(), x_normed.dtype()},
      grid, tg,
      {{"T", x_normed.dtype()}, {"n_in", K}, {"n_seq", seq}, {"ffn_dim", ffn_dim}},
      std::nullopt, false, {});
  return mx::multiply(results[0], results[1]);
}

mx::array fusedQ4MatmulResidual(const mx::array &h,
                                 const mx::array &residual,
                                 const mx::array &w_bytes,
                                 int K_ffn, int N) {
  if (h.ndim() != 2 || h.shape(1) != K_ffn)
    throw std::runtime_error("fusedQ4MatmulResidual: h must be [seq, K_ffn]");
  if (residual.ndim() != 2 || residual.shape(0) != h.shape(0) ||
      residual.shape(1) != N)
    throw std::runtime_error("fusedQ4MatmulResidual: residual must be [seq, N]");
  int seq = h.shape(0);
  auto &kern = kernelQ4Residual();
  std::tuple<int,int,int> grid{N * 32, seq, 1};
  std::tuple<int,int,int> tg{32, 1, 1};
  auto results = kern(
      {h, residual, w_bytes},
      {mx::Shape{seq, N}},
      {h.dtype()},
      grid, tg,
      {{"T", h.dtype()}, {"n_in", K_ffn}, {"n_out", N}, {"n_seq", seq}},
      std::nullopt, false, {});
  return results[0];
}

} // namespace mlir::mlc::exec
