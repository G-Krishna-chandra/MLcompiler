#include "compiler/mlir/exec/Q4MatMul.h"

#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/ops.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace mx = mlx::core;

namespace mlir::mlc::exec {

using ::mlc::frontend::GGUFLoader;
using ::mlc::frontend::GGUFTensorInfo;

namespace {

// One simdgroup (32 lanes on Apple GPUs) cooperates on a single output
// element y[s, o]. The lanes interleave over the `blocks` of weight row
// `o`, each decoding its assigned 18-byte Q4_0 blocks on the fly and
// accumulating a partial dot product. A final `simd_sum` reduces the 32
// partial sums into one; lane 0 writes the result.
//
// Q4_0 layout (per 32-element block, 18 bytes total):
//   bytes [0..1]   : fp16 scale `d`
//   bytes [2..17]  : 16 bytes of packed nibbles. byte j → values at
//                    positions j and j+16; nibble value is (n - 8).
// Source matches `dotProductRowQ4_0` in compiler/runtime/quantization.cpp
// for bit-level parity.
//
// Dispatch: grid = (n_out * 32, n_seq, 1), threadgroup = (32, 1, 1). The
// 32 here is intentionally the simdgroup width.
constexpr const char *kQ4MatMulSource = R"(
  uint o = thread_position_in_grid.x / 32u;
  uint lane = thread_position_in_grid.x % 32u;
  uint s = thread_position_in_grid.y;
  if (o >= (uint)n_out || s >= (uint)n_seq) return;
  const uint blocks = (uint)n_in / 32u;
  const uint row_bytes = blocks * 18u;
  const device uchar* row = w + o * row_bytes;
  const device T* x_row = x + s * (uint)n_in;
  float acc = 0.0f;
  for (uint b = lane; b < blocks; b += 32u) {
    const device uchar* block = row + b * 18u;
    half scale_h = *((const device half*)block);
    float scale = (float)scale_h;
    const device uchar* qs = block + 2;
    uint base = b * 32u;
    for (uint j = 0; j < 16u; ++j) {
      uchar byte = qs[j];
      int lo = (int)(byte & 0x0F) - 8;
      int hi = (int)((byte >> 4) & 0x0F) - 8;
      acc += (float)x_row[base + j]      * (float)lo * scale;
      acc += (float)x_row[base + j + 16] * (float)hi * scale;
    }
  }
  float total = simd_sum(acc);
  if (lane == 0u) {
    y[s * (uint)n_out + o] = (T)total;
  }
)";

// MLX's metal_kernel returns a callable we want to keep around for the life
// of the process so we don't pay the JIT cost more than once.
mx::fast::CustomKernelFunction &getKernel() {
  static std::once_flag flag;
  static mx::fast::CustomKernelFunction kernel;
  std::call_once(flag, [] {
    kernel = mx::fast::metal_kernel(
        /*name=*/"mlc_q4_0_matmul",
        /*input_names=*/{"x", "w"},
        /*output_names=*/{"y"},
        /*source=*/kQ4MatMulSource,
        /*header=*/"",
        /*ensure_row_contiguous=*/true,
        /*atomic_outputs=*/false);
  });
  return kernel;
}

// 3-output Q4_0 matmul. Identical dot-product math; the only difference
// is the lane-0 store, which routes each output row to one of three
// destination buffers based on whether `o` falls in the Q, K, or V
// region of the concat'd weight.
constexpr const char *kQ4MatMulQKVSource = R"(
  uint o = thread_position_in_grid.x / 32u;
  uint lane = thread_position_in_grid.x % 32u;
  uint s = thread_position_in_grid.y;
  uint nq = (uint)n_q;
  uint nk = (uint)n_k;
  uint nv = (uint)n_v;
  uint n_out = nq + nk + nv;
  if (o >= n_out || s >= (uint)n_seq) return;
  const uint blocks = (uint)n_in / 32u;
  const uint row_bytes = blocks * 18u;
  const device uchar* row = w + o * row_bytes;
  const device T* x_row = x + s * (uint)n_in;
  float acc = 0.0f;
  for (uint b = lane; b < blocks; b += 32u) {
    const device uchar* block = row + b * 18u;
    half scale_h = *((const device half*)block);
    float scale = (float)scale_h;
    const device uchar* qs = block + 2;
    uint base = b * 32u;
    for (uint j = 0; j < 16u; ++j) {
      uchar byte = qs[j];
      int lo = (int)(byte & 0x0F) - 8;
      int hi = (int)((byte >> 4) & 0x0F) - 8;
      acc += (float)x_row[base + j]      * (float)lo * scale;
      acc += (float)x_row[base + j + 16] * (float)hi * scale;
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

mx::fast::CustomKernelFunction &getKernelQKV() {
  static std::once_flag flag;
  static mx::fast::CustomKernelFunction kernel;
  std::call_once(flag, [] {
    kernel = mx::fast::metal_kernel(
        /*name=*/"mlc_q4_0_matmul_qkv",
        /*input_names=*/{"x", "w"},
        /*output_names=*/{"y_q", "y_k", "y_v"},
        /*source=*/kQ4MatMulQKVSource,
        /*header=*/"",
        /*ensure_row_contiguous=*/true,
        /*atomic_outputs=*/false);
  });
  return kernel;
}

// 2-output variant for FFN gate+up. Same routing pattern, two output
// buffers instead of three.
constexpr const char *kQ4MatMulGateUpSource = R"(
  uint o = thread_position_in_grid.x / 32u;
  uint lane = thread_position_in_grid.x % 32u;
  uint s = thread_position_in_grid.y;
  uint ng = (uint)n_gate;
  uint nu = (uint)n_up;
  uint n_out = ng + nu;
  if (o >= n_out || s >= (uint)n_seq) return;
  const uint blocks = (uint)n_in / 32u;
  const uint row_bytes = blocks * 18u;
  const device uchar* row = w + o * row_bytes;
  const device T* x_row = x + s * (uint)n_in;
  float acc = 0.0f;
  for (uint b = lane; b < blocks; b += 32u) {
    const device uchar* block = row + b * 18u;
    half scale_h = *((const device half*)block);
    float scale = (float)scale_h;
    const device uchar* qs = block + 2;
    uint base = b * 32u;
    for (uint j = 0; j < 16u; ++j) {
      uchar byte = qs[j];
      int lo = (int)(byte & 0x0F) - 8;
      int hi = (int)((byte >> 4) & 0x0F) - 8;
      acc += (float)x_row[base + j]      * (float)lo * scale;
      acc += (float)x_row[base + j + 16] * (float)hi * scale;
    }
  }
  float total = simd_sum(acc);
  if (lane == 0u) {
    if (o < ng) {
      y_gate[s * ng + o] = (T)total;
    } else {
      y_up[s * nu + (o - ng)] = (T)total;
    }
  }
)";

mx::fast::CustomKernelFunction &getKernelGateUp() {
  static std::once_flag flag;
  static mx::fast::CustomKernelFunction kernel;
  std::call_once(flag, [] {
    kernel = mx::fast::metal_kernel(
        /*name=*/"mlc_q4_0_matmul_gate_up",
        /*input_names=*/{"x", "w"},
        /*output_names=*/{"y_gate", "y_up"},
        /*source=*/kQ4MatMulGateUpSource,
        /*header=*/"",
        /*ensure_row_contiguous=*/true,
        /*atomic_outputs=*/false);
  });
  return kernel;
}

} // namespace

mx::array gufWeightToBytesMLX(const GGUFLoader &loader,
                              const std::string &tensor_name) {
  const auto &info = loader.tensors().at(tensor_name);
  auto raw = loader.loadTensorData(info);
  uint8_t *buf = static_cast<uint8_t *>(std::malloc(raw.size()));
  if (!buf)
    throw std::bad_alloc();
  std::memcpy(buf, raw.data(), raw.size());
  mx::Shape shape{static_cast<int>(raw.size())};
  return mx::array(static_cast<void *>(buf), std::move(shape), mx::uint8,
                   [](void *p) { std::free(p); });
}

mx::array q4_0_matmul(const mx::array &x, const mx::array &w_bytes,
                      int in_dim, int out_dim) {
  if (x.ndim() != 2)
    throw std::runtime_error("q4_0_matmul: x must be rank-2");
  if (x.shape(1) != in_dim)
    throw std::runtime_error("q4_0_matmul: x last dim must equal in_dim");
  int seq = x.shape(0);
  auto &kernel = getKernel();
  // 32 lanes per output, n_out outputs along grid.x, n_seq along grid.y.
  std::tuple<int, int, int> grid{out_dim * 32, seq, 1};
  std::tuple<int, int, int> threadgroup{32, 1, 1};
  auto results = kernel(
      /*inputs=*/{x, w_bytes},
      /*output_shapes=*/{mx::Shape{seq, out_dim}},
      /*output_dtypes=*/{x.dtype()},
      grid, threadgroup,
      /*template_args=*/{{"T", x.dtype()}, {"n_in", in_dim}, {"n_out", out_dim},
                          {"n_seq", seq}},
      /*init_value=*/std::nullopt,
      /*verbose=*/false,
      /*stream=*/{});
  return results[0];
}

Q4MatMulQKVResult q4_0_matmul_qkv(const mx::array &x,
                                  const mx::array &w_bytes,
                                  int in_dim, int n_q, int n_k, int n_v) {
  if (x.ndim() != 2)
    throw std::runtime_error("q4_0_matmul_qkv: x must be rank-2");
  if (x.shape(1) != in_dim)
    throw std::runtime_error("q4_0_matmul_qkv: x last dim must equal in_dim");
  int seq = x.shape(0);
  int n_total = n_q + n_k + n_v;
  auto &kernel = getKernelQKV();
  std::tuple<int, int, int> grid{n_total * 32, seq, 1};
  std::tuple<int, int, int> threadgroup{32, 1, 1};
  auto results = kernel(
      /*inputs=*/{x, w_bytes},
      /*output_shapes=*/{mx::Shape{seq, n_q}, mx::Shape{seq, n_k},
                          mx::Shape{seq, n_v}},
      /*output_dtypes=*/{x.dtype(), x.dtype(), x.dtype()},
      grid, threadgroup,
      /*template_args=*/{{"T", x.dtype()}, {"n_in", in_dim},
                          {"n_q", n_q}, {"n_k", n_k}, {"n_v", n_v},
                          {"n_seq", seq}},
      /*init_value=*/std::nullopt,
      /*verbose=*/false,
      /*stream=*/{});
  return {results[0], results[1], results[2]};
}

Q4MatMulGateUpResult q4_0_matmul_gate_up(const mx::array &x,
                                         const mx::array &w_bytes,
                                         int in_dim, int n_gate, int n_up) {
  if (x.ndim() != 2)
    throw std::runtime_error("q4_0_matmul_gate_up: x must be rank-2");
  if (x.shape(1) != in_dim)
    throw std::runtime_error("q4_0_matmul_gate_up: x last dim must equal in_dim");
  int seq = x.shape(0);
  int n_total = n_gate + n_up;
  auto &kernel = getKernelGateUp();
  std::tuple<int, int, int> grid{n_total * 32, seq, 1};
  std::tuple<int, int, int> threadgroup{32, 1, 1};
  auto results = kernel(
      /*inputs=*/{x, w_bytes},
      /*output_shapes=*/{mx::Shape{seq, n_gate}, mx::Shape{seq, n_up}},
      /*output_dtypes=*/{x.dtype(), x.dtype()},
      grid, threadgroup,
      /*template_args=*/{{"T", x.dtype()}, {"n_in", in_dim},
                          {"n_gate", n_gate}, {"n_up", n_up},
                          {"n_seq", seq}},
      /*init_value=*/std::nullopt,
      /*verbose=*/false,
      /*stream=*/{});
  return {results[0], results[1]};
}

} // namespace mlir::mlc::exec
