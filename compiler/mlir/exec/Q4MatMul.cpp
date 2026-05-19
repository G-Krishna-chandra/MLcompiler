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

} // namespace mlir::mlc::exec
