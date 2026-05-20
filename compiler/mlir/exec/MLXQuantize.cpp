#include "compiler/mlir/exec/MLXQuantize.h"

#include "compiler/frontends/ggml_types.hpp"

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;

namespace mlir::mlc::exec {

using ::mlc::frontend::GGUFLoader;
using ::mlc::frontend::GGUFTensorInfo;

namespace {

// IEEE-754 binary16 → binary32. Matches MLX's internal fp16. We could pull
// in <Accelerate/Accelerate.h>'s vImage helpers, but a one-shot scalar
// converter is enough since this runs once per weight at load time.
float fp16ToFp32(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1u;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = sign << 31;
    } else {
      // Subnormal: renormalize.
      exp = 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3ffu;
      exp += 127 - 15;
      out = (sign << 31) | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    out = (sign << 31) | (0xffu << 23) | (mant << 13);
  } else {
    out = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}

// IEEE-754 binary32 → binary16. Round to nearest even. Sufficient for the
// small magnitudes (-8*d for Q4_0 scales) we feed here.
uint16_t fp32ToFp16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  uint32_t sign = (x >> 31) & 0x1u;
  uint32_t exp = (x >> 23) & 0xffu;
  uint32_t mant = x & 0x7fffffu;
  uint16_t out;
  if (exp == 0) {
    out = static_cast<uint16_t>(sign << 15);
  } else if (exp == 0xff) {
    out = static_cast<uint16_t>((sign << 15) | (0x1f << 10) | (mant ? 0x200u : 0));
  } else {
    int new_exp = static_cast<int>(exp) - 127 + 15;
    if (new_exp >= 31) {
      out = static_cast<uint16_t>((sign << 15) | (0x1f << 10));
    } else if (new_exp <= 0) {
      if (new_exp < -10) {
        out = static_cast<uint16_t>(sign << 15);
      } else {
        mant = (mant | 0x800000u) >> (1 - new_exp);
        uint32_t round = (mant >> 12) & 1u;
        uint16_t m = static_cast<uint16_t>(mant >> 13);
        m += static_cast<uint16_t>(round);
        out = static_cast<uint16_t>((sign << 15) | m);
      }
    } else {
      uint32_t round = (mant >> 12) & 1u;
      uint32_t m13 = mant >> 13;
      uint32_t m = m13 + round;
      if (m & 0x400u) {
        m = 0;
        new_exp += 1;
        if (new_exp >= 31)
          return static_cast<uint16_t>((sign << 15) | (0x1f << 10));
      }
      out = static_cast<uint16_t>((sign << 15) | (new_exp << 10) | m);
    }
  }
  return out;
}

mx::array allocU32MLX(std::vector<uint32_t> &&buf, mx::Shape shape) {
  uint32_t *p = static_cast<uint32_t *>(std::malloc(buf.size() * sizeof(uint32_t)));
  if (!p)
    throw std::bad_alloc();
  std::memcpy(p, buf.data(), buf.size() * sizeof(uint32_t));
  return mx::array(static_cast<void *>(p), std::move(shape), mx::uint32,
                   [](void *q) { std::free(q); });
}

mx::array allocF16MLX(std::vector<uint16_t> &&buf, mx::Shape shape) {
  uint16_t *p = static_cast<uint16_t *>(std::malloc(buf.size() * sizeof(uint16_t)));
  if (!p)
    throw std::bad_alloc();
  std::memcpy(p, buf.data(), buf.size() * sizeof(uint16_t));
  return mx::array(static_cast<void *>(p), std::move(shape), mx::float16,
                   [](void *q) { std::free(q); });
}

} // namespace

MLXQuantWeights ggufQ4_0ToMLXQuantized(const GGUFLoader &loader,
                                       const std::string &tensor_name) {
  const auto &info = loader.tensors().at(tensor_name);
  if (info.dtype != ::mlc::frontend::GGML_TYPE_Q4_0)
    throw std::runtime_error("ggufQ4_0ToMLXQuantized: not a Q4_0 tensor: " + tensor_name);
  if (info.shape.size() != 2)
    throw std::runtime_error("ggufQ4_0ToMLXQuantized: expected rank-2 tensor");

  // GGUF shape is innermost-first: shape[0]=in, shape[1]=out.
  int in_dim = static_cast<int>(info.shape[0]);
  int out_dim = static_cast<int>(info.shape[1]);
  if (in_dim % 32 != 0)
    throw std::runtime_error("ggufQ4_0ToMLXQuantized: in_dim must be multiple of 32");

  const int n_groups = in_dim / 32;
  const int u32_per_group = 4;   // 32 nibbles × 4 bits = 128 bits = 4 uint32
  const int u32_per_row = n_groups * u32_per_group;

  auto raw = loader.loadTensorData(info);
  // GGUF Q4_0: 18 bytes per group → row_bytes = n_groups * 18.
  const size_t row_bytes = static_cast<size_t>(n_groups) * 18u;
  if (raw.size() != row_bytes * static_cast<size_t>(out_dim))
    throw std::runtime_error("ggufQ4_0ToMLXQuantized: byte count mismatch");

  std::vector<uint32_t> w_q(static_cast<size_t>(out_dim) * u32_per_row);
  std::vector<uint16_t> scales(static_cast<size_t>(out_dim) * n_groups);
  std::vector<uint16_t> biases(static_cast<size_t>(out_dim) * n_groups);

  for (int r = 0; r < out_dim; ++r) {
    const uint8_t *row = raw.data() + static_cast<size_t>(r) * row_bytes;
    for (int g = 0; g < n_groups; ++g) {
      const uint8_t *blk = row + static_cast<size_t>(g) * 18u;
      uint16_t d_h;
      std::memcpy(&d_h, blk, sizeof(d_h));
      float d = fp16ToFp32(d_h);
      // MLX affine: dequant = n * scale + bias. We want (n - 8) * d, so
      // scale = d and bias = -8 * d.
      size_t sb_idx = static_cast<size_t>(r) * n_groups + g;
      scales[sb_idx] = d_h;                  // scale = d, already in fp16
      biases[sb_idx] = fp32ToFp16(-8.0f * d);

      // GGUF byte j (j=0..15) packs nibbles for elements j (low) and j+16
      // (high). MLX byte k (k=0..15) packs nibbles for elements 2k (low)
      // and 2k+1 (high). Decode 32 raw nibbles then repack.
      uint8_t nib[32];
      for (int j = 0; j < 16; ++j) {
        uint8_t byte = blk[2 + j];
        nib[j] = byte & 0x0f;
        nib[j + 16] = (byte >> 4) & 0x0f;
      }
      // MLX bits=4: each uint32 packs 8 consecutive elements in nibbles at
      // bits [0..3], [4..7], ..., [28..31] (verified against qdot in
      // mlx/backend/metal/kernels/quantized.h — reading the bytes as
      // uint16 puts element 4i+j in bits [4j..4j+3] of ws[i]).
      size_t base = static_cast<size_t>(r) * u32_per_row +
                    static_cast<size_t>(g) * u32_per_group;
      for (int u = 0; u < 4; ++u) {
        uint32_t v = 0;
        for (int e = 0; e < 8; ++e) {
          v |= static_cast<uint32_t>(nib[u * 8 + e]) << (e * 4);
        }
        w_q[base + u] = v;
      }
    }
  }

  mx::Shape w_shape{out_dim, u32_per_row};
  mx::Shape sb_shape{out_dim, n_groups};
  MLXQuantWeights out{
      allocU32MLX(std::move(w_q), w_shape),
      allocF16MLX(std::move(scales), sb_shape),
      allocF16MLX(std::move(biases), sb_shape),
      in_dim,
      out_dim,
  };
  mx::eval({out.w_q, out.scales, out.biases});
  return out;
}

} // namespace mlir::mlc::exec
