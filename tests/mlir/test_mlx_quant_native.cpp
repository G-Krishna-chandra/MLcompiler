// Native-format converter test: GGUF Q4_0 → MLX (w_q, scales, biases)
// for `mx::quantized_matmul(group_size=32, bits=4, "affine")`.
//
// The conversion is bit-exact (scale=d, bias=-8*d, nibble repack) so the
// dequantized value at each (row, col) must match GGUF Q4_0's
// `(n - 8) * d` to within fp16 rounding on the bias term. Cosine vs the
// fp32 reference should be ≥ 0.9999 on a single matmul — much tighter than
// R3's double-quantize bar of 0.99.

#include "compiler/frontends/gguf_loader.hpp"
#include "compiler/mlir/exec/MLXBuilder.h"
#include "compiler/mlir/exec/MLXQuantize.h"
#include "compiler/mlir/exec/Q4MatMul.h"
#include "compiler/mlir/exec/Quantize.h"

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>

namespace mx = mlx::core;

namespace {

constexpr const char *kModelPath =
    "/Users/kc/MLcompiler/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";

float cosineSim(const std::vector<float> &a, const std::vector<float> &b) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30));
}

} // namespace

TEST(MLXQuantNative, DequantMatchesGGUFExactly) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present";

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  // attn_q: [in, out] = [2048, 2048] in GGUF order.
  const std::string wname = "blk.0.attn_q.weight";
  auto w_ref = mlir::mlc::exec::dequantizeToF32(loader, wname);
  auto pkg = mlir::mlc::exec::ggufQ4_0ToMLXQuantized(loader, wname);
  ASSERT_EQ(pkg.in_dim, 2048);
  ASSERT_EQ(pkg.out_dim, 2048);

  // MLX `dequantize(w_q, scales, biases)` yields the dequantized weight in
  // the same [out, in] shape. Compare against the GGUF reference.
  auto w_deq = mx::dequantize(pkg.w_q, pkg.scales, pkg.biases,
                              /*group_size=*/32, /*bits=*/4,
                              /*mode=*/"affine");
  mx::eval(w_deq);
  auto w_deq_h = mlir::mlc::exec::mlxToF32(w_deq);

  ASSERT_EQ(w_deq_h.size(), w_ref.size());
  float max_err = 0.0f;
  float sum_sq = 0.0f, ref_sq = 0.0f;
  for (size_t i = 0; i < w_ref.size(); ++i) {
    float e = std::fabs(w_deq_h[i] - w_ref[i]);
    if (e > max_err) max_err = e;
    sum_sq += e * e;
    ref_sq += w_ref[i] * w_ref[i];
  }
  std::cout << "[dequant] max_err=" << max_err
            << " rmse=" << std::sqrt(sum_sq / w_ref.size())
            << " rel_rmse=" << std::sqrt(sum_sq / ref_sq) << std::endl;
  // The only source of difference is bias=fp16(-8*d) vs the implicit
  // -8*fp16(d) in GGUF Q4_0. fp16 rounding of a single multiply is well
  // below 1e-4 in absolute terms for typical scales.
  EXPECT_LT(max_err, 1e-3f);
}

TEST(MLXQuantNative, QuantizedMatmulVsFP32Reference) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present";

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  const std::string wname = "blk.0.attn_q.weight";
  auto w_f32 = mlir::mlc::exec::dequantizeToF32(loader, wname);
  const int OUT = 2048, IN = 2048;

  std::vector<float> x(IN);
  for (int i = 0; i < IN; ++i) x[i] = std::sin(0.01f * i);
  auto x_mlx = mlir::mlc::exec::fp32ToMLX(x, {1, IN});

  // fp32 reference using the GGUF Q4_0 dequant (the value our custom
  // kernel computes against).
  auto w_mlx = mlir::mlc::exec::fp32ToMLX(w_f32, {OUT, IN});
  auto y_ref = mx::matmul(x_mlx, mx::transpose(w_mlx));
  mx::eval(y_ref);
  auto y_ref_h = mlir::mlc::exec::mlxToF32(y_ref);

  // MLX native quantized path via our converter.
  auto pkg = mlir::mlc::exec::ggufQ4_0ToMLXQuantized(loader, wname);
  auto y_q = mx::quantized_matmul(x_mlx, pkg.w_q, pkg.scales, pkg.biases,
                                  /*transpose=*/true, /*group_size=*/32,
                                  /*bits=*/4, "affine");
  mx::eval(y_q);
  auto y_q_h = mlir::mlc::exec::mlxToF32(y_q);

  ASSERT_EQ(y_q_h.size(), y_ref_h.size());
  float cos = cosineSim(y_ref_h, y_q_h);
  std::cout << "[native quant matmul cos] " << cos << std::endl;
  EXPECT_GE(cos, 0.9999f);
}

TEST(MLXQuantNative, QuantizedMatmulVsCustomKernel) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present";

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  const std::string wname = "blk.0.attn_q.weight";
  const int IN = 2048;

  std::vector<float> x(IN);
  for (int i = 0; i < IN; ++i) x[i] = std::sin(0.01f * i);
  auto x_mlx_f32 = mlir::mlc::exec::fp32ToMLX(x, {1, IN});
  auto x_mlx = mx::astype(x_mlx_f32, mx::float16);

  // Custom Q4_0 kernel path: needs the raw bytes.
  auto w_bytes = mlir::mlc::exec::gufWeightToBytesMLX(loader, wname);
  auto y_custom = mlir::mlc::exec::q4_0_matmul(x_mlx, w_bytes, IN, 2048);
  mx::eval(y_custom);
  auto y_custom_h = mlir::mlc::exec::mlxToF32(y_custom);

  // MLX native quantized path.
  auto pkg = mlir::mlc::exec::ggufQ4_0ToMLXQuantized(loader, wname);
  auto y_q = mx::quantized_matmul(x_mlx, pkg.w_q, pkg.scales, pkg.biases,
                                  /*transpose=*/true, /*group_size=*/32,
                                  /*bits=*/4, "affine");
  mx::eval(y_q);
  auto y_q_h = mlir::mlc::exec::mlxToF32(y_q);

  ASSERT_EQ(y_q_h.size(), y_custom_h.size());
  float cos = cosineSim(y_custom_h, y_q_h);
  std::cout << "[custom vs native cos] " << cos << std::endl;
  EXPECT_GE(cos, 0.999f);
}
