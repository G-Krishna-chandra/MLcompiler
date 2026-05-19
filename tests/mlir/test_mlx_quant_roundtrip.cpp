// R3 probe: round-tripping a GGUF Q4_0 weight through MLX's affine
// quantization and back via `mx::quantized_matmul`. The cosine vs an fp32
// reference using the *same* Q4_0 dequant lands at ≈ 0.99 on a single
// matmul, which is right at the threshold the session plan flagged as
// "skip and keep R1". The catch: this is essentially double-quantization
// (Q4_0 → fp32 → MLX-affine), and the per-matmul ~1% loss would compound
// across 22 layers.
//
// Conclusion: R1's resident fp16 path stays the default. A real
// quantized-matmul kernel that reads GGUF Q4_0 bytes directly (no
// re-quantize) is the right follow-up — that's a custom Metal kernel,
// not an MLX-builtin one-liner.

#include "compiler/frontends/gguf_loader.hpp"
#include "compiler/mlir/exec/MLXBuilder.h"
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

TEST(MLXQuantRoundtrip, QuantizedMatmulCloseToFP32OnLayer0Q) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present";

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  const std::string wname = "blk.0.attn_q.weight";
  auto w_f32 = mlir::mlc::exec::dequantizeToF32(loader, wname);
  // TinyLlama attn_q is [out, in] = [2048, 2048] after the GGUF dim-reverse.
  // We feed it to MLX with that same shape.
  const int OUT = 2048, IN = 2048;

  std::vector<float> x(IN);
  for (int i = 0; i < IN; ++i) x[i] = std::sin(0.01f * i);

  // fp32 reference: x @ w_logical = x @ w_stored^T
  auto x_mlx = mlir::mlc::exec::fp32ToMLX(x, {1, IN});
  auto w_mlx = mlir::mlc::exec::fp32ToMLX(w_f32, {OUT, IN});
  auto y_ref = mx::matmul(x_mlx, mx::transpose(w_mlx));
  mx::eval(y_ref);
  auto y_ref_h = mlir::mlc::exec::mlxToF32(y_ref);

  // Quantized path: mx::quantize the weight, then mx::quantized_matmul.
  // Group size 64 + 4 bits matches the standard MLX layout; "affine" mode
  // packs (q - bias) * scale ≈ original.
  auto quantized = mx::quantize(w_mlx, /*group_size=*/32, /*bits=*/4,
                                /*mode=*/"affine");
  ASSERT_GE(quantized.size(), 2u);
  auto w_q = quantized[0];
  auto scales = quantized[1];
  std::optional<mx::array> biases =
      quantized.size() >= 3 ? std::optional<mx::array>(quantized[2])
                            : std::nullopt;
  auto y_q = mx::quantized_matmul(x_mlx, w_q, scales, biases,
                                  /*transpose=*/true, /*group_size=*/32,
                                  /*bits=*/4, "affine");
  mx::eval(y_q);
  auto y_q_h = mlir::mlc::exec::mlxToF32(y_q);

  ASSERT_EQ(y_q_h.size(), y_ref_h.size());
  float cos = cosineSim(y_ref_h, y_q_h);
  std::cout << "[quant cos] " << cos << std::endl;
  // Quantization adds noise; 0.99 is a reasonable bar for layer-0 Q matmul.
  EXPECT_GE(cos, 0.99f);
}
