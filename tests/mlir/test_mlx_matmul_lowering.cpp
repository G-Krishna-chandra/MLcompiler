// Q1 validation: an mlc.matmul lowered through the MLX walker should agree
// with a textbook triple-loop fp32 matmul on the dequantized Q4_0 weight.
//
// Both paths share the same dequantization (runtime::dequantizeRowTo), so
// any disagreement is in matmul math — wrong shape, wrong transpose, MLX
// kernel bug, etc. This catches the class of bugs we actually have at this
// point in the compiler-path build (not Q4_0 correctness, which the runtime
// already verifies against llama.cpp).

#include "compiler/frontends/gguf_loader.hpp"
#include "compiler/mlir/exec/MLXBuilder.h"
#include "compiler/mlir/exec/Quantize.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

constexpr const char *kModelPath =
    "/Users/kc/MLcompiler/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";

// Cosine similarity between two fp32 vectors. Same length assumed.
float cosineSim(const std::vector<float> &a, const std::vector<float> &b) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na  += static_cast<double>(a[i]) * a[i];
    nb  += static_cast<double>(b[i]) * b[i];
  }
  return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30));
}

// Reference matmul: y[m, n] = sum_k x[m, k] * w[k, n]
// Used as the gold for cross-checking the MLX path.
std::vector<float> matmulRef(const std::vector<float> &x, int M, int K,
                             const std::vector<float> &w, int N) {
  std::vector<float> y(static_cast<size_t>(M) * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      float xv = x[static_cast<size_t>(m) * K + k];
      for (int n = 0; n < N; ++n) {
        y[static_cast<size_t>(m) * N + n] +=
            xv * w[static_cast<size_t>(k) * N + n];
      }
    }
  }
  return y;
}

} // namespace

TEST(MLXMatmulLowering, AgreesWithCPUReferenceOnLayer0Q) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present at " << kModelPath;

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  const std::string wname = "blk.0.attn_q.weight";
  const auto &info = loader.tensors().at(wname);
  ASSERT_EQ(info.shape.size(), 2u);
  const int K = static_cast<int>(info.shape[0]);  // in dim
  const int N = static_cast<int>(info.shape[1]);  // out dim
  ASSERT_EQ(K, 2048);
  ASSERT_EQ(N, 2048);

  // Deterministic synthetic input: x[i] = sin(0.01 * i). Stable, non-trivial.
  const int M = 1;
  std::vector<float> x(K);
  for (int i = 0; i < K; ++i)
    x[i] = std::sin(0.01f * i);

  // Dequantize the weight once; both paths share this so we only test the
  // matmul, not the dequant.
  auto w_f32 = mlir::mlc::exec::dequantizeToF32(loader, wname);

  // Reference: triple-loop fp32 matmul.
  auto y_ref = matmulRef(x, M, K, w_f32, N);

  // MLX path: same fp32 buffers, lowered through MLX matmul.
  auto x_mlx = mlir::mlc::exec::fp32ToMLX(x, {M, K});
  auto w_mlx = mlir::mlc::exec::fp32ToMLX(w_f32, {K, N});
  auto y_mlx_arr = mlir::mlc::exec::matmul(x_mlx, w_mlx, /*transpose_b=*/false);
  auto y_mlx = mlir::mlc::exec::mlxToF32(y_mlx_arr);

  ASSERT_EQ(y_mlx.size(), y_ref.size());
  float cos = cosineSim(y_ref, y_mlx);
  EXPECT_GE(cos, 0.999f) << "cosine=" << cos;
  // Surface the actual cosine so the test log doubles as a regression record.
  std::cout << "[mlx-vs-cpu cosine] " << cos << std::endl;
}
