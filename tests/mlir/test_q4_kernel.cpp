// S1 validation: a Q4_0 matvec computed by the custom Metal kernel agrees
// with a CPU triple-loop matmul on the same Q4_0-dequant weight.

#include "compiler/frontends/gguf_loader.hpp"
#include "compiler/mlir/exec/MLXBuilder.h"
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

TEST(Q4Kernel, AgreesWithCPUDequantOnLayer0Q) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present";

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  const std::string wname = "blk.0.attn_q.weight";
  const auto &info = loader.tensors().at(wname);
  // GGUF shape is innermost-first; numpy order is [out, in].
  ASSERT_EQ(info.shape.size(), 2u);
  const int IN = static_cast<int>(info.shape[0]);   // 2048
  const int OUT = static_cast<int>(info.shape[1]);  // 2048

  // Deterministic synthetic input.
  std::vector<float> x(IN);
  for (int i = 0; i < IN; ++i) x[i] = std::sin(0.01f * i);

  // CPU reference: dequant Q4_0 to fp32, then triple-loop.
  auto w_f32 = mlir::mlc::exec::dequantizeToF32(loader, wname);
  std::vector<float> y_ref(OUT, 0.0f);
  for (int o = 0; o < OUT; ++o) {
    for (int k = 0; k < IN; ++k) {
      // Stored as [out=row, in=col] row-major; w_f32 idx = o*IN + k.
      y_ref[o] += x[k] * w_f32[o * IN + k];
    }
  }

  // GPU path: raw Q4_0 bytes + the custom kernel. fp16 to match the v3
  // production path.
  auto w_bytes = mlir::mlc::exec::gufWeightToBytesMLX(loader, wname);
  auto x_mlx_fp32 = mlir::mlc::exec::fp32ToMLX(x, {1, IN});
  auto x_mlx = mx::astype(x_mlx_fp32, mx::float16);
  mx::eval(x_mlx);
  auto y_mlx = mlir::mlc::exec::q4_0_matmul(x_mlx, w_bytes, IN, OUT);
  auto y_host = mlir::mlc::exec::mlxToF32(y_mlx);

  ASSERT_EQ(y_host.size(), y_ref.size());
  float cos = cosineSim(y_ref, y_host);
  std::cout << "[q4 kernel cos] " << cos << std::endl;
  EXPECT_GE(cos, 0.999f);
}
