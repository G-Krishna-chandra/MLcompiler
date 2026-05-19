// T1 validation:
//   1. ANEMatMul produces fp32-cosine ≥ 0.999 against the MLX-GPU matmul
//      on TinyLlama's blk.0.attn_q.weight.
//   2. Back-to-back predict() inside a tight loop reports realistic
//      per-call latency — the headline number that decides whether
//      the GPU+ANE hybrid in T2 has a chance.

#include "compiler/frontends/gguf_loader.hpp"
#include "compiler/mlir/exec/ANEMatMul.h"
#include "compiler/mlir/exec/MLXBuilder.h"
#include "compiler/mlir/exec/Quantize.h"

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <filesystem>

namespace mx = mlx::core;

namespace {

constexpr const char *kModelPath =
    "/Users/kc/MLcompiler/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
constexpr const char *kAneCacheDir = "/tmp/mlc-ane-cache";

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

TEST(ANEMatMul, OnesXOnesGivesK) {
  const int K = 2048, N = 2048;
  std::vector<float> w(static_cast<size_t>(K) * N, 1.0f);
  auto w_mlx_fp32 = mlir::mlc::exec::fp32ToMLX(w, {K, N});
  auto w_fp16 = mx::astype(w_mlx_fp32, mx::float16);
  mx::eval(w_fp16);
  std::string pkg = mlir::mlc::exec::buildANEMatMulPackage(
      "/tmp/mlc-ane-test-ones", "ones", 1, K, N, w_fp16);
  mlir::mlc::exec::ANEMatMul ane(pkg, 1, K, N);
  std::vector<float> x(K, 1.0f);
  auto x_fp32 = mlir::mlc::exec::fp32ToMLX(x, {1, K});
  auto x_fp16 = mx::astype(x_fp32, mx::float16);
  mx::eval(x_fp16);
  auto y = ane.predict(x_fp16);
  auto y_host = mlir::mlc::exec::mlxToF32(y);
  std::cout << "[ones test] y[0]=" << y_host[0] << " y[1]=" << y_host[1]
            << " y[N-1]=" << y_host[N - 1] << " expected " << K << "\n";
  // K=2048 is exactly representable in fp16. Sum of 2048 ones should be 2048.
  EXPECT_NEAR(y_host[0], static_cast<float>(K), 1.0f);
}

TEST(ANEMatMul, MatchesMLXAndMeasuresOverhead) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present";

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  const std::string wname = "blk.0.attn_q.weight";
  const auto &info = loader.tensors().at(wname);
  ASSERT_EQ(info.shape.size(), 2u);
  int K = static_cast<int>(info.shape[0]);  // 2048 (in)
  int N = static_cast<int>(info.shape[1]);  // 2048 (out)
  int M = 1;

  // Dequant Q4_0 → fp32 → upload as fp32 mx::array → cast to fp16.
  auto w32_host = mlir::mlc::exec::dequantizeToF32(loader, wname);
  // GGUF row layout is [out, in] in numpy terms; we need to bake the
  // CoreML weight as [K, N]=[in, out] for x[M,K] @ w[K,N]. So we
  // transpose during the cast.
  std::vector<float> w_kn(static_cast<size_t>(K) * N);
  for (int o = 0; o < N; ++o)
    for (int k = 0; k < K; ++k)
      w_kn[static_cast<size_t>(k) * N + o] =
          w32_host[static_cast<size_t>(o) * K + k];
  auto w_mlx_kn = mlir::mlc::exec::fp32ToMLX(w_kn, {K, N});
  auto w_mlx_kn_fp16 = mx::astype(w_mlx_kn, mx::float16);
  mx::eval(w_mlx_kn_fp16);
  // Dump first few values for cross-check with the python script's print.
  auto host_check = mlir::mlc::exec::mlxToF32(w_mlx_kn_fp16);
  std::cout << "[debug] w_kn fp16-readback row0[0..3]=" << host_check[0]
            << "," << host_check[1] << "," << host_check[2]
            << " row1[0..3]=" << host_check[N] << "," << host_check[N+1]
            << "," << host_check[N+2] << "\n";

  // Bake the model. First call also runs coremltools — slow.
  std::string pkg_path = mlir::mlc::exec::buildANEMatMulPackage(
      kAneCacheDir, "tinyllama_blk0_attn_q_2048x2048", M, K, N,
      w_mlx_kn_fp16);
  std::cout << "[ane pkg] " << pkg_path << std::endl;
  mlir::mlc::exec::ANEMatMul ane(pkg_path, M, K, N);

  // Deterministic input.
  std::vector<float> x(K);
  for (int i = 0; i < K; ++i) x[i] = std::sin(0.01f * i);
  auto x_mlx_fp32 = mlir::mlc::exec::fp32ToMLX(x, {M, K});
  auto x_mlx = mx::astype(x_mlx_fp32, mx::float16);
  mx::eval(x_mlx);

  // ANE path.
  auto y_ane = ane.predict(x_mlx);
  auto y_ane_host = mlir::mlc::exec::mlxToF32(y_ane);

  // MLX/GPU reference: x @ w_kn (both fp16).
  auto y_mlx = mx::matmul(x_mlx, w_mlx_kn_fp16);
  mx::eval(y_mlx);
  auto y_mlx_host = mlir::mlc::exec::mlxToF32(y_mlx);

  std::cout << "[debug] ane y[0..3]=" << y_ane_host[0] << "," << y_ane_host[1] << "," << y_ane_host[2] << "\n";
  std::cout << "[debug] mlx y[0..3]=" << y_mlx_host[0] << "," << y_mlx_host[1] << "," << y_mlx_host[2] << "\n";

  ASSERT_EQ(y_ane_host.size(), y_mlx_host.size());
  float cos = cosineSim(y_ane_host, y_mlx_host);
  std::cout << "[ane cos] " << cos << std::endl;
  EXPECT_GE(cos, 0.999f);

  // Back-to-back latency. Mimics the realistic "tight loop of
  // predict calls" the hybrid forward pass would do.
  ane.predict(x_mlx);  // warmup
  const int iters = 200;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i)
    (void)ane.predict(x_mlx);
  auto t1 = std::chrono::steady_clock::now();
  double avg_us =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  std::cout << "[ane back-to-back] M=" << M << " K=" << K << " N=" << N
            << " iters=" << iters << " avg=" << avg_us << " us/call\n";
}
