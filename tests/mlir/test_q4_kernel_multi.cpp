// U4 validation:
//   - q4_0_matmul_qkv produces the same Q, K, V tensors as
//     q4_0_matmul(concat) + 3x mx::slice on the same input/weight.
//   - q4_0_matmul_gate_up matches q4_0_matmul(concat) + 2x mx::slice.
//   Bit-exact for now (same kernel math, same accumulation order — the
//   only difference is the output store routing). We still allow a tiny
//   epsilon since fp16 reads can quirk between MLX runs.

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
#include <vector>

namespace mx = mlx::core;

namespace {

constexpr const char *kModelPath =
    "/Users/kc/MLcompiler/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";

float maxAbsDiff(const std::vector<float> &a, const std::vector<float> &b) {
  float m = 0.f;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

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

TEST(Q4KernelMulti, QKVMatchesSlicedPath) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present";

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  // TinyLlama layer 0 QKV: K=2048 in, Nq=2048, Nk=Nv=256 out each.
  auto wq = mlir::mlc::exec::gufWeightToBytesMLX(loader, "blk.0.attn_q.weight");
  auto wk = mlir::mlc::exec::gufWeightToBytesMLX(loader, "blk.0.attn_k.weight");
  auto wv = mlir::mlc::exec::gufWeightToBytesMLX(loader, "blk.0.attn_v.weight");
  auto wcat = mx::concatenate({wq, wk, wv}, /*axis=*/0);
  mx::eval(wcat);

  const int K = 2048, Nq = 2048, Nk = 256, Nv = 256;
  int seq = 2;  // exercise more than just M=1
  std::vector<float> xh(static_cast<size_t>(seq) * K);
  for (size_t i = 0; i < xh.size(); ++i) xh[i] = std::sin(0.005f * i + 0.1f);
  auto x32 = mlir::mlc::exec::fp32ToMLX(xh, {seq, K});
  auto x16 = mx::astype(x32, mx::float16);
  mx::eval(x16);

  // Old path: one matmul, three slices.
  auto y_concat = mlir::mlc::exec::q4_0_matmul(x16, wcat, K, Nq + Nk + Nv);
  auto q_old = mx::slice(y_concat, {0, 0}, {seq, Nq});
  auto k_old = mx::slice(y_concat, {0, Nq}, {seq, Nq + Nk});
  auto v_old = mx::slice(y_concat, {0, Nq + Nk}, {seq, Nq + Nk + Nv});
  mx::eval({q_old, k_old, v_old});

  // New path: one multi-output kernel.
  auto r = mlir::mlc::exec::q4_0_matmul_qkv(x16, wcat, K, Nq, Nk, Nv);
  mx::eval({r.q, r.k, r.v});

  auto q_old_h = mlir::mlc::exec::mlxToF32(q_old);
  auto k_old_h = mlir::mlc::exec::mlxToF32(k_old);
  auto v_old_h = mlir::mlc::exec::mlxToF32(v_old);
  auto q_new_h = mlir::mlc::exec::mlxToF32(r.q);
  auto k_new_h = mlir::mlc::exec::mlxToF32(r.k);
  auto v_new_h = mlir::mlc::exec::mlxToF32(r.v);

  std::cout << "[qkv] |maxDiff| q=" << maxAbsDiff(q_old_h, q_new_h)
            << " k=" << maxAbsDiff(k_old_h, k_new_h)
            << " v=" << maxAbsDiff(v_old_h, v_new_h) << "\n";
  std::cout << "[qkv] cosine q=" << cosineSim(q_old_h, q_new_h)
            << " k=" << cosineSim(k_old_h, k_new_h)
            << " v=" << cosineSim(v_old_h, v_new_h) << "\n";

  // Same kernel arithmetic, so identical down to floating noise.
  EXPECT_LT(maxAbsDiff(q_old_h, q_new_h), 1e-3f);
  EXPECT_LT(maxAbsDiff(k_old_h, k_new_h), 1e-3f);
  EXPECT_LT(maxAbsDiff(v_old_h, v_new_h), 1e-3f);
}

TEST(Q4KernelMulti, GateUpMatchesSlicedPath) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present";

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  // TinyLlama layer 0 FFN: K=2048 in, gate/up = 5632 each.
  auto wg = mlir::mlc::exec::gufWeightToBytesMLX(loader, "blk.0.ffn_gate.weight");
  auto wu = mlir::mlc::exec::gufWeightToBytesMLX(loader, "blk.0.ffn_up.weight");
  auto wcat = mx::concatenate({wg, wu}, /*axis=*/0);
  mx::eval(wcat);

  const int K = 2048, Ng = 5632, Nu = 5632;
  int seq = 2;
  std::vector<float> xh(static_cast<size_t>(seq) * K);
  for (size_t i = 0; i < xh.size(); ++i) xh[i] = std::cos(0.007f * i);
  auto x32 = mlir::mlc::exec::fp32ToMLX(xh, {seq, K});
  auto x16 = mx::astype(x32, mx::float16);
  mx::eval(x16);

  auto y_concat = mlir::mlc::exec::q4_0_matmul(x16, wcat, K, Ng + Nu);
  auto g_old = mx::slice(y_concat, {0, 0}, {seq, Ng});
  auto u_old = mx::slice(y_concat, {0, Ng}, {seq, Ng + Nu});
  mx::eval({g_old, u_old});

  auto r = mlir::mlc::exec::q4_0_matmul_gate_up(x16, wcat, K, Ng, Nu);
  mx::eval({r.gate, r.up});

  auto g_old_h = mlir::mlc::exec::mlxToF32(g_old);
  auto u_old_h = mlir::mlc::exec::mlxToF32(u_old);
  auto g_new_h = mlir::mlc::exec::mlxToF32(r.gate);
  auto u_new_h = mlir::mlc::exec::mlxToF32(r.up);

  std::cout << "[gu] |maxDiff| gate=" << maxAbsDiff(g_old_h, g_new_h)
            << " up=" << maxAbsDiff(u_old_h, u_new_h) << "\n";

  EXPECT_LT(maxAbsDiff(g_old_h, g_new_h), 1e-3f);
  EXPECT_LT(maxAbsDiff(u_old_h, u_new_h), 1e-3f);
}
