// U1 validation:
//   1. The packed-layer CoreML model produces the same Q‖K‖V and
//      attn_output as MLX-GPU fp16 matmul on real TinyLlama layer-0
//      weights (cosine ≥ 0.999 per output).
//   2. Microbench three configurations and report wall-clock per token
//      of attention-projection work:
//
//        (a) 1 packed predict() on the ANELayer model.
//        (b) 2 individual predict() calls on two ANEMatMul models
//            (the T2 layout).
//        (c) 2 GPU Q4_0 kernel launches (the v3 baseline layout).
//
//      The U1 hypothesis: (a) ≪ (b). If true → boundary cost
//      amortizes over packed ops and U2 is worth chasing. If (a) ≈ (b)
//      or (a) > (b) → boundary cost is per-op, not per-call, and packing
//      doesn't buy us much.

#include "compiler/frontends/gguf_loader.hpp"
#include "compiler/mlir/exec/ANELayer.h"
#include "compiler/mlir/exec/ANEMatMul.h"
#include "compiler/mlir/exec/MLXBuilder.h"
#include "compiler/mlir/exec/Q4MatMul.h"
#include "compiler/mlir/exec/Quantize.h"

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

namespace mx = mlx::core;

namespace {

constexpr const char *kModelPath =
    "/Users/kc/MLcompiler/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
constexpr const char *kCacheDir = "/tmp/mlc-ane-layer-cache";

float cosineSim(const std::vector<float> &a, const std::vector<float> &b) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30));
}

// Load a Q4_0 weight, dequant to fp32, then transpose [out, in] (GGUF
// row-major) → [in, out] = [K, N] (CoreML matmul layout).
mx::array loadFP16WeightKN(const mlc::frontend::GGUFLoader &loader,
                            const std::string &name) {
  auto host = mlir::mlc::exec::dequantizeToF32(loader, name);
  const auto &info = loader.tensors().at(name);
  int OUT = static_cast<int>(info.shape[1]);
  int IN = static_cast<int>(info.shape[0]);
  std::vector<float> w_kn(static_cast<size_t>(IN) * OUT);
  for (int o = 0; o < OUT; ++o)
    for (int k = 0; k < IN; ++k)
      w_kn[static_cast<size_t>(k) * OUT + o] =
          host[static_cast<size_t>(o) * IN + k];
  auto m32 = mlir::mlc::exec::fp32ToMLX(w_kn, {IN, OUT});
  auto m16 = mx::astype(m32, mx::float16);
  mx::eval(m16);
  return m16;
}

} // namespace

TEST(ANELayer, MatchesMLXAndAmortizesBoundary) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present";

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  // TinyLlama layer 0 attention shapes:
  //   attn_q: [in=2048, out=2048]
  //   attn_k: [in=2048, out=256]   (GQA: 4 KV heads * 64 head_dim)
  //   attn_v: [in=2048, out=256]
  //   attn_output: [in=2048, out=2048]
  //
  // QKV concat → [in=2048, out=2560] (the FusedNormQKVMatMul shape).
  const std::string nq = "blk.0.attn_q.weight";
  const std::string nk = "blk.0.attn_k.weight";
  const std::string nv = "blk.0.attn_v.weight";
  const std::string no = "blk.0.attn_output.weight";

  auto wq = loadFP16WeightKN(loader, nq);
  auto wk = loadFP16WeightKN(loader, nk);
  auto wv = loadFP16WeightKN(loader, nv);
  auto wo = loadFP16WeightKN(loader, no);

  int K_qkv = wq.shape(0);
  int Nq = wq.shape(1), Nk = wk.shape(1), Nv = wv.shape(1);
  int N_qkv = Nq + Nk + Nv;
  int K_attn = wo.shape(0);
  int N_attn = wo.shape(1);

  auto w_qkv_concat = mx::concatenate({wq, wk, wv}, /*axis=*/1);
  mx::eval(w_qkv_concat);

  std::cout << "[shapes] K_qkv=" << K_qkv << " N_qkv=" << N_qkv
            << " K_attn=" << K_attn << " N_attn=" << N_attn << "\n";

  // --- (a) Bake the packed ANELayer model.
  std::string layer_pkg = mlir::mlc::exec::buildANELayerPackage(
      kCacheDir, "tinyllama_blk0_layer", /*M=*/1, K_qkv, N_qkv,
      K_attn, N_attn, w_qkv_concat, wo);
  std::cout << "[ane layer pkg] " << layer_pkg << "\n";
  mlir::mlc::exec::ANELayer ane_layer(layer_pkg, 1, K_qkv, N_qkv,
                                       K_attn, N_attn);

  // Deterministic inputs (need them now for the isolated bench).
  std::vector<float> xq_iso(K_qkv), xa_iso(K_attn);
  for (int i = 0; i < K_qkv; ++i) xq_iso[i] = std::sin(0.011f * i);
  for (int i = 0; i < K_attn; ++i) xa_iso[i] = std::cos(0.013f * i);
  auto xq_iso32 = mlir::mlc::exec::fp32ToMLX(xq_iso, {1, K_qkv});
  auto xa_iso32 = mlir::mlc::exec::fp32ToMLX(xa_iso, {1, K_attn});
  auto xq_iso16 = mx::astype(xq_iso32, mx::float16);
  auto xa_iso16 = mx::astype(xa_iso32, mx::float16);
  mx::eval({xq_iso16, xa_iso16});

  // (a') Packed-in-isolation timing — no other CoreML models loaded.
  // This isolates the packed model's per-call cost from any model-
  // cycling pressure that might appear once the (b) layout is loaded.
  ane_layer.predict(xq_iso16, xa_iso16);  // warm
  {
    const int iters = 200;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
      (void)ane_layer.predict(xq_iso16, xa_iso16);
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    std::cout << "[bench (a') packed iso] " << us
              << " us/iter (only ANELayer loaded)\n";
  }

  // --- (b) Bake two separate ANEMatMul models (the T2 layout).
  std::string qkv_pkg = mlir::mlc::exec::buildANEMatMulPackage(
      kCacheDir, "tinyllama_blk0_qkv_only", 1, K_qkv, N_qkv, w_qkv_concat);
  std::string o_pkg = mlir::mlc::exec::buildANEMatMulPackage(
      kCacheDir, "tinyllama_blk0_o_only", 1, K_attn, N_attn, wo);
  mlir::mlc::exec::ANEMatMul ane_qkv(qkv_pkg, 1, K_qkv, N_qkv);
  mlir::mlc::exec::ANEMatMul ane_o(o_pkg, 1, K_attn, N_attn);

  // Deterministic inputs.
  std::vector<float> xq_host(K_qkv), xa_host(K_attn);
  for (int i = 0; i < K_qkv; ++i) xq_host[i] = std::sin(0.011f * i);
  for (int i = 0; i < K_attn; ++i) xa_host[i] = std::cos(0.013f * i);
  auto xq32 = mlir::mlc::exec::fp32ToMLX(xq_host, {1, K_qkv});
  auto xa32 = mlir::mlc::exec::fp32ToMLX(xa_host, {1, K_attn});
  auto xq16 = mx::astype(xq32, mx::float16);
  auto xa16 = mx::astype(xa32, mx::float16);
  mx::eval({xq16, xa16});

  // --- Correctness: packed predict() vs MLX fp16 matmul reference.
  auto [y_qkv_a, y_o_a] = ane_layer.predict(xq16, xa16);
  auto y_qkv_mlx = mx::matmul(xq16, w_qkv_concat);
  auto y_o_mlx = mx::matmul(xa16, wo);
  mx::eval({y_qkv_mlx, y_o_mlx});

  auto host_a_qkv = mlir::mlc::exec::mlxToF32(y_qkv_a);
  auto host_a_o = mlir::mlc::exec::mlxToF32(y_o_a);
  auto host_m_qkv = mlir::mlc::exec::mlxToF32(y_qkv_mlx);
  auto host_m_o = mlir::mlc::exec::mlxToF32(y_o_mlx);

  float cos_qkv = cosineSim(host_a_qkv, host_m_qkv);
  float cos_o = cosineSim(host_a_o, host_m_o);
  std::cout << "[cos] qkv=" << cos_qkv << " o=" << cos_o << "\n";
  EXPECT_GE(cos_qkv, 0.999f);
  EXPECT_GE(cos_o, 0.999f);

  // --- Timing. Warm each path once before timing.
  const int iters = 200;
  ane_layer.predict(xq16, xa16);
  ane_qkv.predict(xq16);
  ane_o.predict(xa16);

  // (a) one packed predict per iter.
  {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
      (void)ane_layer.predict(xq16, xa16);
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    std::cout << "[bench (a) packed]    " << us << " us/iter (1 predict, 2 matmuls)\n";
  }

  // (b) two individual predicts per iter.
  {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      (void)ane_qkv.predict(xq16);
      (void)ane_o.predict(xa16);
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    std::cout << "[bench (b) 2x indiv]  " << us << " us/iter (2 predicts, 2 matmuls)\n";
  }

  // (c) two GPU Q4_0 kernel launches.
  // Convert weights back to Q4_0 bytes (GGUF [out, in] row-major) for the
  // kernel — matches what the executor does. Q4_0 bytes for blk.0 already
  // resident; use gufWeightToBytesMLX to load them.
  auto wq_bytes_concat = [&]() {
    auto bq = mlir::mlc::exec::gufWeightToBytesMLX(loader, nq);
    auto bk = mlir::mlc::exec::gufWeightToBytesMLX(loader, nk);
    auto bv = mlir::mlc::exec::gufWeightToBytesMLX(loader, nv);
    auto c = mx::concatenate({bq, bk, bv}, /*axis=*/0);
    mx::eval(c);
    return c;
  }();
  auto wo_bytes = mlir::mlc::exec::gufWeightToBytesMLX(loader, no);
  mx::eval(wo_bytes);

  {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      auto y1 = mlir::mlc::exec::q4_0_matmul(xq16, wq_bytes_concat,
                                              K_qkv, N_qkv);
      auto y2 = mlir::mlc::exec::q4_0_matmul(xa16, wo_bytes,
                                              K_attn, N_attn);
      mx::eval({y1, y2});
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    std::cout << "[bench (c) 2x q4gpu]  " << us << " us/iter (2 kernel launches w/ eval)\n";
  }
}
