// Validates the fused multi-op Metal kernels (X1–X3).
//
// Each test compares the fused kernel output against the reference
// (separate ops) and requires cosine >= 0.999.
//
// Also measures the wall-clock time to confirm dispatch reduction is real
// (fused = 1 forced eval, separate = 2 forced evals).

#include "compiler/frontends/gguf_loader.hpp"
#include "compiler/mlir/exec/FusedKernels.h"
#include "compiler/mlir/exec/MLXBuilder.h"
#include "compiler/mlir/exec/Q4MatMul.h"
#include "compiler/mlir/exec/Quantize.h"

#include <mlx/array.h>
#include <mlx/fast.h>
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

float cosineSim(const std::vector<float> &a, const std::vector<float> &b) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += (double)a[i] * b[i];
    na  += (double)a[i] * a[i];
    nb  += (double)b[i] * b[i];
  }
  return (float)(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30));
}

// Build a random [seq, K] fp32 activation array.
mx::array randomAct(int seq, int K, float scale = 0.01f) {
  std::vector<float> buf(static_cast<size_t>(seq) * K);
  for (size_t i = 0; i < buf.size(); ++i)
    buf[i] = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f);
  mx::Shape sh{seq, K};
  float *p = static_cast<float *>(std::malloc(buf.size() * sizeof(float)));
  std::memcpy(p, buf.data(), buf.size() * sizeof(float));
  return mx::array(static_cast<void *>(p), std::move(sh), mx::float32,
                   [](void *q) { std::free(q); });
}

// Time a lazy MLX op: build, then forced eval.
template <typename Fn>
double timedMs(Fn fn) {
  auto arr = fn();
  auto t0 = std::chrono::steady_clock::now();
  mx::eval(arr);
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace

// ── X1: Fused norm+Q4_0 matmul ───────────────────────────────────────────────

TEST(FusedKernels, X1_NormQ4Matmul_Cosine) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present";

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  // Use attn_output weight: K=2048, N=2048.
  const std::string wname = "blk.0.attn_output.weight";
  const int K = 2048, N = 2048, SEQ = 4;

  auto w_bytes = mlir::mlc::exec::gufWeightToBytesMLX(loader, wname);
  mx::eval(w_bytes);

  // Load gamma (attn_norm for this layer is blk.0.attn_norm.weight).
  // Actually attn_output has no preceding norm in the graph — use ffn_norm
  // as the gamma tensor and construct a synthetic test input.
  // (We're validating math correctness, not model semantics.)
  const std::string gamma_name = "blk.0.ffn_norm.weight";
  auto gamma_mlx = mlir::mlc::exec::gufWeightToMLX(loader, gamma_name);  // [K] fp32
  gamma_mlx = mx::astype(gamma_mlx, mx::float32);
  mx::eval(gamma_mlx);

  auto x = randomAct(SEQ, K);
  mx::eval(x);

  // ── Reference: separate rms_norm + q4_0_matmul ──────────────────────────
  auto x_f16 = mx::astype(x, mx::float16);
  mx::eval(x_f16);
  auto norm_ref  = mx::fast::rms_norm(x, gamma_mlx, 1e-5f);
  mx::eval(norm_ref);
  auto y_ref_arr = mlir::mlc::exec::q4_0_matmul(norm_ref, w_bytes, K, N);
  mx::eval(y_ref_arr);
  auto y_ref = mlir::mlc::exec::mlxToF32(y_ref_arr);

  // ── Fused kernel ─────────────────────────────────────────────────────────
  auto y_fused_arr = mlir::mlc::exec::fusedNormQ4Matmul(x, gamma_mlx, w_bytes,
                                                         1e-5f, K, N);
  mx::eval(y_fused_arr);
  auto y_fused = mlir::mlc::exec::mlxToF32(y_fused_arr);

  ASSERT_EQ(y_ref.size(), y_fused.size());
  float cos = cosineSim(y_ref, y_fused);
  std::printf("[X1] SEQ=%d K=%d N=%d  cosine=%.6f\n", SEQ, K, N, cos);
  EXPECT_GE(cos, 0.999f) << "fused norm+Q4 cosine vs reference failed";
}

TEST(FusedKernels, X1_NormQ4Matmul_TimingOneDispatch) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present";

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  const std::string wname  = "blk.0.attn_output.weight";
  const std::string gnm    = "blk.0.ffn_norm.weight";
  const int K = 2048, N = 2048;

  auto w_bytes = mlir::mlc::exec::gufWeightToBytesMLX(loader, wname);
  auto gamma   = mx::astype(mlir::mlc::exec::gufWeightToMLX(loader, gnm), mx::float32);
  mx::eval({w_bytes, gamma});

  auto x = randomAct(1, K);
  mx::eval(x);

  // Warmup.
  for (int i = 0; i < 5; ++i) {
    auto r = mlir::mlc::exec::fusedNormQ4Matmul(x, gamma, w_bytes, 1e-5f, K, N);
    mx::eval(r);
    auto n = mx::fast::rms_norm(x, gamma, 1e-5f);
    auto m = mlir::mlc::exec::q4_0_matmul(n, w_bytes, K, N);
    mx::eval(m);
  }

  // Time separate (2 evals).
  const int REPS = 30;
  double sep_ms = 0.0;
  for (int i = 0; i < REPS; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    auto n = mx::fast::rms_norm(x, gamma, 1e-5f);
    mx::eval(n);
    auto m = mlir::mlc::exec::q4_0_matmul(n, w_bytes, K, N);
    mx::eval(m);
    auto t1 = std::chrono::steady_clock::now();
    sep_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  sep_ms /= REPS;

  // Time fused (1 eval).
  double fused_ms = 0.0;
  for (int i = 0; i < REPS; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    auto r = mlir::mlc::exec::fusedNormQ4Matmul(x, gamma, w_bytes, 1e-5f, K, N);
    mx::eval(r);
    auto t1 = std::chrono::steady_clock::now();
    fused_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  fused_ms /= REPS;

  std::printf("[X1 timing] separate=%.2f ms  fused=%.2f ms  speedup=%.2fx\n",
              sep_ms, fused_ms, sep_ms / fused_ms);
  // We expect fused to be faster (one fewer dispatch), but don't hard-assert
  // since Metal JIT timing can be noisy. Just log and check it's not slower.
  EXPECT_LT(fused_ms, sep_ms * 1.1)
      << "fused kernel should not be more than 10% slower than separate ops";
}

// ── X2: Fused norm+QKV ───────────────────────────────────────────────────────

TEST(FusedKernels, X2_NormQKV_Cosine) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present";

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  const int K = 2048, N_Q = 2048, N_K = 256, N_V = 256, SEQ = 2;

  // Load Q, K, V weight bytes and concat (same order as executor).
  auto wq = mlir::mlc::exec::gufWeightToBytesMLX(loader, "blk.0.attn_q.weight");
  auto wk = mlir::mlc::exec::gufWeightToBytesMLX(loader, "blk.0.attn_k.weight");
  auto wv = mlir::mlc::exec::gufWeightToBytesMLX(loader, "blk.0.attn_v.weight");
  auto w_cat = mx::concatenate({wq, wk, wv}, 0);
  mx::eval(w_cat);

  auto gamma = mx::astype(
      mlir::mlc::exec::gufWeightToMLX(loader, "blk.0.attn_norm.weight"),
      mx::float32);
  mx::eval(gamma);

  auto x = randomAct(SEQ, K);
  mx::eval(x);

  // Reference: separate norm + 3×matmul.
  auto norm_out = mx::fast::rms_norm(x, gamma, 1e-5f);
  mx::eval(norm_out);
  auto yq_ref = mlir::mlc::exec::q4_0_matmul(norm_out, wq, K, N_Q);
  auto yk_ref = mlir::mlc::exec::q4_0_matmul(norm_out, wk, K, N_K);
  auto yv_ref = mlir::mlc::exec::q4_0_matmul(norm_out, wv, K, N_V);
  mx::eval({yq_ref, yk_ref, yv_ref});

  // Fused kernel.
  auto fused = mlir::mlc::exec::fusedNormQ4MatmulQKV(x, gamma, w_cat, 1e-5f,
                                                      K, N_Q, N_K, N_V);
  mx::eval({fused.q, fused.k, fused.v});

  auto cos_q = cosineSim(mlir::mlc::exec::mlxToF32(yq_ref),
                         mlir::mlc::exec::mlxToF32(fused.q));
  auto cos_k = cosineSim(mlir::mlc::exec::mlxToF32(yk_ref),
                         mlir::mlc::exec::mlxToF32(fused.k));
  auto cos_v = cosineSim(mlir::mlc::exec::mlxToF32(yv_ref),
                         mlir::mlc::exec::mlxToF32(fused.v));
  std::printf("[X2] cosQ=%.6f cosK=%.6f cosV=%.6f\n", cos_q, cos_k, cos_v);
  EXPECT_GE(cos_q, 0.999f) << "X2 Q cosine failed";
  EXPECT_GE(cos_k, 0.999f) << "X2 K cosine failed";
  EXPECT_GE(cos_v, 0.999f) << "X2 V cosine failed";
}

// ── X3a: Fused norm+gate+up+silu ─────────────────────────────────────────────

TEST(FusedKernels, X3a_NormGateUpSilu_Cosine) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present";

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  const int K = 2048, FFN_DIM = 5632, SEQ = 2;

  auto wg = mlir::mlc::exec::gufWeightToBytesMLX(loader, "blk.0.ffn_gate.weight");
  auto wu = mlir::mlc::exec::gufWeightToBytesMLX(loader, "blk.0.ffn_up.weight");
  auto w_gu_cat = mx::concatenate({wg, wu}, 0);
  mx::eval(w_gu_cat);

  auto gamma = mx::astype(
      mlir::mlc::exec::gufWeightToMLX(loader, "blk.0.ffn_norm.weight"),
      mx::float32);
  mx::eval(gamma);

  auto x = randomAct(SEQ, K);
  mx::eval(x);

  // Reference.
  auto norm_out = mx::fast::rms_norm(x, gamma, 1e-5f);
  mx::eval(norm_out);
  auto gate_raw = mlir::mlc::exec::q4_0_matmul(norm_out, wg, K, FFN_DIM);
  auto up_raw   = mlir::mlc::exec::q4_0_matmul(norm_out, wu, K, FFN_DIM);
  auto gate_ref = mx::multiply(gate_raw, mx::sigmoid(gate_raw));
  auto h_ref    = mx::multiply(gate_ref, up_raw);
  mx::eval(h_ref);

  // Fused kernel.
  auto h_fused = mlir::mlc::exec::fusedNormQ4GateUpSilu(x, gamma, w_gu_cat,
                                                         1e-5f, K, FFN_DIM);
  mx::eval(h_fused);

  float cos = cosineSim(mlir::mlc::exec::mlxToF32(h_ref),
                        mlir::mlc::exec::mlxToF32(h_fused));
  std::printf("[X3a] SEQ=%d K=%d FFN=%d  cosine=%.6f\n", SEQ, K, FFN_DIM, cos);
  EXPECT_GE(cos, 0.999f) << "X3a fused norm+gate+up+silu cosine failed";
}

// ── X3b: Fused Q4_0 matmul + residual add ────────────────────────────────────

TEST(FusedKernels, X3b_Q4MatmulResidual_Cosine) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present";

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  const int K_FFN = 5632, N = 2048, SEQ = 2;

  auto w_bytes = mlir::mlc::exec::gufWeightToBytesMLX(loader,
                                                        "blk.0.ffn_down.weight");
  mx::eval(w_bytes);

  auto h        = randomAct(SEQ, K_FFN, 0.001f);
  auto residual = randomAct(SEQ, N, 0.01f);
  mx::eval({h, residual});

  // Reference.
  auto proj_ref = mlir::mlc::exec::q4_0_matmul(h, w_bytes, K_FFN, N);
  auto y_ref    = mx::add(residual, proj_ref);
  mx::eval(y_ref);

  // Fused.
  auto y_fused = mlir::mlc::exec::fusedQ4MatmulResidual(h, residual,
                                                         w_bytes, K_FFN, N);
  mx::eval(y_fused);

  float cos = cosineSim(mlir::mlc::exec::mlxToF32(y_ref),
                        mlir::mlc::exec::mlxToF32(y_fused));
  std::printf("[X3b] SEQ=%d K_FFN=%d N=%d  cosine=%.6f\n", SEQ, K_FFN, N, cos);
  EXPECT_GE(cos, 0.999f) << "X3b fused Q4 matmul+residual cosine failed";
}
