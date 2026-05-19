// Q2 validation: end-to-end compiler-path forward pass on TinyLlama. Feeds
// a short synthetic prompt, walks the entire 223-op emitted IR through MLX,
// and checks the resulting logits are finite and well-shaped.
//
// We do NOT compare against the runtime here — runtime instrumentation is
// out of scope for this commit, and Q1 already validated matmul math
// against a textbook reference. The Top-1 = "Paris" sanity check lives in
// the Q4 driver, which adds tokenization.

#include "compiler/frontends/gguf_loader.hpp"
#include "compiler/mlir/emit/GGUFToMLIR.h"
#include "compiler/mlir/exec/MLIRExecutor.h"

#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>

namespace {

constexpr const char *kModelPath =
    "/Users/kc/MLcompiler/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";

} // namespace

TEST(CompilerPathForward, TinyLlamaProducesFiniteLogits) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present";

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  mlir::MLIRContext ctx;
  auto module = mlir::mlc::emitMLIR(ctx, loader);
  ASSERT_TRUE(module && *module);

  mlir::mlc::exec::MLIRExecutor exec(*module, loader);

  // Synthetic 4-token prompt. Real tokenization (and the Top-1 check) lives
  // in the Q4 driver — the goal here is just to prove the walker runs the
  // whole 223-op graph end-to-end on real weights without numeric explosion.
  std::vector<int32_t> ids       = {1, 2, 3, 4};
  std::vector<int32_t> positions = {0, 1, 2, 3};

  auto out = exec.run(ids, positions);

  // Shape: [seq, vocab]. For TinyLlama vocab = 32000.
  ASSERT_EQ(out.shape.size(), 2u);
  EXPECT_EQ(out.shape[0], 4);
  EXPECT_EQ(out.shape[1], 32000);
  ASSERT_EQ(out.data.size(), static_cast<size_t>(4) * 32000);

  // Every logit finite. NaN/Inf would indicate softmax/norm/RoPE breakage.
  size_t n_finite = 0;
  float lmax = -1e30f, lmin = 1e30f;
  for (float v : out.data) {
    if (std::isfinite(v)) {
      ++n_finite;
      lmax = std::max(lmax, v);
      lmin = std::min(lmin, v);
    }
  }
  EXPECT_EQ(n_finite, out.data.size());
  EXPECT_GT(lmax, lmin) << "all logits identical — graph likely zero'd out";
  // Sanity: logits magnitudes should be in the single- to double-digit range
  // for a healthy small LLM. Anything wildly outside hints at a scale bug.
  EXPECT_LT(lmax, 1000.0f);
  EXPECT_GT(lmin, -1000.0f);
  std::cout << "[compiler-path logits] min=" << lmin << " max=" << lmax
            << std::endl;
}
