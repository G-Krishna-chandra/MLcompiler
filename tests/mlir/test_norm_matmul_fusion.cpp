// Q3 validation: the norm+matmul fusion pass shrinks the emitted TinyLlama
// IR and produces logits that match the unfused pipeline.

#include "compiler/frontends/gguf_loader.hpp"
#include "compiler/mlir/dialect/MLCDialect.h"
#include "compiler/mlir/dialect/MLCOps.h"
#include "compiler/mlir/emit/GGUFToMLIR.h"
#include "compiler/mlir/exec/MLIRExecutor.h"
#include "compiler/mlir/passes/FuseNormMatMul.h"

#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>

namespace {

constexpr const char *kModelPath =
    "/Users/kc/MLcompiler/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";

unsigned countOps(::mlir::ModuleOp m, ::llvm::StringRef name) {
  unsigned n = 0;
  m.walk([&](::mlir::Operation *op) {
    if (op->getName().getStringRef() == name)
      ++n;
  });
  return n;
}

float cosineSim(const std::vector<float> &a, const std::vector<float> &b) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na  += static_cast<double>(a[i]) * a[i];
    nb  += static_cast<double>(b[i]) * b[i];
  }
  return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30));
}

} // namespace

TEST(NormMatMulFusion, ShrinksTinyLlamaIRAndMatchesUnfused) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present";

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  mlir::MLIRContext ctx_unfused;
  auto unfused_mod = mlir::mlc::emitMLIR(ctx_unfused, loader);

  // Counts before fusion.
  unsigned norm_before   = countOps(*unfused_mod, "mlc.norm");
  unsigned matmul_before = countOps(*unfused_mod, "mlc.matmul");
  unsigned fused_before  = countOps(*unfused_mod, "mlc.fused_norm_matmul");
  EXPECT_EQ(norm_before, 45u);
  EXPECT_EQ(matmul_before, 88u);
  EXPECT_EQ(fused_before, 0u);

  // Emit again into a fresh context so we can compare execution.
  mlir::MLIRContext ctx_fused;
  auto fused_mod = mlir::mlc::emitMLIR(ctx_fused, loader);
  unsigned n_fused = mlir::mlc::fuseNormMatMul(*fused_mod);

  unsigned norm_after   = countOps(*fused_mod, "mlc.norm");
  unsigned matmul_after = countOps(*fused_mod, "mlc.matmul");
  unsigned fnm_after    = countOps(*fused_mod, "mlc.fused_norm_matmul");
  std::cout << "[fusion] fused=" << n_fused
            << " norm: " << norm_before << "->" << norm_after
            << " matmul: " << matmul_before << "->" << matmul_after
            << " fnm: " << fused_before << "->" << fnm_after << std::endl;

  // Each attn_norm (22 of them) has 3 matmul users; per layer, that's 4 ops
  // becoming 3 → save 1. 22 fusions × 3 matmuls each = 66 matmuls absorbed
  // into 66 fused ops, 22 norms erased.
  EXPECT_EQ(n_fused, 66u);
  EXPECT_EQ(norm_after, norm_before - 22);
  EXPECT_EQ(matmul_after, matmul_before - 66);
  EXPECT_EQ(fnm_after, 66u);

  // Execute both and compare last-token logits — must agree closely.
  mlir::mlc::exec::MLIRExecutor exec_a(*unfused_mod, loader);
  mlir::mlc::exec::MLIRExecutor exec_b(*fused_mod, loader);
  std::vector<int32_t> ids = {1, 2, 3, 4};
  std::vector<int32_t> pos = {0, 1, 2, 3};
  auto out_a = exec_a.run(ids, pos);
  auto out_b = exec_b.run(ids, pos);
  ASSERT_EQ(out_a.data.size(), out_b.data.size());
  float cos = cosineSim(out_a.data, out_b.data);
  std::cout << "[fusion cosine] " << cos << std::endl;
  EXPECT_GE(cos, 0.999f);
}
