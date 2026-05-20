// Test: full forward pass with MLC_FUSED_KERNELS=1 vs reference.
// Validates cosine >= 0.999 on prefill logits and first decode step.

#include "compiler/frontends/gguf_loader.hpp"
#include "compiler/mlir/emit/GGUFToMLIR.h"
#include "compiler/mlir/exec/MLIRExecutor.h"
#include "compiler/mlir/passes/FuseNormMatMul.h"
#include "compiler/mlir/passes/FuseQKVMatMul.h"
#include "mlir/IR/MLIRContext.h"
#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <numeric>

namespace {
const char* kModel = "/Users/kc/MLcompiler/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";

float cosineSim(const std::vector<float> &a, const std::vector<float> &b) {
  double d=0, na=0, nb=0;
  for (size_t i=0;i<a.size();++i){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
  return (float)(d/(std::sqrt(na)*std::sqrt(nb)+1e-30));
}
} // namespace

TEST(FusedExecutor, DecodeStep1CosineVsReference) {
  if (!std::filesystem::exists(kModel)) GTEST_SKIP() << "GGUF not present";
  mlc::frontend::GGUFLoader loader(kModel);
  ASSERT_TRUE(loader.load());

  setenv("MLC_FUSED_KERNELS", "0", 1);
  mlir::MLIRContext ctx_ref;
  auto mod_ref = mlir::mlc::emitMLIR(ctx_ref, loader);
  mlir::mlc::fuseNormMatMul(*mod_ref); mlir::mlc::fuseQKVMatMul(*mod_ref);
  mlir::mlc::exec::MLIRExecutor exec_ref(*mod_ref, loader);

  setenv("MLC_FUSED_KERNELS", "1", 1);
  mlir::MLIRContext ctx_fused;
  auto mod_fused = mlir::mlc::emitMLIR(ctx_fused, loader);
  mlir::mlc::fuseNormMatMul(*mod_fused); mlir::mlc::fuseQKVMatMul(*mod_fused);
  mlir::mlc::exec::MLIRExecutor exec_fused(*mod_fused, loader);
  unsetenv("MLC_FUSED_KERNELS");

  const std::vector<int32_t> prompt = {1, 450, 7483, 310, 3444, 338};
  std::vector<int32_t> pos(prompt.size()); std::iota(pos.begin(), pos.end(), 0);

  exec_ref.reset(); exec_fused.reset();
  auto pre_ref   = exec_ref.run(prompt, pos);
  auto pre_fused = exec_fused.run(prompt, pos);

  // Top-1 from reference prefill.
  int vocab = pre_ref.shape[1];
  const float *last = pre_ref.data.data() + (pre_ref.shape[0]-1)*vocab;
  int t0 = 0;
  for (int i=1;i<vocab;++i) if(last[i]>last[t0]) t0=i;

  auto dec_ref   = exec_ref.run({t0},   {(int)prompt.size()});
  auto dec_fused = exec_fused.run({t0}, {(int)prompt.size()});

  float cos_dec = cosineSim(dec_ref.data, dec_fused.data);
  int tok_ref=0, tok_fused=0;
  for(int i=1;i<vocab;++i){
    if(dec_ref.data[i]>dec_ref.data[tok_ref]) tok_ref=i;
    if(dec_fused.data[i]>dec_fused.data[tok_fused]) tok_fused=i;
  }
  std::printf("[fused] decode1 cos=%.6f tok_ref=%d tok_fused=%d\n",
              cos_dec, tok_ref, tok_fused);
  EXPECT_GE(cos_dec, 0.999f) << "Fused executor decode1 cosine";
  EXPECT_EQ(tok_ref, tok_fused) << "Fused executor decode1 token mismatch";
}

TEST(FusedExecutor, PrefillCosineVsReference) {
  if (!std::filesystem::exists(kModel)) GTEST_SKIP() << "GGUF not present";
  mlc::frontend::GGUFLoader loader(kModel);
  ASSERT_TRUE(loader.load());
  
  // Reference (no fused kernels) — needs new module per executor since
  // MLIRExecutor constructor reads env var at construction time.
  // Force env off for ref, on for fused.
  setenv("MLC_FUSED_KERNELS", "0", 1);
  mlir::MLIRContext ctx_ref;
  auto mod_ref = mlir::mlc::emitMLIR(ctx_ref, loader);
  mlir::mlc::fuseNormMatMul(*mod_ref);
  mlir::mlc::fuseQKVMatMul(*mod_ref);
  mlir::mlc::exec::MLIRExecutor exec_ref(*mod_ref, loader);

  setenv("MLC_FUSED_KERNELS", "1", 1);
  mlir::MLIRContext ctx_fused;
  auto mod_fused = mlir::mlc::emitMLIR(ctx_fused, loader);
  mlir::mlc::fuseNormMatMul(*mod_fused);
  mlir::mlc::fuseQKVMatMul(*mod_fused);
  mlir::mlc::exec::MLIRExecutor exec_fused(*mod_fused, loader);
  unsetenv("MLC_FUSED_KERNELS");

  const std::vector<int32_t> prompt = {1, 450, 7483, 310, 3444, 338};
  std::vector<int32_t> pos(prompt.size()); std::iota(pos.begin(), pos.end(), 0);

  exec_ref.reset(); exec_fused.reset();
  auto r_ref   = exec_ref.run(prompt, pos);
  auto r_fused = exec_fused.run(prompt, pos);

  ASSERT_EQ(r_ref.data.size(), r_fused.data.size());
  float cos = cosineSim(r_ref.data, r_fused.data);
  std::printf("[fused executor] prefill logit cosine=%.6f\n", cos);
  EXPECT_GE(cos, 0.999f) << "Fused executor prefill logit cosine vs reference";
}
