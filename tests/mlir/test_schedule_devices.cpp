// S5 validation: the device scheduling pass picks ANE for narrow-output
// matmuls (attn_q/k/v/o) and GPU for the FFN, attention, norm, lm_head
// stack. Counts and per-op spot-checks against the TinyLlama IR.

#include "compiler/frontends/gguf_loader.hpp"
#include "compiler/mlir/dialect/MLCDialect.h"
#include "compiler/mlir/dialect/MLCOps.h"
#include "compiler/mlir/emit/GGUFToMLIR.h"
#include "compiler/mlir/passes/FuseNormMatMul.h"
#include "compiler/mlir/passes/FuseQKVMatMul.h"
#include "compiler/mlir/passes/ScheduleDevices.h"

#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

#include <filesystem>

namespace {

constexpr const char *kModelPath =
    "/Users/kc/MLcompiler/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";

// Look up the target_device on an op, return as the enum value.
::mlir::mlc::Device getDevice(::mlir::Operation *op) {
  auto attr = ::llvm::cast<::mlir::mlc::DeviceAttr>(
      op->getAttr("target_device"));
  return attr.getValue();
}

} // namespace

TEST(ScheduleDevices, TinyLlamaHeuristic) {
  if (!std::filesystem::exists(kModelPath))
    GTEST_SKIP() << "TinyLlama GGUF not present";

  mlc::frontend::GGUFLoader loader(kModelPath);
  ASSERT_TRUE(loader.load());

  mlir::MLIRContext ctx;
  auto module = mlir::mlc::emitMLIR(ctx, loader);
  // Run the fusion passes first; ScheduleDevices targets the post-fusion
  // op set (`fused_norm_qkv_matmul`, `fused_norm_matmul`, etc.).
  mlir::mlc::fuseNormMatMul(*module);
  mlir::mlc::fuseQKVMatMul(*module);

  unsigned changed = mlir::mlc::scheduleDevices(*module);
  std::cout << "[schedule] " << changed
            << " ops re-annotated\n";

  // Count by device per op type, on the post-fusion IR.
  unsigned ane_qkv = 0, gpu_qkv = 0;
  unsigned ane_matmul = 0, gpu_matmul = 0;
  unsigned gpu_ffn = 0, ane_ffn = 0;
  unsigned gpu_attn = 0;
  unsigned gpu_norm = 0;
  unsigned gpu_lmhead = 0;
  module->walk([&](::mlir::Operation *op) {
    if (::llvm::isa<::mlir::mlc::FusedNormQKVMatMulOp>(op)) {
      auto d = getDevice(op);
      d == ::mlir::mlc::Device::ANE ? ++ane_qkv : ++gpu_qkv;
    } else if (::llvm::isa<::mlir::mlc::MatMulOp>(op)) {
      auto d = getDevice(op);
      d == ::mlir::mlc::Device::ANE ? ++ane_matmul : ++gpu_matmul;
    } else if (::llvm::isa<::mlir::mlc::FeedForwardOp>(op)) {
      auto d = getDevice(op);
      d == ::mlir::mlc::Device::ANE ? ++ane_ffn : ++gpu_ffn;
    } else if (::llvm::isa<::mlir::mlc::AttentionOp>(op)) {
      if (getDevice(op) == ::mlir::mlc::Device::GPU) ++gpu_attn;
    } else if (::llvm::isa<::mlir::mlc::NormOp>(op)) {
      if (getDevice(op) == ::mlir::mlc::Device::GPU) ++gpu_norm;
    } else if (::llvm::isa<::mlir::mlc::LMHeadOp>(op)) {
      if (getDevice(op) == ::mlir::mlc::Device::GPU) ++gpu_lmhead;
    }
  });

  std::cout << "[schedule] QKV ANE=" << ane_qkv << " GPU=" << gpu_qkv
            << " | matmul ANE=" << ane_matmul << " GPU=" << gpu_matmul
            << " | FFN GPU=" << gpu_ffn
            << " | attn GPU=" << gpu_attn
            << " | norm GPU=" << gpu_norm
            << " | lm_head GPU=" << gpu_lmhead << "\n";

  // QKV: 22 fused ops, max-output = hidden = 2048 ≤ 4096 → all ANE.
  EXPECT_EQ(ane_qkv, 22u);
  EXPECT_EQ(gpu_qkv, 0u);

  // Remaining mlc.matmul: 22 attn_output ops (out=2048). Should all go ANE.
  EXPECT_EQ(ane_matmul, 22u);
  EXPECT_EQ(gpu_matmul, 0u);

  // FFN, attention, norm, lm_head all stay on GPU.
  EXPECT_EQ(gpu_ffn, 22u);
  EXPECT_EQ(gpu_attn, 22u);
  EXPECT_EQ(gpu_norm, 23u);  // 22 ffn_norm + 1 final_norm (attn_norms were absorbed)
  EXPECT_EQ(gpu_lmhead, 1u);
}
