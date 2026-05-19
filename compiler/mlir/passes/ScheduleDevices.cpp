#include "compiler/mlir/passes/ScheduleDevices.h"
#include "compiler/mlir/dialect/MLCDialect.h"
#include "compiler/mlir/dialect/MLCOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::mlc {

namespace {

// Last-dim threshold separating "ANE-favored" from "GPU-favored" matmuls
// at batch=1 on M-series. From the S4 sweep:
//   N = 256  → ANE 49 µs vs GPU 178 µs (3.6x)
//   N = 2048 → ANE 55 µs vs GPU 286 µs (5.2x)
//   N = 5632 → ANE 367 µs vs GPU 344 µs (0.94x)
// Crossover sits ~4096. Conservative: pick 4096 inclusive on the ANE side.
constexpr int64_t kAneOutDimMax = 4096;

int64_t lastDim(Value v) {
  auto t = ::llvm::dyn_cast<RankedTensorType>(v.getType());
  if (!t || t.getRank() == 0) return -1;
  return t.getShape().back();
}

} // namespace

unsigned scheduleDevices(ModuleOp module) {
  unsigned changed = 0;
  module.walk([&](Operation *op) {
    auto ctx = op->getContext();
    auto pickDevice = [&](Device d) {
      auto newAttr = DeviceAttr::get(ctx, d);
      auto curAttr = op->getAttr("target_device");
      if (curAttr != newAttr) {
        op->setAttr("target_device", newAttr);
        ++changed;
      }
    };

    if (auto mm = ::llvm::dyn_cast<MatMulOp>(op)) {
      int64_t N = lastDim(mm.getY());
      pickDevice(N > 0 && N <= kAneOutDimMax ? Device::ANE : Device::GPU);
    } else if (auto fnm = ::llvm::dyn_cast<FusedNormMatMulOp>(op)) {
      int64_t N = lastDim(fnm.getY());
      pickDevice(N > 0 && N <= kAneOutDimMax ? Device::ANE : Device::GPU);
    } else if (auto qkv = ::llvm::dyn_cast<FusedNormQKVMatMulOp>(op)) {
      // The fused QKV op's effective output is Q ‖ K ‖ V along the last
      // dim. Use the largest of the three for the threshold check — if
      // any one falls in the GPU zone, the whole batched matmul does.
      int64_t maxN = std::max({lastDim(qkv.getQ()), lastDim(qkv.getK()),
                                lastDim(qkv.getV())});
      pickDevice(maxN > 0 && maxN <= kAneOutDimMax ? Device::ANE
                                                    : Device::GPU);
    } else if (::llvm::isa<FeedForwardOp, AttentionOp, NormOp, AddOp,
                            EmbeddingOp, LMHeadOp>(op)) {
      // FFN has a 5632-wide gate/up that put it in GPU territory per S4.
      // Attention needs the custom dense+RoPE kernel which only exists on
      // GPU. The remaining ops are either small or memory-bound; ANE
      // doesn't help.
      pickDevice(Device::GPU);
    }
  });
  return changed;
}

} // namespace mlir::mlc
