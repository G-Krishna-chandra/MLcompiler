#include "compiler/mlir/passes/FuseNormMatMul.h"
#include "compiler/mlir/dialect/MLCDialect.h"
#include "compiler/mlir/dialect/MLCOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#include <vector>

namespace mlir::mlc {

unsigned fuseNormMatMul(ModuleOp module) {
  unsigned fused = 0;
  // First collect candidate matmuls; mutating during walk is awkward.
  std::vector<MatMulOp> candidates;
  module.walk([&](MatMulOp mm) {
    if (auto norm = mm.getX().getDefiningOp<NormOp>()) {
      // For now we only fuse when both ops live on the same target_device
      // — different device hints are signal that scheduling wanted them on
      // separate accelerators, and we shouldn't second-guess that.
      if (norm.getTargetDevice() == mm.getTargetDevice())
        candidates.push_back(mm);
    }
  });

  // Track norms whose users we've replaced so we can drop dead ones at the
  // end. A SetVector-ish behavior via repeated find is fine — at most ~45
  // distinct norms in TinyLlama.
  std::vector<NormOp> touched_norms;
  auto touch = [&](NormOp n) {
    for (auto e : touched_norms)
      if (e == n)
        return;
    touched_norms.push_back(n);
  };

  for (auto mm : candidates) {
    auto norm = mm.getX().getDefiningOp<NormOp>();
    if (!norm)
      continue;  // already erased by a prior iteration

    OpBuilder b(mm);
    auto fusedOp = b.create<FusedNormMatMulOp>(
        mm.getLoc(),
        /*result type=*/mm.getY().getType(),
        /*x=*/norm.getX(),
        /*gamma=*/norm.getGamma(),
        /*w=*/mm.getW(),
        /*target_device=*/mm.getTargetDeviceAttr(),
        /*epsilon=*/norm.getEpsilonAttr(),
        /*quant_format=*/mm.getQuantFormatAttr(),
        /*transpose_b=*/mm.getTransposeBAttr());
    mm.getY().replaceAllUsesWith(fusedOp.getY());
    mm.erase();
    touch(norm);
    ++fused;
  }

  // Erase any norm whose result is now unused (all its matmul consumers got
  // pulled into fused ops). Skip if other ops (feedforward, lm_head, etc.)
  // still depend on the norm.
  for (auto norm : touched_norms) {
    if (norm.getY().use_empty())
      norm.erase();
  }

  return fused;
}

} // namespace mlir::mlc
