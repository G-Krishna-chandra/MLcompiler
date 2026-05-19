#include "compiler/mlir/passes/FuseQKVMatMul.h"
#include "compiler/mlir/dialect/MLCDialect.h"
#include "compiler/mlir/dialect/MLCOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

#include <vector>

namespace mlir::mlc {

namespace {

// Walk the function once, collecting groups of fused_norm_matmul ops keyed
// by their shared (x, gamma) pair. The QKV-projection pattern is exactly
// three ops with the same operands in walk order = q, k, v. Any other
// arity stays alone.
struct Group {
  Value x;
  Value gamma;
  ::llvm::SmallVector<FusedNormMatMulOp, 3> ops;
};

} // namespace

unsigned fuseQKVMatMul(ModuleOp module) {
  unsigned fused = 0;
  module.walk([&](::mlir::func::FuncOp fn) {
    ::std::vector<Group> groups;
    // Linear scan is fine — TinyLlama has ~66 fused_norm_matmul ops after
    // the R3 pass.
    fn.walk([&](FusedNormMatMulOp op) {
      for (auto &g : groups) {
        if (g.x == op.getX() && g.gamma == op.getGamma() &&
            g.ops.size() < 3) {
          g.ops.push_back(op);
          return;
        }
      }
      groups.push_back({op.getX(), op.getGamma(), {op}});
    });

    for (auto &g : groups) {
      if (g.ops.size() != 3) continue;
      // Sanity: all three must share device + epsilon + quant_format. The
      // pass shouldn't fuse a mixed-device cluster; that would defeat the
      // future scheduling pass.
      auto q_op = g.ops[0];
      auto k_op = g.ops[1];
      auto v_op = g.ops[2];
      if (q_op.getTargetDevice() != k_op.getTargetDevice() ||
          q_op.getTargetDevice() != v_op.getTargetDevice())
        continue;
      if (q_op.getEpsilon() != k_op.getEpsilon() ||
          q_op.getEpsilon() != v_op.getEpsilon())
        continue;
      if (q_op.getQuantFormat() != k_op.getQuantFormat() ||
          q_op.getQuantFormat() != v_op.getQuantFormat())
        continue;
      if (q_op.getTransposeB() != k_op.getTransposeB() ||
          q_op.getTransposeB() != v_op.getTransposeB())
        continue;

      OpBuilder b(q_op);
      auto new_op = b.create<FusedNormQKVMatMulOp>(
          q_op.getLoc(),
          /*result_types=*/::mlir::TypeRange{q_op.getY().getType(),
                                              k_op.getY().getType(),
                                              v_op.getY().getType()},
          /*x=*/g.x,
          /*gamma=*/g.gamma,
          /*w_q=*/q_op.getW(),
          /*w_k=*/k_op.getW(),
          /*w_v=*/v_op.getW(),
          /*target_device=*/q_op.getTargetDeviceAttr(),
          /*epsilon=*/q_op.getEpsilonAttr(),
          /*quant_format=*/q_op.getQuantFormatAttr(),
          /*transpose_b=*/q_op.getTransposeBAttr());
      q_op.getY().replaceAllUsesWith(new_op.getQ());
      k_op.getY().replaceAllUsesWith(new_op.getK());
      v_op.getY().replaceAllUsesWith(new_op.getV());
      q_op.erase();
      k_op.erase();
      v_op.erase();
      ++fused;
    }
  });
  return fused;
}

} // namespace mlir::mlc
