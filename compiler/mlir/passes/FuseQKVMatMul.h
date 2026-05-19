#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace mlir::mlc {

// Group sets of three `fused_norm_matmul` ops sharing the same `x`,
// `gamma`, and op attributes into a single `fused_norm_qkv_matmul`.
// Replaces 3 separate norm+matmul ops (each recomputing the norm) with
// one shared norm + one batched matmul + a 3-way split. The weight
// concatenation happens at executor cache time, so this op is "free" at
// dispatch — three small kernel launches collapse into one large one.
//
// Returns the number of groups fused.
unsigned fuseQKVMatMul(ModuleOp module);

} // namespace mlir::mlc
